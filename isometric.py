"""Leitura de isométrico de tubulação (PDF vetorial) e geração da casca 3D da linha.

Fluxo: traço mais grosso do desenho = linha de centro -> direções isométricas (30°/90°/150°)
mapeadas para N/E/Elevação pela rosa dos ventos -> comprimentos pelas coordenadas das caixas
(N/E/EL) ou, na falta delas, pelas cotas -> curvas 90° com raio 1,5·DN -> superfície de casca
(STEP) varrida ao longo da linha de centro.
"""
import math
import os
import re
import tempfile
from dataclasses import dataclass, field

import gmsh
import numpy as np
import pymupdf

from mesher import _LOCK

BEND_FACTOR = 1.5  # raio de curvatura = fator * DN
IN = 25.4

# NPS (pol) -> diâmetro externo (mm), ASME B36.10
OD_TABLE = {0.5: 21.3, 0.75: 26.7, 1.0: 33.4, 1.5: 48.3, 2.0: 60.3, 3.0: 88.9, 4.0: 114.3,
            6.0: 168.3, 8.0: 219.1, 10.0: 273.1, 12.0: 323.9, 14.0: 355.6, 16.0: 406.4,
            18.0: 457.0, 20.0: 508.0, 24.0: 610.0}
# NPS -> espessura SCH 40 (mm)
SCH40 = {0.5: 2.77, 0.75: 2.87, 1.0: 3.38, 1.5: 3.68, 2.0: 3.91, 3.0: 5.49, 4.0: 6.02,
         6.0: 7.11, 8.0: 8.18, 10.0: 9.27, 12.0: 10.31}

# Eixos isométricos no desenho (ângulo no plano da página, anti-horário, y para cima)
ISO_ANGLES = {"A+": 30.0, "B+": 150.0, "Z+": 90.0, "A-": 210.0, "B-": 330.0, "Z-": 270.0}
ANGLE_TOL = 8.0


@dataclass
class Segment:
    axis: str  # "N", "E" ou "U" (sinal em `sign`)
    sign: int
    length: float  # mm
    source: str  # origem do comprimento


@dataclass
class IsoResult:
    segments: list[Segment]
    nps: float | None
    wall: float | None
    messages: list[str] = field(default_factory=list)
    page_png: bytes = b""
    pressure_op: float | None = None  # MPa (manométrica), condição de operação
    pressure_design: float | None = None  # MPa (manométrica), condição de projeto
    temp_op: float | None = None  # °C, operação
    temp_design: float | None = None  # °C, projeto
    @property
    def od(self) -> float | None:
        return OD_TABLE.get(self.nps) if self.nps else None


def _num(s: str) -> float:
    s = s.strip().replace("+", "")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    return float(s)


def _ang(dx: float, dy_down: float) -> float:
    return math.degrees(math.atan2(-dy_down, dx)) % 360


def _ang_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def _centerline(page) -> list[tuple[float, float]]:
    lines = []
    for d in page.get_drawings():
        w = d.get("width") or 0
        for it in d["items"]:
            if it[0] == "l":
                lines.append((w, (it[1].x, it[1].y), (it[2].x, it[2].y)))
    if not lines:
        raise ValueError("O PDF não contém desenho vetorial (apenas imagem?).")
    wmax = max(w for w, _, _ in lines)
    thick = [(a, b) for w, a, b in lines if w >= 0.8 * wmax and w >= 1.5]
    if not thick:
        raise ValueError("Não encontrei o traço grosso da linha de centro no PDF.")
    # Encadeia os segmentos pelas extremidades coincidentes.
    pts: list[tuple[float, float]] = []

    def key(p):
        return (round(p[0], 0), round(p[1], 0))

    adj: dict = {}
    for a, b in thick:
        adj.setdefault(key(a), []).append((key(b), b))
        adj.setdefault(key(b), []).append((key(a), a))
        pts += [a, b]
    coord = {key(p): p for p in pts}
    if any(len(v) > 2 for v in adj.values()):
        raise ValueError("A linha possui ramais; só linhas sem derivação são suportadas.")
    ends = [k for k, v in adj.items() if len(v) == 1]
    if len(ends) != 2:
        raise ValueError("A linha de centro deve ser uma única cadeia aberta.")
    # começa pela extremidade mais baixa na página (S1 costuma ser a origem do fluxo)
    order = [ends[0]]
    prev = None
    while True:
        nxt = [k for k, _ in adj[order[-1]] if k != prev]
        if not nxt:
            break
        prev = order[-1]
        order.append(nxt[0])
    verts = [coord[k] for k in order]
    # remove vértices colineares
    out = [verts[0]]
    for i in range(1, len(verts) - 1):
        a = _ang(verts[i][0] - out[-1][0], verts[i][1] - out[-1][1])
        b = _ang(verts[i + 1][0] - verts[i][0], verts[i + 1][1] - verts[i][1])
        if _ang_diff(a, b) > 1.0:
            out.append(verts[i])
    out.append(verts[-1])
    return out


def _axis_of(angle: float) -> str:
    for name, a in ISO_ANGLES.items():
        if _ang_diff(angle, a) <= ANGLE_TOL:
            return name
    raise ValueError(f"Trecho a {angle:.0f}° não está sobre um eixo isométrico (30°/90°/150°).")


def _compass(page, msgs):
    """Mapeia os eixos isométricos A (30°) e B (150°) para N/E usando a rosa dos ventos."""
    letters = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            t = "".join(s["text"] for s in ln["spans"]).strip()
            if t in ("N", "E"):
                r = ln["bbox"]
                letters.append((t, (r[0] + r[2]) / 2, (r[1] + r[3]) / 2))
    circle = None
    for d in page.get_drawings():
        r = d["rect"]
        if (60 < r.width < 250 and abs(r.width - r.height) < 3
                and sum(1 for it in d["items"] if it[0] == "c") >= 4):
            if circle is None or r.width > circle.width:
                circle = r
    if circle is None or not letters:
        return None
    cx, cy = (circle.x0 + circle.x1) / 2, (circle.y0 + circle.y1) / 2
    best = {}
    for t, x, y in letters:
        dist = math.hypot(x - cx, y - cy)
        if dist < circle.width * 1.2 and (t not in best or dist < best[t][0]):
            best[t] = (dist, _ang(x - cx, y - cy))
    if set(best) != {"N", "E"}:
        return None
    return {t: best[t][1] for t in best}


def _nearest_axis(angle: float) -> str:
    return min(("A+", "B+", "A-", "B-"), key=lambda k: _ang_diff(angle, ISO_ANGLES[k]))


def _boxes(page):
    """Caixas de coordenadas (N/E/EL) com o retângulo que as contém."""
    rx = {
        "N": re.compile(r"\bN\s+(-?[\d.,]+)(?=\s|$)"),
        "E": re.compile(r"\bE\s+(-?[\d.,]+)(?=\s|$)"),
    }
    full = re.compile(r"^(N\s+-?[\d.,]+)?\s*(E\s+-?[\d.,]+)?\s*$")
    el = re.compile(r"^EL\s*([+-]?[\d.,]+)(\s*\(\w+\))?$")
    items = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            t = "".join(s["text"] for s in ln["spans"]).strip()
            vals = {}
            if t and full.match(t):
                for k, r in rx.items():
                    m = r.search(t)
                    if m:
                        vals[k] = _num(m.group(1))
            else:
                m = el.match(t)
                if m:
                    vals["EL"] = _num(m.group(1))
            if vals:
                items.append((ln["bbox"], vals))
    rects = []
    for d in page.get_drawings():
        r = d["rect"]
        if 20 < r.width < 260 and 15 < r.height < 120:
            rects.append(r)
    boxes = []
    for bb, vals in items:
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        cand = [r for r in rects if r.x0 - 1 <= bb[0] and bb[2] <= r.x1 + 1
                and r.y0 - 1 <= bb[1] and bb[3] <= r.y1 + 1]
        rect = min(cand, key=lambda r: r.width * r.height) if cand else None
        for b in boxes:
            if (rect is not None and b["rect"] is not None and rect == b["rect"]) or (
                    rect is None and b["rect"] is None and abs(b["x"] - bb[0]) < 3
                    and 0 <= bb[1] - b["y1"] < 6):
                b["vals"].update(vals)
                b["y1"] = bb[3]
                break
        else:
            boxes.append({"rect": rect, "vals": dict(vals), "x": bb[0], "y1": bb[3], "c": (cx, cy)})
    return boxes


def _leader_vertex(page, boxes, verts):
    """Associa cada caixa ao vértice ligado a ela por uma linha de chamada."""
    thin = []
    for d in page.get_drawings():
        if (d.get("width") or 0) > 2:
            continue
        for it in d["items"]:
            if it[0] == "l":
                thin.append(((it[1].x, it[1].y), (it[2].x, it[2].y)))
    res = {}
    for bi, b in enumerate(boxes):
        r = b["rect"]
        if r is None:
            continue
        rr = pymupdf.Rect(r) + (-5, -5, 5, 5)
        for a, c in thin:
            for e, o in ((a, c), (c, a)):
                if rr.contains(pymupdf.Point(*o)) and not rr.contains(pymupdf.Point(*e)):
                    for vi, v in enumerate(verts):
                        if math.hypot(e[0] - v[0], e[1] - v[1]) < 16:
                            res[bi] = vi
    # Sem linha de chamada: vértice mais próximo ainda livre.
    used = set(res.values())
    for bi, b in enumerate(boxes):
        if bi in res:
            continue
        free = [i for i in range(len(verts)) if i not in used] or list(range(len(verts)))
        vi = min(free, key=lambda i: math.hypot(verts[i][0] - b["c"][0], verts[i][1] - b["c"][1]))
        res[bi] = vi
        used.add(vi)
    return res


def _dimension_texts(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            t = "".join(s["text"] for s in ln["spans"]).strip()
            if re.fullmatch(r"\d{3,6}", t):
                r = ln["bbox"]
                dx, dy = ln["dir"]
                out.append((float(t), ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2), _ang(dx, dy)))
    return out


def parse_isometric(pdf: bytes) -> IsoResult:
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    page = doc[0]
    msgs: list[str] = []
    verts = _centerline(page)

    segs_dir = []
    for a, b in zip(verts[:-1], verts[1:]):
        segs_dir.append(_axis_of(_ang(b[0] - a[0], b[1] - a[1])))

    comp = _compass(page, msgs)
    if comp:
        north = _nearest_axis(comp["N"])
        east = _nearest_axis(comp["E"])
    else:
        north, east = "A+", "B-"
        msgs.append("Rosa dos ventos não identificada: assumi N = direita-acima, E = direita-abaixo.")
    to_geo = {north: ("N", 1), east: ("E", 1)}
    for k in list(to_geo):
        opp = k.replace("+", "#").replace("-", "+").replace("#", "-")
        to_geo[opp] = (to_geo[k][0], -1)
    to_geo["Z+"], to_geo["Z-"] = ("U", 1), ("U", -1)
    if len({v[0] for v in to_geo.values()}) != 3:
        raise ValueError("Norte e Leste devem estar em eixos isométricos distintos.")

    # coordenadas das caixas, por vértice
    unit = 1000.0
    text_all = page.get_text()
    if re.search(r"COORDENADAS[^;\n]*EM\s+mm", text_all, re.I):
        unit = 1.0
    coords: dict[int, dict] = {}
    try:
        boxes = _boxes(page)
        for bi, vi in _leader_vertex(page, boxes, verts).items():
            coords.setdefault(vi, {}).update(boxes[bi]["vals"])
    except Exception as exc:  # caixas são opcionais
        msgs.append(f"Caixas de coordenadas não lidas ({exc}).")

    dims = _dimension_texts(page)
    segments = []
    for i, name in enumerate(segs_dir):
        axis, sign = to_geo[name]
        key = "EL" if axis == "U" else axis
        a, b = coords.get(i, {}), coords.get(i + 1, {})
        length, src = None, ""
        if key in a and key in b:
            delta = (b[key] - a[key]) * unit
            length, src = abs(delta), "coordenadas do desenho"
            if delta * sign < 0:
                msgs.append(f"Trecho {i + 1}: o sentido do desenho contradiz as coordenadas "
                            f"({key} {a[key]:g} → {b[key]:g}).")
        else:
            p, q = verts[i], verts[i + 1]
            ang = _ang(q[0] - p[0], q[1] - p[1])
            mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            L = math.hypot(q[0] - p[0], q[1] - p[1])
            ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
            cand = []
            for val, (tx, ty), tang in dims:
                if _ang_diff(tang % 180, ang % 180) > 5 and _ang_diff(tang % 180, ang % 180) < 175:
                    continue
                along = (tx - p[0]) * ux + (ty - p[1]) * uy
                perp = abs(-(tx - p[0]) * uy + (ty - p[1]) * ux)
                if -0.05 * L <= along <= 1.05 * L and perp < 90:
                    cand.append(val)
            if cand:
                length, src = max(cand), "cota do desenho"
                if len(set(cand)) > 1:
                    msgs.append(f"Trecho {i + 1}: várias cotas candidatas {sorted(set(cand))}; "
                                f"usei a maior. Confira.")
        if length is None:
            raise ValueError(f"Trecho {i + 1}: sem coordenadas nem cota; informe o comprimento.")
        segments.append(Segment(axis, sign, float(length), src))

    nps = wall = None
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            t = "".join(s["text"] for s in ln["spans"])
            if nps is None and t.strip().upper().startswith("LINHA"):
                m = re.search(r'(\d+(?:[.,]\d+)?)\s*["”″\']', t)
                if m:
                    nps = _num(m.group(1))
            if wall is None:
                m = re.search(r"\be\s*=\s*(\d+[.,]\d+)", t)
                if m:
                    wall = _num(m.group(1))
    if nps is None:
        msgs.append("Diâmetro nominal não identificado: informe o DN.")
    elif wall is None:
        wall = SCH40.get(nps)
    p_op = p_design = None
    words = [w[4] for w in page.get_text("words")]
    for i in range(len(words) - 5):
        if words[i].upper().startswith("PRESS") and words[i + 1].lower() == "(bar":
            vals = [_num(x) for x in words[i + 3:i + 5] if re.fullmatch(r"\d+(?:[.,]\d+)?", x)]
            if len(vals) == 2:
                p_op, p_design = vals[0] / 10, vals[1] / 10  # bar -> MPa
                break
    t_op = t_design = None
    for i in range(len(words) - 5):
        if words[i].upper().startswith("TEMPERATURA") and words[i + 1] == "(°C)":
            vals = [_num(x) for x in words[i + 2:i + 4] if re.fullmatch(r"-?\d+(?:[.,]\d+)?", x)]
            if len(vals) == 2:
                t_op, t_design = vals
                break
    if p_design is None:
        msgs.append("Pressão do isométrico não identificada: informe a pressão interna.")
    if t_design is None:
        msgs.append("Temperatura do isométrico não identificada: informe a temperatura.")
    png = page.get_pixmap(dpi=90).tobytes("png")
    return IsoResult(segments, nps, wall, msgs, png, p_op, p_design, t_op, t_design)


def centerline_points(segments: list[Segment]) -> np.ndarray:
    """Vértices 3D (X=Leste, Y=Norte, Z=Elevação) em mm, partindo da origem."""
    idx = {"E": 0, "N": 1, "U": 2}
    p = np.zeros(3)
    pts = [p.copy()]
    for s in segments:
        p[idx[s.axis]] += s.sign * s.length
        pts.append(p.copy())
    return np.array(pts)


def bend_radius(nps: float) -> float:
    return BEND_FACTOR * nps * IN


def build_pipe_step(pts: np.ndarray, mean_diameter: float, radius: float) -> bytes:
    """Casca da tubulação: trechos retos (cilindros) e curvas (toros) tangentes entre si."""
    r = mean_diameter / 2
    n = len(pts) - 1
    d = [(pts[i + 1] - pts[i]) / np.linalg.norm(pts[i + 1] - pts[i]) for i in range(n)]
    length = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(n)]
    tang = [0.0] * (n + 1)
    theta = [0.0] * (n + 1)
    for i in range(1, n):
        c = float(np.clip(np.dot(d[i - 1], d[i]), -1, 1))
        theta[i] = math.acos(c)
        if theta[i] > 1e-6:
            tang[i] = radius * math.tan(theta[i] / 2)
    for i in range(n):
        if length[i] - tang[i] - tang[i + 1] < 1e-3 * length[i]:
            raise ValueError(
                f"O trecho {i + 1} ({length[i]:.0f} mm) é curto demais para as curvas de raio "
                f"{radius:.0f} mm (precisa de {tang[i] + tang[i + 1]:.0f} mm).")

    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
        path = f.name
    with _LOCK:
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("pipe")
            occ = gmsh.model.occ
            start = pts[0] + d[0] * tang[0]
            xax = np.cross(d[0], [0, 0, 1.0])
            if np.linalg.norm(xax) < 1e-6:
                xax = np.cross(d[0], [1.0, 0, 0])
            xax /= np.linalg.norm(xax)
            prof = occ.addCircle(*start, r, zAxis=list(d[0]), xAxis=list(xax))
            for i in range(n):
                v = d[i] * (length[i] - tang[i] - tang[i + 1])
                out = occ.extrude([(1, prof)], *v)
                prof = out[0][1]
                if i < n - 1 and theta[i + 1] > 1e-6:
                    nrm = d[i + 1] - np.dot(d[i + 1], d[i]) * d[i]
                    nrm /= np.linalg.norm(nrm)
                    p_end = pts[i + 1] - d[i] * tang[i + 1]
                    center = p_end + nrm * radius
                    axis = np.cross(d[i], d[i + 1])
                    axis /= np.linalg.norm(axis)
                    out = occ.revolve([(1, prof)], *center, *axis, theta[i + 1])
                    prof = out[0][1]
            occ.synchronize()
            gmsh.write(path)
        finally:
            gmsh.finalize()
    try:
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)
