"""Linear-elastic flat-facet shell FEA on triangular meshes.

Element: 3-node triangle, 6 DOF/node [UX, UY, UZ, RX, RY, RZ].
  - membrane: constant-strain triangle (CST)
  - bending + transverse shear: Reissner-Mindlin with discrete shear gap (DSG3), locking-free
  - drilling DOF (rotation about the normal): small artificial stiffness
Units are whatever the user chooses, as long as they are consistent (e.g. mm, N, MPa).
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DOFS = ["UX", "UY", "UZ", "RX", "RY", "RZ"]
DOF_INDEX = {d: i for i, d in enumerate(DOFS)}
SHEAR_CORRECTION = 5.0 / 6.0
DRILLING_FACTOR = 1e-5
CHUNK = 20000

# Coeficiente médio de dilatação térmica do aço carbono (ASME B31.3, Tab. C-1, grupo 1),
# em 1e-6 mm/mm/°C, medido a partir da temperatura de referência de 20 °C.
T_REF = 20.0
_ALPHA_T = np.array([-200, -100, -50, 20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300,
                     325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650,
                     675, 700, 725, 750, 775, 800], dtype=float)
_ALPHA_V = np.array([9.9, 10.7, 11.1, 11.5, 11.8, 11.9, 12.1, 12.3, 12.4, 12.6, 12.7, 12.9,
                     13.0, 13.2, 13.3, 13.4, 13.6, 13.7, 13.8, 14.0, 14.1, 14.2, 14.4, 14.5,
                     14.6, 14.7, 14.8, 14.9, 15.0, 15.1, 15.1, 15.2, 15.3, 15.3, 15.4]) * 1e-6


def steel_alpha(temp: float) -> float:
    """Mean thermal expansion coefficient (1/°C) of carbon steel between 20 °C and temp."""
    return float(np.interp(temp, _ALPHA_T, _ALPHA_V))


def thermal_strain(temp: float) -> float:
    """Free isotropic thermal strain relative to the 20 °C stress-free state."""
    return steel_alpha(temp) * (temp - T_REF)


# Tensão admissível S e tensão de escoamento Sy (MPa) em função da temperatura (°C).
_S_T = np.array([40, 65, 100, 125, 150, 200, 250, 300, 325, 350, 375, 400, 425, 450, 475], float)
_S_V = np.array([118, 118, 118, 118, 118, 118, 118, 118, 118, 117, 105, 88.9, 75.3, 62.7, 45.5])
_SY_T = np.array([40, 65, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425,
                  450, 475, 500, 525], float)
_SY_V = np.array([241, 227, 220, 217, 214, 210, 207, 203, 198, 194, 188, 183, 177, 172, 167,
                  162, 158, 154, 150, 146], float)


def allowable_sp(temp: float):
    """(S, Sy, Sp) em MPa na temperatura dada, com Sp = max(1,5·S, Sy); interpolação linear.

    Abaixo de 40 °C usa os valores de 40 °C. Acima do fim de uma tabela (475 °C para S,
    525 °C para Sy) o valor não é definido e Sp devolve None.
    """
    def lookup(xs, ys, x):
        return float(np.interp(max(x, xs[0]), xs, ys)) if x <= xs[-1] else None
    s, sy = lookup(_S_T, _S_V, temp), lookup(_SY_T, _SY_V, temp)
    sp = max(1.5 * s, sy) if s is not None and sy is not None else None
    return s, sy, sp

_MEM = np.array([0, 1, 6, 7, 12, 13])
_PLATE = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16])
_DRILL = np.array([5, 11, 17])


@dataclass
class Material:
    E: float
    nu: float
    t: float

    def validate(self):
        if self.E <= 0 or self.t <= 0:
            raise ValueError("E e espessura devem ser positivos.")
        if not 0 <= self.nu < 0.5:
            raise ValueError("O coeficiente de Poisson deve estar em [0, 0.5).")


@dataclass
class Equilibrium:
    ok: bool
    restrained_modes: int
    force: np.ndarray
    moment: np.ndarray
    messages: list = field(default_factory=list)


@dataclass
class FEAResult:
    equilibrium: Equilibrium
    solved: bool = False
    u: np.ndarray | None = None  # (N, 6)
    reactions: np.ndarray | None = None  # (N, 6)
    vm_nodal: dict | None = None
    vm_element: dict | None = None
    u_max: float = 0.0
    post: dict | None = None
    thickness: np.ndarray | None = None  # per-triangle thickness used
    local: dict | None = None  # tensões locais por camada (sl_*), eixos (ax) e keep_el


def oriented_triangles(nodes: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Make triangle orientation consistent and, per connected piece, pointing outward."""
    n = len(tri)
    he_a = tri.ravel()
    he_b = tri[:, [1, 2, 0]].ravel()
    key = np.minimum(he_a, he_b).astype(np.int64) * len(nodes) + np.maximum(he_a, he_b)
    order = np.argsort(key, kind="stable")
    uk, idx, cnt = np.unique(key[order], return_index=True, return_counts=True)
    first = order[idx[cnt == 2]]
    second = order[idx[cnt == 2] + 1]
    e1, e2 = first // 3, second // 3
    same = he_a[first] == he_a[second]

    adj = [[] for _ in range(n)]
    for a, b, s in zip(e1.tolist(), e2.tolist(), same.tolist()):
        adj[a].append((b, s))
        adj[b].append((a, s))

    flip = np.zeros(n, dtype=bool)
    comp = np.full(n, -1)
    ncomp = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        comp[s] = ncomp
        stack = [s]
        while stack:
            e = stack.pop()
            for f, sm in adj[e]:
                if comp[f] < 0:
                    comp[f] = ncomp
                    flip[f] = flip[e] ^ sm
                    stack.append(f)
        ncomp += 1

    tri = tri.copy()
    tri[flip] = tri[flip][:, [0, 2, 1]]
    p = nodes[tri]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    unit = nrm / np.maximum(2 * area, 1e-300)[:, None]
    cen = p.mean(axis=1)
    for c in range(ncomp):
        m = comp == c
        centre = (area[m, None] * cen[m]).sum(0) / area[m].sum()
        if (area[m] * np.einsum("ij,ij->i", unit[m], cen[m] - centre)).sum() < 0:
            tri[m] = tri[m][:, [0, 2, 1]]
    return tri


def _kinematics(nodes, tri, mat: Material):
    p = nodes[tri]
    v12, v13 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    nrm = np.cross(v12, v13)
    twoA = np.linalg.norm(nrm, axis=1)
    if np.any(twoA <= 1e-12 * np.max(twoA)):
        raise ValueError("A malha contém triângulos degenerados.")
    e3 = nrm / twoA[:, None]
    e1 = v12 / np.linalg.norm(v12, axis=1)[:, None]
    e2 = np.cross(e3, e1)
    T = np.stack([e1, e2, e3], axis=1)
    loc = np.einsum("nij,nkj->nki", T, p - p[:, :1])
    x, y = loc[:, :, 0], loc[:, :, 1]
    A = twoA / 2
    n = len(tri)

    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)

    Bm = np.zeros((n, 3, 6))
    Bb = np.zeros((n, 3, 9))
    G = np.zeros((n, 2, 9))
    for i in range(3):
        Bm[:, 0, 2 * i] = b[:, i]
        Bm[:, 1, 2 * i + 1] = c[:, i]
        Bm[:, 2, 2 * i] = c[:, i]
        Bm[:, 2, 2 * i + 1] = b[:, i]
        Bb[:, 0, 3 * i + 2] = b[:, i]
        Bb[:, 1, 3 * i + 1] = -c[:, i]
        Bb[:, 2, 3 * i + 1] = -b[:, i]
        Bb[:, 2, 3 * i + 2] = c[:, i]
    Bm /= twoA[:, None, None]
    Bb /= twoA[:, None, None]

    # Shear gaps g_1j = w_j - w_1 + 0.5 (phi_1 + phi_j).(r_j - r_1), phi = (RY, -RX)
    D = np.zeros((n, 2, 2))
    for j in (1, 2):
        dx, dy = x[:, j] - x[:, 0], y[:, j] - y[:, 0]
        D[:, j - 1, 0], D[:, j - 1, 1] = dx, dy
        G[:, j - 1, 0] = -1.0
        G[:, j - 1, 3 * j] = 1.0
        G[:, j - 1, 1] = -0.5 * dy
        G[:, j - 1, 3 * j + 1] = -0.5 * dy
        G[:, j - 1, 2] = 0.5 * dx
        G[:, j - 1, 3 * j + 2] = 0.5 * dx
    Bs = np.linalg.inv(D) @ G
    return T, A, Bm, Bb, Bs


def _constitutive(mat: Material, t):
    """Plane-stress matrix and per-element membrane/bending matrices for thickness array t."""
    E, nu = mat.E, mat.nu
    base = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    Dps = E / (1 - nu**2) * base
    return Dps, t[:, None, None] * Dps, (t**3 / 12)[:, None, None] * Dps


def _element_stiffness(nodes, tri, mat: Material, t):
    T, A, Bm, Bb, Bs = _kinematics(nodes, tri, mat)
    _, Dm, Db = _constitutive(mat, t)
    n = len(tri)
    Kl = np.zeros((n, 18, 18))
    Km = Bm.transpose(0, 2, 1) @ (Dm @ Bm)
    Kb = Bb.transpose(0, 2, 1) @ (Db @ Bb)
    Gs = SHEAR_CORRECTION * mat.E / (2 * (1 + mat.nu)) * t
    Ks = Gs[:, None, None] * (Bs.transpose(0, 2, 1) @ Bs)
    Kl[:, _MEM[:, None], _MEM[None, :]] += A[:, None, None] * Km
    Kl[:, _PLATE[:, None], _PLATE[None, :]] += A[:, None, None] * (Kb + Ks)
    kd = DRILLING_FACTOR * mat.E * t * A / 3
    for d in _DRILL:
        Kl[:, d, d] += kd
    T18 = np.zeros((n, 18, 18))
    for blk in range(6):
        T18[:, 3 * blk:3 * blk + 3, 3 * blk:3 * blk + 3] = T
    return T18.transpose(0, 2, 1) @ Kl @ T18


def assemble(nodes, tri, mat: Material, thickness=None) -> sp.csr_matrix:
    """Global stiffness; thickness is an optional per-triangle array (default mat.t)."""
    thickness = np.full(len(tri), mat.t) if thickness is None else thickness
    ndof = 6 * len(nodes)
    K = sp.csr_matrix((ndof, ndof))
    for s in range(0, len(tri), CHUNK):
        t = tri[s:s + CHUNK]
        Ke = _element_stiffness(nodes, t, mat, thickness[s:s + CHUNK])
        dofs = (6 * t[:, :, None] + np.arange(6)).reshape(len(t), 18).astype(np.int32)
        I = np.broadcast_to(dofs[:, :, None], Ke.shape)
        J = np.broadcast_to(dofs[:, None, :], Ke.shape)
        K = K + sp.coo_matrix((Ke.ravel(), (I.ravel(), J.ravel())), shape=(ndof, ndof)).tocsr()
    return K


def thermal_loads(nodes, tri, mat: Material, thickness, eps_th: float) -> np.ndarray:
    """Equivalent nodal forces of a uniform in-plane thermal strain (membrane only)."""
    f = np.zeros(6 * len(nodes))
    if eps_th == 0:
        return f
    e0 = np.array([eps_th, eps_th, 0.0])
    for s in range(0, len(tri), CHUNK):
        t = tri[s:s + CHUNK]
        T, A, Bm, _, _ = _kinematics(nodes, t, mat)
        _, Dm, _ = _constitutive(mat, thickness[s:s + CHUNK])
        fm = A[:, None] * np.einsum("nji,nj->ni", Bm, Dm @ e0)
        fl = np.zeros((len(t), 6, 3))
        fl.reshape(len(t), 18)[:, _MEM] = fm
        fg = np.einsum("nji,nbj->nbi", T, fl)  # local -> global, per node
        for k in range(3):
            for d in range(3):
                np.add.at(f, 6 * t[:, k] + d, fg[:, 2 * k, d])
    return f


def pressure_loads(nodes, tri, tri_surface, pressure: dict) -> np.ndarray:
    """Nodal forces from pressure. Positive pressure acts against the outward normal."""
    f = np.zeros(6 * len(nodes))
    p = nodes[tri]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    unit = nrm / (2 * area)[:, None]
    for surf, val in pressure.items():
        m = np.flatnonzero(tri_surface == surf)
        if len(m) == 0 or val == 0:
            continue
        fe = -val * unit[m] * (area[m] / 3)[:, None]
        for k in range(3):
            base = 6 * tri[m, k]
            for d in range(3):
                np.add.at(f, base + d, fe[:, d])
    return f


def edge_loads(nodes, curves: dict, edge_forces: dict) -> np.ndarray:
    """Nodal forces from total forces (N, global X/Y/Z) spread uniformly along each edge."""
    f = np.zeros(6 * len(nodes))
    for tag, total in edge_forces.items():
        seg = curves.get(tag)
        if seg is None:
            continue
        length = np.linalg.norm(nodes[seg[:, 1]] - nodes[seg[:, 0]], axis=1)
        if length.sum() == 0:
            continue
        for k in range(2):
            for d in range(3):
                np.add.at(f, 6 * seg[:, k] + d, 0.5 * length / length.sum() * total[d])
    return f


def prescribed_dofs(mesh, curve_bc: dict) -> dict:
    """Map global dof index -> imposed value from edge conditions."""
    out = {}
    for tag, dofs in curve_bc.items():
        if tag not in mesh.curves:
            continue
        for node in np.unique(mesh.curves[tag]):
            for name, val in dofs.items():
                out[6 * int(node) + DOF_INDEX[name]] = float(val)
    return out


def _rigid_modes(nodes):
    nn = len(nodes)
    c = nodes.mean(axis=0)
    X = (nodes - c) / max(np.ptp(nodes, axis=0).max(), 1e-12)
    R = np.zeros((6 * nn, 6))
    for k in range(3):
        R[k::6, k] = 1.0
    x, y, z = X.T
    R[0::6, 3], R[1::6, 3], R[2::6, 3] = 0, -z, y
    R[0::6, 4], R[1::6, 4], R[2::6, 4] = z, 0, -x
    R[0::6, 5], R[1::6, 5], R[2::6, 5] = -y, x, 0
    R[3::6, 3] = 1.0
    R[4::6, 4] = 1.0
    R[5::6, 5] = 1.0
    return R


def resultants(nodes, f):
    """Total force and moment (about the mesh centroid) of a nodal force/moment vector."""
    c = nodes.mean(axis=0)
    F = f.reshape(-1, 6)
    force = F[:, :3].sum(axis=0)
    moment = np.cross(nodes - c, F[:, :3]).sum(axis=0) + F[:, 3:].sum(axis=0)
    return force, moment


def check_equilibrium(nodes, f, prescribed: dict) -> Equilibrium:
    """Static equilibrium: loads must be balanced by the supports.

    A load system can only be in equilibrium if it does no work on any rigid-body motion
    that the supports leave free (those modes carry no reaction).
    """
    force, moment = resultants(nodes, f)
    R = _rigid_modes(nodes)
    P = np.array(sorted(prescribed), dtype=int)
    msgs = []
    if len(P):
        U, S, Vt = np.linalg.svd(R[P], full_matrices=True)
        rank = int(np.sum(S > 1e-8 * S[0])) if S[0] > 0 else 0
        null = Vt[rank:].T
    else:
        rank, null = 0, np.eye(6)
    free_modes = 6 - rank

    ok = True
    if free_modes:
        modes = R @ null
        work = modes.T @ f
        scale = np.abs(modes).T @ np.abs(f) + 1e-300
        unbalanced = np.abs(work) > 1e-6 * scale
        if np.any(unbalanced):
            ok = False
            msgs.append(
                f"As cargas não estão em equilíbrio: realizam trabalho em {int(unbalanced.sum())} "
                f"movimento(s) de corpo rígido que os apoios não restringem (sem reação possível)."
            )
        else:
            ok = False
            msgs.append(
                f"As cargas estão autoequilibradas, mas os apoios deixam {free_modes} "
                f"movimento(s) de corpo rígido livre(s) (modelo hipostático). Restrinja mais GDL."
            )
    else:
        msgs.append("Os apoios restringem os 6 movimentos de corpo rígido: as cargas são "
                    "equilibradas pelas reações.")
    return Equilibrium(ok, rank, force, moment, msgs)


def _stresses(nodes, tri, u, mat: Material, thickness, eps_th: float = 0.0):
    """Von Mises stress (key `name`) and equivalent strain in % (key `eps_name`) per layer."""
    n = len(tri)
    Dps = _constitutive(mat, np.ones(1))[0]
    names = ("top", "bottom", "mid")
    out = {k: np.zeros(n) for k in ("max", "eps_max") + names + tuple("eps_" + k for k in names)
           + tuple("p1_" + k for k in names) + tuple("p2_" + k for k in names)}
    for k in names:
        out["d1_" + k] = np.zeros((n, 3))
        out["d2_" + k] = np.zeros((n, 3))
        out["sl_" + k] = np.zeros((n, 3))  # tensões locais [σx, σy, τxy]
    out["ax"] = np.zeros((n, 2, 3))  # eixos locais x,y (global) de cada elemento
    for s in range(0, n, CHUNK):
        t = tri[s:s + CHUNK]
        T, A, Bm, Bb, Bs = _kinematics(nodes, t, mat)
        ug = u.reshape(-1, 6)[t].reshape(len(t), 6, 3)
        ul = np.einsum("nij,nbj->nbi", T, ug).reshape(len(t), 18)
        em = np.einsum("nij,nj->ni", Bm, ul[:, _MEM]) - np.array([eps_th, eps_th, 0.0])
        kp = np.einsum("nij,nj->ni", Bb, ul[:, _PLATE])
        sl = slice(s, s + len(t))
        th = thickness[sl]
        for name, z in (("top", th / 2), ("bottom", -th / 2), ("mid", 0 * th)):
            sg = (em + z[:, None] * kp) @ Dps.T
            out["sl_" + name][sl] = sg
            out["ax"][sl] = T[:, :2, :]
            out[name][sl] = np.sqrt(np.maximum(
                sg[:, 0]**2 - sg[:, 0] * sg[:, 1] + sg[:, 1]**2 + 3 * sg[:, 2]**2, 0))
            out["eps_" + name][sl] = 100 * out[name][sl] / mat.E  # equivalent elastic strain
            # Maior tensão principal no plano (estado plano de tensões)
            out["p1_" + name][sl] = (sg[:, 0] + sg[:, 1]) / 2 + np.sqrt(
                ((sg[:, 0] - sg[:, 1]) / 2)**2 + sg[:, 2]**2)
            # Direção (global, unitária) da tensão principal: eixos locais são as linhas de T
            ang = 0.5 * np.arctan2(2 * sg[:, 2], sg[:, 0] - sg[:, 1])
            out["d1_" + name][sl] = (np.cos(ang)[:, None] * T[:, 0, :]
                                     + np.sin(ang)[:, None] * T[:, 1, :])
            out["p2_" + name][sl] = (sg[:, 0] + sg[:, 1]) / 2 - np.sqrt(
                ((sg[:, 0] - sg[:, 1]) / 2)**2 + sg[:, 2]**2)
            out["d2_" + name][sl] = (-np.sin(ang)[:, None] * T[:, 0, :]
                                     + np.cos(ang)[:, None] * T[:, 1, :])
    out["max"] = np.maximum(out["top"], out["bottom"])
    out["eps_max"] = np.maximum(out["eps_top"], out["eps_bottom"])
    return out


def _nodal_average(nn, tri, area, values):
    num = np.zeros(nn)
    den = np.zeros(nn)
    for k in range(3):
        np.add.at(num, tri[:, k], values * area)
        np.add.at(den, tri[:, k], area)
    return num / np.maximum(den, 1e-300)


def normal_stress_to_edge(nodes, tri, seg, local) -> dict | None:
    """Maior tensão normal (com sinal) ortogonal à aresta, no plano da casca, por camada.

    Considera os elementos que têm um segmento da aresta como lado. Para cada um, a direção
    ortogonal à aresta é normal_do_elemento × tangente do segmento. Elementos dentro da zona
    desprezada junto às restrições ficam de fora. Retorna {"mid","top","bottom"} ou None.
    """
    nn = len(nodes)
    seg = np.asarray(seg)
    skey = np.minimum(seg[:, 0], seg[:, 1]).astype(np.int64) * nn + np.maximum(seg[:, 0], seg[:, 1])
    pairs = ((0, 1), (1, 2), (2, 0))
    best = {k: -np.inf for k in ("mid", "top", "bottom")}
    ax = local["ax"]
    keep = local["keep_el"]
    for i, j in pairs:
        a, b = tri[:, i], tri[:, j]
        key = np.minimum(a, b).astype(np.int64) * nn + np.maximum(a, b)
        el = np.flatnonzero(np.isin(key, skey) & keep)
        if len(el) == 0:
            continue
        t = nodes[b[el]] - nodes[a[el]]
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
        nrm = np.cross(ax[el, 0], ax[el, 1])
        d = np.cross(nrm, t)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-300)
        nx, ny = (d * ax[el, 0]).sum(1), (d * ax[el, 1]).sum(1)
        for k in best:
            s = local["sl_" + k][el]
            best[k] = max(best[k], float((s[:, 0] * nx**2 + s[:, 1] * ny**2 + 2 * s[:, 2] * nx * ny).max()))
    return None if not np.isfinite(best["mid"]) else best


def distance_to_segments(points, nodes, seg) -> np.ndarray:
    """Minimum distance from each point to a set of line segments (node index pairs)."""
    a, b = nodes[seg[:, 0]], nodes[seg[:, 1]]
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-300)
    out = np.full(len(points), np.inf)
    step = max(1, 2_000_000 // max(len(seg), 1))
    for s in range(0, len(points), step):
        p = points[s:s + step, None, :]
        w = np.clip(((p - a) * ab).sum(2) / denom, 0, 1)
        d = np.linalg.norm(p - (a + w[:, :, None] * ab), axis=2)
        out[s:s + step] = d.min(1)
    return out


def corrosion_zone(nodes, tri, node: int, radius: float) -> np.ndarray:
    """Boolean mask of triangles within `radius` of `node`, measured along the mesh surface.

    Distance is the shortest path over mesh edges, so the opposite wall of a thin
    part is not captured; a triangle belongs to the zone if its mean nodal distance <= radius.
    """
    from scipy.sparse.csgraph import dijkstra
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    # Extra links between the opposite vertices of neighbouring triangles add directions
    # to the graph, so path lengths approximate surface distance better than mesh edges alone.
    opp = np.concatenate([tri[:, 2], tri[:, 0], tri[:, 1]])
    key = np.minimum(e[:, 0], e[:, 1]) * len(nodes) + np.maximum(e[:, 0], e[:, 1])
    order = np.argsort(key, kind="stable")
    same = key[order][1:] == key[order][:-1]
    diag = np.stack([opp[order][:-1][same], opp[order][1:][same]], axis=1)
    e = np.concatenate([e, diag])
    w = np.linalg.norm(nodes[e[:, 0]] - nodes[e[:, 1]], axis=1)
    g = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(len(nodes), len(nodes))).tocsr()
    dist = dijkstra(g, directed=False, indices=int(node))
    return dist[tri].mean(axis=1) <= radius


def element_thickness(mesh, nominal: float, corrosion: list) -> np.ndarray:
    """Per-triangle thickness: nominal, reduced inside each corrosion zone (minimum wins)."""
    t = np.full(len(mesh.triangles), float(nominal))
    for i, d in enumerate(corrosion, 1):
        if not d["radius"] > 0:
            raise ValueError(f"Corrosão {i}: o raio deve ser maior que 0.")
        if not 0 < d["t"] < nominal:
            raise ValueError(f"Corrosão {i}: a espessura corroída deve ser maior que 0 e menor que a "
                             f"espessura nominal ({nominal:g}).")
        if not 0 <= d["node"] < len(mesh.nodes):
            raise ValueError(f"Corrosão {i}: nó inválido.")
        zone = corrosion_zone(mesh.nodes, mesh.triangles, d["node"], d["radius"])
        if not zone.any():
            raise ValueError(f"Corrosão {i}: nenhum elemento na área; refine a malha.")
        t[zone] = np.minimum(t[zone], d["t"])
    return t


def gravity_loads(nodes, tri, area, thickness, gravity: dict) -> np.ndarray:
    """Self-weight (and optional water fill) as nodal forces; gravity acts along -Z.

    Steel: rho * t * A per element (t is the corroded thickness where applicable).
    Water: for a circular pipe, fluid volume per unit wall area is ri^2 / (2 * rm), which is
    exact for cylinders and for torus bends (Pappus), so it is spread over the wall area.
    """
    g = gravity["g"]
    w = gravity["rho_steel"] * thickness * area * g
    water = gravity.get("water")
    if water:
        ri = water["ri"]
        rm = ri + gravity["t_nominal"] / 2
        w = w + water["rho"] * area * ri ** 2 / (2 * rm) * g
    f = np.zeros(6 * len(nodes))
    np.add.at(f, 6 * tri.ravel() + 2, np.repeat(-w / 3, 3))
    return f


def run_analysis(mesh, mat: Material, bc: dict, gravity: dict | None = None,
                 temperature: float | None = None, ignore_dist: float = 0.0) -> FEAResult:
    mat.validate()
    nodes = mesh.nodes
    nn = len(nodes)
    tri = oriented_triangles(nodes, mesh.triangles)
    thickness = element_thickness(mesh, mat.t, bc.get("corrosion", []))
    area = 0.5 * np.linalg.norm(
        np.cross(nodes[tri][:, 1] - nodes[tri][:, 0], nodes[tri][:, 2] - nodes[tri][:, 0]), axis=1)

    f = pressure_loads(nodes, tri, mesh.tri_surface, bc.get("pressure", {}))
    if gravity:
        f = f + gravity_loads(nodes, tri, area, thickness, {**gravity, "t_nominal": mat.t})
    f = f + edge_loads(nodes, mesh.curves, bc.get("edge_forces", {}))
    eps_th = thermal_strain(temperature) if temperature is not None else 0.0
    f = f + thermal_loads(nodes, tri, mat, thickness, eps_th)
    for load in bc.get("forces", []):
        f[6 * load["node"]:6 * load["node"] + 3] += load["F"]
    presc = prescribed_dofs(mesh, bc.get("curves", {}))
    if not presc:
        raise ValueError("Defina ao menos uma restrição em aresta.")
    if not np.any(f) and not any(v != 0 for v in presc.values()):
        raise ValueError("Defina ao menos uma carga (pressão, gravidade) ou deslocamento imposto não nulo.")

    eq = check_equilibrium(nodes, f, presc)
    if not eq.ok:
        return FEAResult(equilibrium=eq)

    K = assemble(nodes, tri, mat, thickness)
    ndof = 6 * nn
    pidx = np.array(sorted(presc), dtype=int)
    pval = np.array([presc[i] for i in pidx])
    fidx = np.setdiff1d(np.arange(ndof), pidx)

    u = np.zeros(ndof)
    u[pidx] = pval
    Kff = K[fidx][:, fidx].tocsc()
    rhs = f[fidx] - K[fidx][:, pidx] @ pval
    if len(fidx):
        sol = spla.spsolve(Kff, rhs)
        if not np.all(np.isfinite(sol)):
            raise ValueError("Sistema singular: o modelo possui mecanismo ou restrições insuficientes.")
        res = np.linalg.norm(Kff @ sol - rhs) / max(np.linalg.norm(rhs), 1e-300)
        if res > 1e-6:
            raise ValueError(f"Solução imprecisa (resíduo relativo {res:.2e}); verifique as restrições.")
        u[fidx] = sol

    reactions = np.zeros(ndof)
    reactions[pidx] = (K @ u - f)[pidx]
    rf, rm = resultants(nodes, reactions)
    lf, lm = eq.force, eq.moment
    fscale = max(np.abs(f.reshape(-1, 6)[:, :3]).sum(), np.abs(reactions.reshape(-1, 6)[:, :3]).sum(), 1e-300)
    L = np.ptp(nodes, axis=0).max()
    post = {
        "load_force": lf, "load_moment": lm, "reaction_force": rf, "reaction_moment": rm,
        "force_residual": rf + lf, "moment_residual": rm + lm,
        "force_error": float(np.linalg.norm(rf + lf) / fscale),
        "moment_error": float(np.linalg.norm(rm + lm) / (fscale * L)),
    }

    vm_el = _stresses(nodes, tri, u, mat, thickness, eps_th)
    dir_el = {k: vm_el.pop(k) for k in [k for k in vm_el if k.startswith(("d1_", "d2_"))]}
    local = {k: vm_el.pop(k) for k in [k for k in vm_el if k.startswith("sl_") or k == "ax"]}
    keep_el = np.ones(len(tri), bool)
    keep_nd = np.ones(nn, bool)
    seg = [mesh.curves[t] for t, d in bc.get("curves", {}).items() if d and t in mesh.curves]
    if ignore_dist > 0 and seg:
        seg = np.vstack(seg)
        keep_el = distance_to_segments(nodes[tri].mean(axis=1), nodes, seg) >= ignore_dist
        keep_nd = distance_to_segments(nodes, nodes, seg) >= ignore_dist
        vm_el = {k: np.where(keep_el, v, 0.0) for k, v in vm_el.items()}
    vm_nd = {}
    for k, v in vm_el.items():
        nodal = _nodal_average(nn, tri, area * keep_el, v)
        vm_nd[k] = np.where(keep_nd, nodal, 0.0)
    # Direção nodal: vetores dos elementos alinhados em sinal (a direção é um eixo, ±) e
    # médios ponderados pela área.
    for k, d in dir_el.items():
        ref = np.zeros((nn, 3))
        for c in range(3):
            ref[tri[:, c]] = d
        acc = np.zeros((nn, 3))
        for c in range(3):
            sgn = np.sign((d * ref[tri[:, c]]).sum(1))
            sgn[sgn == 0] = 1
            np.add.at(acc, tri[:, c], d * (sgn * area * keep_el)[:, None])
        vm_nd[k] = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-300)
    U = u.reshape(-1, 6)
    return FEAResult(equilibrium=eq, solved=True, u=U, reactions=reactions.reshape(-1, 6),
                     vm_nodal=vm_nd, vm_element=vm_el,
                     u_max=float(np.linalg.norm(U[:, :3], axis=1).max()), post=post,
                     thickness=thickness, local={**local, "keep_el": keep_el})
