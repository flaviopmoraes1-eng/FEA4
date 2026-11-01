// Interactive 3D viewer: orbit/pan/zoom, hover and click selection of edges and surfaces.
// THREE_SRC is prepended by viewer.py (vendored three.js module source).

function loadThree() {
  if (!window.__pipeThree) {
    const url = URL.createObjectURL(new Blob([THREE_SRC], { type: "text/javascript" }));
    window.__pipeThree = import(url);
  }
  return window.__pipeThree;
}

// Blue -> cyan -> green -> yellow -> red colour ramp, t in [0, 1].
const STOPS = [[0, 0, 0.6], [0, 0.35, 1], [0, 0.9, 0.9], [0.2, 0.85, 0.2], [1, 0.9, 0], [1, 0.45, 0], [0.8, 0, 0]];
function ramp(t) {
  const x = Math.min(Math.max(t, 0), 1) * (STOPS.length - 1);
  const i = Math.min(Math.floor(x), STOPS.length - 2), f = x - i;
  return STOPS[i].map((v, k) => v + (STOPS[i + 1][k] - v) * f);
}

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  const wrap = parentElement.querySelector(".wrap");
  const hint = parentElement.querySelector(".hint");
  const legend = parentElement.querySelector(".legend");
  if (data.field) {
    legend.style.display = "flex";
    legend.querySelector(".lg-title").textContent = data.field_label || "";
    legend.querySelector(".lg-max").textContent = data.field_max.toPrecision(4);
    legend.querySelector(".lg-min").textContent = data.field_min.toPrecision(4);
    legend.querySelector(".lg-bar").style.background =
      "linear-gradient(to top," + STOPS.map((s, i) =>
        `rgb(${s.map((v) => Math.round(v * 255)).join(",")}) ${(i / (STOPS.length - 1)) * 100}%`).join(",") + ")";
  } else {
    legend.style.display = "none";
  }
  // Streamlit may re-run this function on data updates without calling the previous
  // cleanup, so dispose whatever was mounted before.
  if (wrap.__pipeDispose) wrap.__pipeDispose();
  wrap.replaceChildren();
  let disposed = false;
  let cleanup = () => {};
  const dispose = () => {
    disposed = true;
    cleanup();
    cleanup = () => {};
  };
  wrap.__pipeDispose = dispose;

  loadThree().then((THREE) => {
    if (disposed) return;
    cleanup = build(THREE);
  });

  function build(THREE) {
    const W = () => wrap.clientWidth || 800;
    const H = () => wrap.clientHeight || 600;

    const nodes = data.nodes;
    const tris = data.tris;
    const triSurf = data.tri_surf;
    const nTri = triSurf.length;
    const diag = data.diag;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(W(), H());
    renderer.setClearColor(0xf4f6f9);
    wrap.appendChild(renderer.domElement);
    const canvas = renderer.domElement;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, W() / H(), diag * 0.001, diag * 50);
    camera.up.set(0, 0, 1);
    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const light = new THREE.DirectionalLight(0xffffff, 0.8);
    scene.add(light);

    // Mesh (non-indexed so each triangle has its own colour).
    const pos = new Float32Array(nTri * 9);
    for (let t = 0; t < nTri; t++) {
      for (let k = 0; k < 3; k++) {
        const n = tris[t * 3 + k] * 3;
        pos.set([nodes[n], nodes[n + 1], nodes[n + 2]], t * 9 + k * 3);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.computeVertexNormals();
    const triColors = new Float32Array(nTri * 9);
    geo.setAttribute("color", new THREE.BufferAttribute(triColors, 3));
    const mesh = new THREE.Mesh(
      geo,
      new THREE.MeshLambertMaterial({
        vertexColors: true, side: THREE.DoubleSide,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      })
    );
    scene.add(mesh);
    geo.computeBoundingSphere();
    geo.computeBoundingBox();

    // Corroded zones: translucent magenta triangles over the surface.
    const corrT = data.corr_tris || [];
    if (corrT.length) {
      const cp = new Float32Array(corrT.length * 9);
      corrT.forEach((t, i) => cp.set(pos.subarray(t * 9, t * 9 + 9), i * 9));
      const cg = new THREE.BufferGeometry();
      cg.setAttribute("position", new THREE.BufferAttribute(cp, 3));
      scene.add(new THREE.Mesh(cg, new THREE.MeshBasicMaterial({
        color: 0xd81b60, transparent: true, opacity: 0.45, side: THREE.DoubleSide, depthWrite: false,
        polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
      })));
    }
    const nodeMode = data.mode === "node";

    if (data.show_mesh) {
      const wf = new THREE.LineSegments(
        new THREE.WireframeGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0x000000 })
      );
      scene.add(wf);
    }

    // Undeformed reference shape.
    if (data.ghost_nodes) {
      const g = data.ghost_nodes;
      const gpos = new Float32Array(nTri * 9);
      for (let t = 0; t < nTri; t++) {
        for (let k = 0; k < 3; k++) {
          const n = tris[t * 3 + k] * 3;
          gpos.set([g[n], g[n + 1], g[n + 2]], t * 9 + k * 3);
        }
      }
      const ggeo = new THREE.BufferGeometry();
      ggeo.setAttribute("position", new THREE.BufferAttribute(gpos, 3));
      scene.add(new THREE.LineSegments(
        new THREE.WireframeGeometry(ggeo),
        new THREE.LineBasicMaterial({ color: 0x9aa5b1, transparent: true, opacity: 0.8 })
      ));
    }

    // Geometric curves, one contiguous range of segments per curve.
    const curveTags = Object.keys(data.curves).map(Number);
    const curveRange = {};
    let nSeg = 0;
    for (const c of curveTags) {
      const n = data.curves[c].length / 2;
      curveRange[c] = [nSeg, nSeg + n];
      nSeg += n;
    }
    const cpos = new Float32Array(nSeg * 6);
    const segs = []; // [curveTag, ax, ay, az, bx, by, bz]
    for (const c of curveTags) {
      const arr = data.curves[c];
      for (let s = 0; s < arr.length / 2; s++) {
        const a = arr[s * 2] * 3, b = arr[s * 2 + 1] * 3;
        const idx = curveRange[c][0] + s;
        cpos.set([nodes[a], nodes[a + 1], nodes[a + 2], nodes[b], nodes[b + 1], nodes[b + 2]], idx * 6);
        segs.push([c, nodes[a], nodes[a + 1], nodes[a + 2], nodes[b], nodes[b + 1], nodes[b + 2]]);
      }
    }
    const cgeo = new THREE.BufferGeometry();
    cgeo.setAttribute("position", new THREE.BufferAttribute(cpos, 3));
    const curveColors = new Float32Array(nSeg * 6);
    cgeo.setAttribute("color", new THREE.BufferAttribute(curveColors, 3));
    const lines = new THREE.LineSegments(cgeo, new THREE.LineBasicMaterial({ vertexColors: true }));
    scene.add(lines);

    // Selection state.
    const sel = {
      curves: new Set((data.selection && data.selection.curves) || []),
      surfaces: new Set((data.selection && data.selection.surfaces) || []),
      node: data.selection && data.selection.node != null ? data.selection.node : null,
    };
    let hover = null;
    const curveBc = data.curve_bc || {};
    const surfP = data.surf_p || {};

    const col = (hex) => new THREE.Color(hex);
    const C = {
      surf: col(0xb0c4de), surfP: col(0x5dade2), surfSel: col(0xffa726), surfHover: col(0xffe0a3),
      cur: col(0x1f2933), curAnchor: col(0x2e7d32), curCustom: col(0x8e24aa),
      curSel: col(0xe53935), curHover: col(0xff8a80), curCrack: col(0x00e676),
    };
    const crackEdges = new Set((data.crack_edges || []).map(String));

    function recolor() {
      const fld = data.field;
      const fmin = data.field_min, frange = Math.max(data.field_max - data.field_min, 1e-300);
      for (let t = 0; t < nTri; t++) {
        const s = triSurf[t];
        if (fld) {
          for (let k = 0; k < 3; k++) {
            const c = ramp((fld[tris[t * 3 + k]] - fmin) / frange);
            triColors.set(c, t * 9 + k * 3);
          }
          continue;
        }
        let c = surfP[s] !== undefined ? C.surfP : C.surf;
        if (hover && hover.type === "surface" && hover.id === s) c = C.surfHover;
        if (sel.surfaces.has(s)) c = C.surfSel;
        for (let k = 0; k < 3; k++) triColors.set([c.r, c.g, c.b], t * 9 + k * 3);
      }
      geo.attributes.color.needsUpdate = true;
      for (const cTag of curveTags) {
        let c = curveBc[cTag] === "anchor" ? C.curAnchor : curveBc[cTag] ? C.curCustom : C.cur;
        if (crackEdges.has(cTag)) c = C.curCrack;
        if (hover && hover.type === "curve" && hover.id === cTag) c = C.curHover;
        if (sel.curves.has(cTag)) c = C.curSel;
        const [a, b] = curveRange[cTag];
        for (let i = a * 2; i < b * 2; i++) curveColors.set([c.r, c.g, c.b], i * 3);
      }
      cgeo.attributes.color.needsUpdate = true;
    }

    // Orbit camera (Z up) with persistence across Streamlit reruns.
    const ctr = geo.boundingBox.getCenter(new THREE.Vector3());
    const saved = window.__pipeCam && window.__pipeCam.sig === data.sig ? window.__pipeCam : null;
    const fitFov = Math.min(camera.fov, (2 * Math.atan(Math.tan((camera.fov * Math.PI) / 360) * camera.aspect) * 180) / Math.PI);
    const cam = saved || {
      sig: data.sig, theta: -0.9, phi: 1.1,
      dist: (geo.boundingSphere.radius / Math.sin((fitFov * Math.PI) / 360)) * 1.1,
      target: [ctr.x, ctr.y, ctr.z],
    };
    window.__pipeCam = cam;
    // Axis triad (X red, Y green, Z blue) drawn in a corner viewport, following the camera.
    const axScene = new THREE.Scene();
    const axCam = new THREE.PerspectiveCamera(30, 1, 0.1, 10);
    const AX = [["X", 0xe53935, [1, 0, 0]], ["Y", 0x43a047, [0, 1, 0]], ["Z", 0x1e66f5, [0, 0, 1]]];
    for (const [name, color, d] of AX) {
      const dir = new THREE.Vector3(...d);
      axScene.add(new THREE.ArrowHelper(dir, new THREE.Vector3(), 1, color, 0.28, 0.16));
      const cv = document.createElement("canvas");
      cv.width = cv.height = 64;
      const g = cv.getContext("2d");
      g.fillStyle = "#" + color.toString(16).padStart(6, "0");
      g.font = "bold 44px sans-serif";
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.fillText(name, 32, 34);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: new THREE.CanvasTexture(cv), depthTest: false, transparent: true }));
      sprite.position.copy(dir).multiplyScalar(1.3);
      sprite.scale.set(0.5, 0.5, 0.5);
      axScene.add(sprite);
    }
    // Result overlays: "Máx." flag at the peak node and a tooltip for the node under the mouse.
    wrap.style.position = "relative";
    const mkOverlay = (css, html) => {
      const el = document.createElement("div");
      el.style.cssText = "position:absolute;pointer-events:none;display:none;font:12px sans-serif;" + css;
      el.innerHTML = html || "";
      wrap.appendChild(el);
      return el;
    };
    let maxNode = -1;
    if (data.field) {
      maxNode = 0;
      for (let i = 1; i < data.field.length; i++) if (data.field[i] > data.field[maxNode]) maxNode = i;
    }
    const flagEl = data.field ? mkOverlay("transform:translate(-1px,-100%);",
      '<div style="background:#d32f2f;color:#fff;font-weight:700;font-size:12px;padding:1px 5px;' +
      'border-radius:2px;white-space:nowrap">Máx.' +
      (data.max_note ? " " + data.max_note : "") + '</div>' +
      '<div style="width:2px;height:12px;background:#b71c1c"></div>') : null;
    const corrFlags = [
      ...(data.corr_flags || []).map((f) => ({ ...f, bg: "#ef6c00", bar: "#bf360c" })),
      ...(data.crack_flags || []).map((f) => ({ ...f, bg: "#00a152", bar: "#007a3d" })),
    ].map((f) => ({
      node: f.node,
      el: mkOverlay("transform:translate(-1px,-100%);",
        '<div style="background:' + f.bg + ';color:#fff;font-weight:700;font-size:12px;padding:1px 5px;' +
        'border-radius:2px;white-space:nowrap">' + f.text + '</div>' +
        '<div style="width:2px;height:12px;background:' + f.bar + '"></div>'),
    }));
    const dotEl = mkOverlay(
      "width:10px;height:10px;margin:-5px 0 0 -5px;border:2px solid #111;border-radius:50%;" +
      "background:rgba(255,255,255,0.6);box-sizing:border-box;");
    const selEl = mkOverlay(
      "width:16px;height:16px;margin:-8px 0 0 -8px;border:3px solid #d81b60;border-radius:50%;" +
      "background:rgba(216,27,96,0.25);box-sizing:border-box;");
    const tipEl = mkOverlay(
      "background:rgba(17,24,39,0.92);color:#fff;padding:4px 8px;border-radius:4px;" +
      "white-space:nowrap;transform:translate(12px,-120%);");
    // Seta da direção da tensão principal no nó sob o mouse (SVG sobreposto)
    const NS = "http://www.w3.org/2000/svg";
    const arrowSvg = document.createElementNS(NS, "svg");
    arrowSvg.setAttribute("style", "position:absolute;left:0;top:0;width:100%;height:100%;" +
      "pointer-events:none;display:none;overflow:visible");
    arrowSvg.innerHTML = '<defs><marker id="pfArrow" markerWidth="8" markerHeight="8" refX="6" refY="4" ' +
      'orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 z" fill="#0d47a1"/></marker></defs>' +
      '<line stroke="#0d47a1" stroke-width="3" marker-end="url(#pfArrow)"/>' +
      '<line stroke="#0d47a1" stroke-width="3" marker-end="url(#pfArrow)"/>' +
      '<line stroke="#e65100" stroke-width="3" marker-end="url(#pfArrow2)"/>' +
      '<line stroke="#e65100" stroke-width="3" marker-end="url(#pfArrow2)"/>';
    arrowSvg.querySelector("defs").insertAdjacentHTML("beforeend",
      '<marker id="pfArrow2" markerWidth="8" markerHeight="8" refX="6" refY="4" ' +
      'orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 z" fill="#e65100"/></marker>');
    wrap.appendChild(arrowSvg);
    const arrowLines = arrowSvg.querySelectorAll("line");
    const arrowV = new THREE.Vector3();
    function updateArrow(i) {
      if (!data.field_dir || i < 0) { arrowSvg.style.display = "none"; return; }
      const L = diag * 0.06;
      const [x0, y0] = screenOf(i);
      let any = false;
      arrowLines.forEach(ln => ln.setAttribute("x2", x0));
      [[data.field_dir, 0], [data.field_dir2, 2]].forEach(([dir, off]) => {
        if (!dir) { arrowLines[off].style.display = arrowLines[off + 1].style.display = "none"; return; }
        [1, -1].forEach((s, k) => {
          arrowV.set(nodes[i * 3] + s * L * dir[i * 3],
                     nodes[i * 3 + 1] + s * L * dir[i * 3 + 1],
                     nodes[i * 3 + 2] + s * L * dir[i * 3 + 2]).project(camera);
          const x1 = (arrowV.x + 1) / 2 * W(), y1 = (1 - arrowV.y) / 2 * H();
          const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy);
          // Direção quase perpendicular à tela: encurta para não virar um ponto
          const ln = arrowLines[off + k];
          ln.style.display = "";
          ln.setAttribute("x1", x0); ln.setAttribute("y1", y0);
          ln.setAttribute("x2", len < 1 ? x0 : x1); ln.setAttribute("y2", len < 1 ? y0 : y1);
          any = any || len >= 1;
        });
      });
      arrowSvg.style.display = any ? "block" : "none";
    }
    let hoverNode = -1;
    const nodeV = new THREE.Vector3();
    const occRay = new THREE.Raycaster();
    function nodeVisible(i) {
      nodeV.set(nodes[i * 3], nodes[i * 3 + 1], nodes[i * 3 + 2]);
      const d = camera.position.distanceTo(nodeV);
      occRay.set(camera.position, nodeV.clone().sub(camera.position).normalize());
      const hit = occRay.intersectObject(mesh)[0];
      return !hit || hit.distance >= d - diag * 0.01;
    }
    function screenOf(i) {
      nodeV.set(nodes[i * 3], nodes[i * 3 + 1], nodes[i * 3 + 2]).project(camera);
      return [(nodeV.x + 1) / 2 * W(), (1 - nodeV.y) / 2 * H()];
    }
    function place(el, i) {
      const [x, y] = screenOf(i);
      el.style.display = "block";
      el.style.left = x + "px";
      el.style.top = y + "px";
    }
    function updateOverlays() {
      if (flagEl) {
        if (maxNode >= 0 && nodeVisible(maxNode)) place(flagEl, maxNode); else flagEl.style.display = "none";
      }
      for (const f of corrFlags) {
        if (nodeVisible(f.node)) place(f.el, f.node); else f.el.style.display = "none";
      }
      if (sel.node !== null && nodeVisible(sel.node)) place(selEl, sel.node); else selEl.style.display = "none";
      if (hoverNode >= 0 && nodeVisible(hoverNode)) {
        place(dotEl, hoverNode);
        place(tipEl, hoverNode);
        const i = hoverNode * 3;
        tipEl.textContent = data.field
          ? `${data.field_label}: ${data.field[hoverNode].toPrecision(5)}  (nó ${hoverNode})`
          : `Nó ${hoverNode}  (${nodes[i].toFixed(1)}, ${nodes[i + 1].toFixed(1)}, ${nodes[i + 2].toFixed(1)})`;
        updateArrow(hoverNode);
      } else { dotEl.style.display = tipEl.style.display = "none"; updateArrow(-1); }
    }
    function pickNode(clientX, clientY) {
      if (!data.field && !nodeMode) return -1;
      const r = canvas.getBoundingClientRect();
      const px = clientX - r.left, py = clientY - r.top;
      let best = -1, bestD = 12;
      for (let i = 0; i < nodes.length / 3; i++) {
        nodeV.set(nodes[i * 3], nodes[i * 3 + 1], nodes[i * 3 + 2]).project(camera);
        if (nodeV.z > 1) continue;
        const d = Math.hypot(px - (nodeV.x + 1) / 2 * r.width, py - (1 - nodeV.y) / 2 * r.height);
        if (d < bestD && nodeVisible(i)) { best = i; bestD = d; }
      }
      return best;
    }
    function draw() {
      renderer.setScissorTest(false);
      renderer.autoClear = true;
      renderer.setViewport(0, 0, W(), H());
      renderer.render(scene, camera);
      const s = Math.min(110, Math.min(W(), H()) * 0.3);
      const dir = camera.position.clone().sub(new THREE.Vector3(...cam.target)).normalize();
      axCam.position.copy(dir).multiplyScalar(5.6);
      axCam.up.copy(camera.up);
      axCam.lookAt(0, 0, 0);
      renderer.setViewport(8, 8, s, s);
      renderer.setScissor(8, 8, s, s);
      renderer.setScissorTest(true);
      renderer.autoClear = false;
      renderer.clearDepth();
      renderer.render(axScene, axCam);
      renderer.setScissorTest(false);
      renderer.autoClear = true;
      updateOverlays();
    }
    function updateCamera() {
      const t = new THREE.Vector3(...cam.target);
      cam.phi = Math.min(Math.max(cam.phi, 0.02), Math.PI - 0.02);
      camera.position.set(
        t.x + cam.dist * Math.sin(cam.phi) * Math.cos(cam.theta),
        t.y + cam.dist * Math.sin(cam.phi) * Math.sin(cam.theta),
        t.z + cam.dist * Math.cos(cam.phi)
      );
      camera.lookAt(t);
      light.position.copy(camera.position);
      draw();
    }

    // Picking.
    const raycaster = new THREE.Raycaster();
    function pick(clientX, clientY) {
      const r = canvas.getBoundingClientRect();
      const px = clientX - r.left, py = clientY - r.top;
      const ndc = new THREE.Vector2((px / r.width) * 2 - 1, -(py / r.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      const hit = raycaster.intersectObject(mesh)[0];
      const hitDist = hit ? hit.distance : Infinity;

      // Closest curve in screen space that is not hidden behind the surface.
      let best = null, bestD = 9;
      const pa = new THREE.Vector3(), pb = new THREE.Vector3();
      for (const s of segs) {
        pa.set(s[1], s[2], s[3]).project(camera);
        pb.set(s[4], s[5], s[6]).project(camera);
        const ax = (pa.x + 1) / 2 * r.width, ay = (1 - pa.y) / 2 * r.height;
        const bx = (pb.x + 1) / 2 * r.width, by = (1 - pb.y) / 2 * r.height;
        const dx = bx - ax, dy = by - ay;
        const len2 = dx * dx + dy * dy;
        let u = len2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
        u = Math.max(0, Math.min(1, u));
        const d = Math.hypot(px - (ax + u * dx), py - (ay + u * dy));
        if (d < bestD) {
          const wx = s[1] + u * (s[4] - s[1]), wy = s[2] + u * (s[5] - s[2]), wz = s[3] + u * (s[6] - s[3]);
          const dist = camera.position.distanceTo(new THREE.Vector3(wx, wy, wz));
          if (dist <= hitDist + diag * 0.01) { best = s[0]; bestD = d; }
        }
      }
      if (best !== null) return { type: "curve", id: best };
      if (hit) return { type: "surface", id: triSurf[hit.faceIndex] };
      return null;
    }

    function emit() {
      setStateValue("selection", { curves: [...sel.curves], surfaces: [...sel.surfaces], node: sel.node });
    }

    // Pointer interaction.
    let down = null, moved = false, hoverPending = false;
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", (e) => {
      down = { x: e.clientX, y: e.clientY, button: e.button, pan: e.button !== 0 };
      moved = false;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (down) {
        const dx = e.clientX - down.x, dy = e.clientY - down.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
        if (moved) {
          if (down.pan) {
            const k = (cam.dist * Math.tan((camera.fov * Math.PI) / 360) * 2) / H();
            const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
            const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
            cam.target[0] += (-dx * right.x + dy * up.x) * k;
            cam.target[1] += (-dx * right.y + dy * up.y) * k;
            cam.target[2] += (-dx * right.z + dy * up.z) * k;
          } else {
            cam.theta -= dx * 0.008;
            cam.phi -= dy * 0.008;
          }
          down.x = e.clientX; down.y = e.clientY;
          updateCamera();
        }
        return;
      }
      if (hoverPending) return;
      hoverPending = true;
      requestAnimationFrame(() => {
        hoverPending = false;
        if (disposed) return;
        const h = nodeMode ? null : pick(e.clientX, e.clientY);
        const hn = pickNode(e.clientX, e.clientY);
        if (JSON.stringify(h) !== JSON.stringify(hover) || hn !== hoverNode) {
          hover = h;
          hoverNode = hn;
          canvas.style.cursor = (h || (nodeMode && hn >= 0)) ? "pointer" : "default";
          recolor();
          draw();
        }
      });
    });
    canvas.addEventListener("pointerleave", () => {
      if (hoverNode >= 0) { hoverNode = -1; draw(); }
    });
    canvas.addEventListener("pointerup", (e) => {
      const wasClick = down && !moved && down.button === 0;
      down = null;
      if (!wasClick) return;
      if (nodeMode) {
        const n = pickNode(e.clientX, e.clientY);
        sel.node = n >= 0 ? n : null;
        draw();
        emit();
        return;
      }
      const h = pick(e.clientX, e.clientY);
      const multi = e.shiftKey || e.ctrlKey || e.metaKey;
      if (!multi) { sel.curves.clear(); sel.surfaces.clear(); }
      if (h) {
        const set = h.type === "curve" ? sel.curves : sel.surfaces;
        if (multi && set.has(h.id)) set.delete(h.id); else set.add(h.id);
      }
      recolor();
      draw();
      emit();
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      cam.dist = Math.min(Math.max(cam.dist * Math.exp(e.deltaY * 0.0012), diag * 0.02), diag * 20);
      updateCamera();
    }, { passive: false });

    const ro = new ResizeObserver(() => {
      renderer.setSize(W(), H());
      camera.aspect = W() / H();
      camera.updateProjectionMatrix();
      updateCamera();
    });
    ro.observe(wrap);

    recolor();
    updateCamera();
    hint.textContent = nodeMode
      ? "Modo nó (corrosão): passe o mouse sobre um nó e clique para selecioná-lo · arrastar: girar · " +
        "botão direito: mover · roda: zoom"
      : "Clique: seleciona aresta/superfície · Shift+clique: soma à seleção · " +
        "arrastar: girar · botão direito: mover · roda: zoom";

    return () => {
      ro.disconnect();
      renderer.dispose();
      geo.dispose();
      cgeo.dispose();
      canvas.remove();
    };
  }

  return dispose;
}
