import json
from pathlib import Path

import streamlit as st
import numpy as np

from mesher import ShellMesh

_DIR = Path(__file__).parent

_HTML = """<div class="box"><div class="wrap"></div>
<div class="legend"><div class="lg-title"></div><div class="lg-max"></div>
<div class="lg-bar"></div><div class="lg-min"></div></div></div><div class="hint"></div>"""
_CSS = """
.box { position: relative; }
.wrap { height: 680px; width: 100%; border: 1px solid #d0d5dd; border-radius: 6px;
        overflow: hidden; }
.wrap canvas { display: block; }
.hint { font: 12px sans-serif; color: #667085; padding-top: 4px; }
.legend { position: absolute; top: 10px; right: 10px; display: none; flex-direction: column;
          align-items: flex-end; font: 12px sans-serif; color: #1f2933; gap: 3px;
          background: rgba(255,255,255,0.8); padding: 6px 8px; border-radius: 4px; }
.lg-bar { width: 18px; height: 240px; border: 1px solid #667085; }
.lg-title { font-weight: 600; }
"""


@st.cache_resource
def _component():
    three = (_DIR / "vendor" / "three.module.min.js").read_text(encoding="utf8")
    js = "const THREE_SRC = " + json.dumps(three) + ";\n" + (_DIR / "viewer.js").read_text(encoding="utf8")
    return st.components.v2.component("pipe_fea_viewer", html=_HTML, css=_CSS, js=js)


def mesh_viewer(mesh: ShellMesh, sig: str, selection: dict, curve_bc: dict,
                surf_p: dict, show_mesh: bool, key: str = "viewer",
                nodes=None, ghost=None, field=None, field_label: str = "",
                mode: str = "edge", corr_tris=None, corr_flags=None, max_note: str = "",
                                field_dir=None, field_dir2=None, crack_edges=None, crack_flags=None):
    """Render the mesh; returns the component result (result.selection holds the picks).

    nodes: optional replacement coordinates (e.g. deformed shape).
    ghost: optional reference coordinates drawn as a grey wireframe (undeformed shape).
    field: optional per-node scalar used to colour the surface (with a legend).
    mode: "edge" picks edges/surfaces; "node" picks a single node (selection["node"]).
    corr_tris: triangle indices highlighted as corroded zones.
    """
    nodes = mesh.nodes if nodes is None else nodes
    data = {
        "sig": sig,
        "diag": mesh.bbox_diagonal,
        "nodes": nodes.round(6).ravel().tolist(),
        "tris": mesh.triangles.ravel().tolist(),
        "tri_surf": mesh.tri_surface.tolist(),
        "curves": {str(k): v.ravel().tolist() for k, v in mesh.curves.items()},
        "selection": selection,
        "curve_bc": {str(k): v for k, v in curve_bc.items()},
        "surf_p": {str(k): v for k, v in surf_p.items()},
        "show_mesh": show_mesh,
        "mode": mode,
        "corr_tris": [] if corr_tris is None else [int(i) for i in corr_tris],
        "corr_flags": corr_flags or [],
        "max_note": max_note,
        "crack_edges": [int(t) for t in (crack_edges or [])],
        "crack_flags": crack_flags or [],
    }
    if ghost is not None:
        data["ghost_nodes"] = np.asarray(ghost).round(6).ravel().tolist()
    if field is not None and field_dir is not None:
        data["field_dir"] = np.asarray(field_dir).round(5).ravel().tolist()
        if field_dir2 is not None:
            data["field_dir2"] = np.asarray(field_dir2).round(5).ravel().tolist()
    if field is not None:
        data.update(field=np.asarray(field).round(6).tolist(), field_label=field_label,
                    field_min=float(np.min(field)), field_max=float(np.max(field)))
    return _component()(data=data, key=key, on_selection_change=lambda: None)
