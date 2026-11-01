import os
import tempfile
import threading
from dataclasses import dataclass

import gmsh
import numpy as np


_LOCK = threading.Lock()


@dataclass
class ShellMesh:
    nodes: np.ndarray  # (N, 3)
    triangles: np.ndarray  # (M, 3) zero-based node indices
    bbox_diagonal: float
    tri_surface: np.ndarray  # (M,) geometric surface tag of each triangle
    curves: dict[int, np.ndarray]  # curve tag -> (K, 2) node index pairs
    curve_length: dict[int, float]  # curve tag -> length in model units
    welded: int = 0  # duplicate nodes merged by preprocessing


def _weld(nodes, tri, curves, diag):
    """Merge coincident nodes so surfaces meeting at an edge share them (no gaps under load).

    Also drops nodes not used by any element (they would be singular DOFs).
    """
    tol = diag * 1e-6
    key = np.round(nodes / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    # Representative node of each group, then compact to nodes referenced by elements.
    rep = first[inverse]
    tri = rep[tri]
    used = np.unique(tri)
    new_index = np.full(len(nodes), -1, dtype=np.int64)
    new_index[used] = np.arange(len(used))
    welded = int(len(nodes) - len(np.unique(rep)))
    out_curves = {}
    for tag, seg in curves.items():
        seg = new_index[rep[seg]]
        seg = seg[(seg >= 0).all(axis=1) & (seg[:, 0] != seg[:, 1])]
        out_curves[tag] = seg
    return nodes[used], new_index[tri], out_curves, welded


def free_curves(mesh: "ShellMesh") -> list[int]:
    """Tags of free edges: curves whose every segment belongs to a single triangle."""
    tri = mesh.triangles
    nn = len(mesh.nodes)
    pairs = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    keys, counts = np.unique(pairs[:, 0].astype(np.int64) * nn + pairs[:, 1], return_counts=True)
    free = []
    for tag, seg in mesh.curves.items():
        if len(seg) == 0:
            continue
        sk = np.sort(seg, axis=1)
        pos = np.searchsorted(keys, sk[:, 0].astype(np.int64) * nn + sk[:, 1])
        pos = np.minimum(pos, len(keys) - 1)
        found = keys[pos] == sk[:, 0].astype(np.int64) * nn + sk[:, 1]
        if found.all() and (counts[pos] == 1).all():
            free.append(int(tag))
    return sorted(free)


def mesh_step_shell(step_bytes: bytes, element_size: float | None = None,
                    divisions: int = 20, refinements: int = 0,
                    suffix: str = ".step") -> ShellMesh:
    """Generate a triangular shell mesh from STEP or STL bytes.

    element_size: target size; if None, bbox diagonal / divisions is used (STEP only).
    refinements: number of uniform refinement passes (each halves the size).
    STL files already are triangle meshes, so they are used as-is and only refined.
    """
    is_stl = suffix.lower() == ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(step_bytes)
        path = f.name

    _LOCK.acquire()
    gmsh.initialize(interruptible=False)  # signal handlers only work in the main thread
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("shell")
        if is_stl:
            gmsh.merge(path)
            # Split the STL into faces/edges at sharp angles so they can be selected.
            gmsh.model.mesh.classifySurfaces(np.radians(40), True, False, np.pi)
            gmsh.model.mesh.createGeometry()
        else:
            gmsh.model.occ.importShapes(path)
            # Merge coincident vertices/edges of separate faces so adjacent surfaces share them.
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
        if not gmsh.model.getEntities(2):
            raise ValueError("O arquivo não contém superfícies (casca).")

        if not is_stl:
            size = element_size or diag / divisions
            gmsh.option.setNumber("Mesh.MeshSizeMax", size)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
            gmsh.model.mesh.generate(2)
        for _ in range(refinements):
            gmsh.model.mesh.refine()

        tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.asarray(coords).reshape(-1, 3)
        lookup = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        lookup[tags] = np.arange(len(tags))

        tri_list, surf_list = [], []
        for _, stag in gmsh.model.getEntities(2):
            types, _, node_tags = gmsh.model.mesh.getElements(2, stag)
            for t, nt in zip(types, node_tags):
                if t == 2:
                    nt = np.asarray(nt, dtype=np.int64).reshape(-1, 3)
                    tri_list.append(lookup[nt])
                    surf_list.append(np.full(len(nt), stag))
        if not tri_list:
            raise ValueError("Nenhum elemento triangular foi gerado.")

        curves = {}
        lengths = {}
        for _, ctag in gmsh.model.getEntities(1):
            types, _, node_tags = gmsh.model.mesh.getElements(1, ctag)
            for t, nt in zip(types, node_tags):
                if t == 1:
                    seg = lookup[np.asarray(nt, dtype=np.int64).reshape(-1, 2)]
                    curves[ctag] = seg
                    lengths[ctag] = float(np.linalg.norm(
                        nodes[seg[:, 1]] - nodes[seg[:, 0]], axis=1).sum())
                    if not is_stl:
                        # Exact CAD length (the polyline underestimates arcs).
                        try:
                            lengths[ctag] = float(gmsh.model.occ.getMass(1, ctag))
                        except Exception:
                            pass
        nodes, tri, curves, welded = _weld(nodes, np.vstack(tri_list), curves, diag)
        return ShellMesh(nodes, tri, diag, np.concatenate(surf_list), curves, lengths, welded)
    finally:
        gmsh.finalize()
        _LOCK.release()
        os.unlink(path)
