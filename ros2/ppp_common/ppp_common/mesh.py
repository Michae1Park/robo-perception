"""Minimal Wavefront OBJ reader (no trimesh dependency).

Only what the pipeline needs: vertices for geometry (grasp boxes, footprints, metrics)
and a full v/vt/f parse for building USD meshes in Isaac Sim.
"""
from dataclasses import dataclass, field
import os

import numpy as np


@dataclass
class ObjMesh:
    vertices: np.ndarray                      # (N, 3)
    uvs: np.ndarray                           # (M, 2), may be empty
    face_counts: list = field(default_factory=list)
    face_v: list = field(default_factory=list)    # flat vertex indices
    face_vt: list = field(default_factory=list)   # flat uv indices (-1 if absent)
    texture: str = ''                         # absolute path of map_Kd, if any

    @property
    def aabb(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def extents(self):
        lo, hi = self.aabb
        return hi - lo


def _index(tok, n):
    i = int(tok)
    return i - 1 if i > 0 else n + i


def read_obj_vertices(path):
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
    return np.asarray(verts, dtype=float)


def _texture_from_mtl(obj_path, mtllib):
    mtl_path = os.path.join(os.path.dirname(obj_path), mtllib)
    if not os.path.isfile(mtl_path):
        return ''
    with open(mtl_path) as f:
        for line in f:
            p = line.strip().split(None, 1)
            if len(p) == 2 and p[0] == 'map_Kd':
                tex = os.path.join(os.path.dirname(mtl_path), p[1].strip())
                return tex if os.path.isfile(tex) else ''
    return ''


def read_obj(path):
    verts, uvs = [], []
    counts, fv, fvt = [], [], []
    texture = ''
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('vt '):
                p = line.split()
                uvs.append((float(p[1]), float(p[2])))
            elif line.startswith('f '):
                toks = line.split()[1:]
                counts.append(len(toks))
                for t in toks:
                    parts = t.split('/')
                    fv.append(_index(parts[0], len(verts)))
                    if len(parts) > 1 and parts[1]:
                        fvt.append(_index(parts[1], len(uvs)))
                    else:
                        fvt.append(-1)
            elif line.startswith('mtllib ') and not texture:
                texture = _texture_from_mtl(path, line.split(None, 1)[1].strip())
    return ObjMesh(np.asarray(verts, dtype=float).reshape(-1, 3),
                   np.asarray(uvs, dtype=float).reshape(-1, 2),
                   counts, fv, fvt, texture)


def box_corners(lo, hi):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def subsample(points, n, seed=0):
    if len(points) <= n:
        return points
    idx = np.random.default_rng(seed).choice(len(points), n, replace=False)
    return points[idx]
