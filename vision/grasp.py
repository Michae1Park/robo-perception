"""Analytic parallel-jaw grasp candidates from the object's bounding box.

Convention (Franka-style hand): TCP frame +z = approach direction, fingers close along +/-y.
"""
from dataclasses import dataclass

import numpy as np

from . import transforms as tf

def read_obj_vertices(path):
    """Vertices (N, 3) of a Wavefront OBJ (no trimesh dependency)."""
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
    return np.asarray(verts, dtype=float)


DOWN = np.array([0.0, 0.0, -1.0])


@dataclass(eq=False)
class GraspCandidate:
    T_obj_tcp: np.ndarray   # TCP pose in object frame
    width: float            # object width between the fingers
    face: str               # e.g. '+z/x' = approach through the +z face, closing along x
    T_world_tcp: np.ndarray = None
    tilt: float = 0.0       # rad between approach and world -z


def generate_candidates(lo, hi, max_opening=0.08, width_margin=0.012, finger_depth=0.03):
    """Grasps through each of the 6 box faces, closing along each of the 2 in-face axes (both signs)."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    center = (lo + hi) / 2.0
    ext = hi - lo
    eye = np.eye(3)
    out = []
    for a in range(3):
        for s in (1.0, -1.0):
            n = s * eye[a]                    # outward normal of the approached face
            z = -n                            # approach direction
            depth = min(finger_depth, ext[a] / 2.0)
            tcp = center + n * (ext[a] / 2.0 - depth)
            for c in range(3):
                if c == a or ext[c] + width_margin > max_opening:
                    continue
                for sy in (1.0, -1.0):
                    y = sy * eye[c]
                    x = np.cross(y, z)
                    R = np.stack([x, y, z], axis=1)
                    face = '%s%s/%s%s' % ('+' if s > 0 else '-', 'xyz'[a], '+' if sy > 0 else '-', 'xyz'[c])
                    out.append(GraspCandidate(tf.make_T(R, tcp), float(ext[c]), face))
    return out


def filter_candidates(cands, T_world_obj, table_z, max_tilt_deg=50.0, clearance=0.012, finger_thickness=0.01):
    """Keep candidates approaching from above whose fingertips stay above the table; sort by tilt."""
    keep = []
    max_tilt = np.deg2rad(max_tilt_deg)
    for c in cands:
        T = T_world_obj @ c.T_obj_tcp
        tilt = tf.angle_between(T[:3, 2], DOWN)
        if tilt > max_tilt:
            continue
        y = T[:3, 1]
        tips = [T[:3, 3] + y * (c.width / 2.0 + finger_thickness),
                T[:3, 3] - y * (c.width / 2.0 + finger_thickness)]
        if min(p[2] for p in tips + [T[:3, 3]]) < table_z + clearance:
            continue
        keep.append(GraspCandidate(c.T_obj_tcp, c.width, c.face, T, tilt))
    keep.sort(key=lambda g: g.tilt)
    return keep
