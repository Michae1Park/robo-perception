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


def gripper_hits(T_world_tcp, width, points_world, max_opening=0.08, finger_len=0.045, finger_thick=0.01,
                 finger_wide=0.02, palm_h=0.02):
    """How many `points_world` lie inside the gripper at this pose: the two fingers (the whole stroke from
    fully open to closed on `width`) and the palm behind them. TCP frame: +z approach, fingers along +/-y."""
    loc = tf.transform_points(tf.invert(T_world_tcp), points_world)
    x, y, z = np.abs(loc[:, 0]), np.abs(loc[:, 1]), loc[:, 2]
    reach = max_opening / 2.0 + finger_thick
    fingers = (x <= finger_wide / 2) & (y >= width / 2.0) & (y <= reach) & (z >= -finger_len) & (z <= 0)
    palm = (x <= finger_wide / 2) & (y <= reach) & (z >= -finger_len - palm_h) & (z < -finger_len)
    return int(np.count_nonzero(fingers | palm))


def evaluate(c, T_world_obj, table_z, max_tilt_deg=50.0, clearance=0.012, finger_thickness=0.01,
             obstacles=None, max_opening=0.08, min_hits=10):
    """World pose, approach tilt (rad), lowest point of TCP + fingertips (m), scene points hit, and why it
    fails ('' = passes). `obstacles`: scene points (world) that are neither table nor the object itself."""
    T = T_world_obj @ c.T_obj_tcp
    tilt = tf.angle_between(T[:3, 2], DOWN)
    y = T[:3, 1]
    tips = [T[:3, 3] + y * (c.width / 2.0 + finger_thickness),
            T[:3, 3] - y * (c.width / 2.0 + finger_thickness)]
    lowest = min(p[2] for p in tips + [T[:3, 3]])
    hits = 0 if obstacles is None else gripper_hits(T, c.width, obstacles, max_opening, finger_thick=finger_thickness)
    if tilt > np.deg2rad(max_tilt_deg):
        why = 'approach too tilted'
    elif lowest < table_z + clearance:
        why = 'fingertips hit the table'
    elif hits >= min_hits:
        why = 'gripper hits the scene'
    else:
        why = ''
    return T, tilt, lowest, hits, why


def filter_candidates(cands, T_world_obj, table_z, max_tilt_deg=50.0, clearance=0.012, finger_thickness=0.01,
                      obstacles=None, max_opening=0.08):
    """Keep candidates approaching from above whose fingertips stay above the table (and, with `obstacles`,
    that don't run into other scene points); sort by tilt."""
    keep = []
    for c in cands:
        T, tilt, _, _, why = evaluate(c, T_world_obj, table_z, max_tilt_deg, clearance, finger_thickness,
                                      obstacles, max_opening)
        if not why:
            keep.append(GraspCandidate(c.T_obj_tcp, c.width, c.face, T, tilt))
    keep.sort(key=lambda g: g.tilt)
    return keep
