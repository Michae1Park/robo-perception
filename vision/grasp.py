"""Stage 4: parallel-jaw grasp candidates from the object's bounding box, filtered against the table and the scene.

TCP frame (Franka Hand): origin between the finger pads, +z = approach direction, fingers close along +/-y.
"""
from dataclasses import dataclass

import numpy as np

from . import transforms as tf

DOWN = np.array([0.0, 0.0, -1.0])
FINGER_THICK = 0.01    # finger pad thickness
FINGER_BASE = 0.0584   # panda_hand -> where the fingers start, along the approach
STOCK_TCP = 0.1034     # panda_hand -> pad centre of the stock Franka fingers


def tcp_offset(gripper_cfg):
    """panda_hand -> TCP along the approach, for fingers `finger_extension` longer than stock (config.yaml gripper)."""
    return STOCK_TCP + gripper_cfg.get('finger_extension', 0.0)


def finger_length(gripper_cfg):
    """TCP -> palm: how far behind the TCP the fingers reach."""
    return tcp_offset(gripper_cfg) - FINGER_BASE


@dataclass(eq=False)
class GraspCandidate:
    T_obj_tcp: np.ndarray   # TCP pose in the object frame
    width: float            # object width between the fingers
    face: str               # e.g. '+z/+x' = approach through the +z face, closing along x
    T_world_tcp: np.ndarray = None
    tilt: float = 0.0       # rad between the approach and straight down


def generate_candidates(lo, hi, max_opening=0.08, width_margin=0.012, finger_depth=0.03):
    """Grasps through each of the 6 box faces, closing along each of the 2 in-face axes (both signs)."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    center, ext = (lo + hi) / 2.0, hi - lo
    out = []
    for a in range(3):
        for s in (1.0, -1.0):
            n = s * np.eye(3)[a]                         # outward normal of the approached face
            tcp = center + n * (ext[a] / 2.0 - min(finger_depth, ext[a] / 2.0))
            for c in range(3):
                if c == a or ext[c] + width_margin > max_opening:
                    continue
                for sy in (1.0, -1.0):
                    y, z = sy * np.eye(3)[c], -n
                    face = '%s%s/%s%s' % ('+' if s > 0 else '-', 'xyz'[a], '+' if sy > 0 else '-', 'xyz'[c])
                    out.append(GraspCandidate(tf.make_T(np.stack([np.cross(y, z), y, z], axis=1), tcp),
                                              float(ext[c]), face))
    return out


def gripper_hits(T_world_tcp, width, points_world, max_opening=0.08, finger_len=0.045, finger_wide=0.02,
                 palm_h=0.02):
    """How many `points_world` lie inside the gripper at this pose: the two fingers (the whole stroke from fully
    open to closed on `width`) and the palm behind them."""
    loc = tf.transform_points(tf.invert(T_world_tcp), points_world)
    x, y, z = np.abs(loc[:, 0]), np.abs(loc[:, 1]), loc[:, 2]
    reach = max_opening / 2.0 + FINGER_THICK
    across = (x <= finger_wide / 2) & (y <= reach)
    fingers = across & (y >= width / 2.0) & (z >= -finger_len) & (z <= 0)
    palm = across & (z >= -finger_len - palm_h) & (z < -finger_len)
    return int(np.count_nonzero(fingers | palm))


def evaluate(c, T_world_obj, table_z, max_tilt_deg=50.0, clearance=0.012, obstacles=None, max_opening=0.08,
             finger_len=0.045, min_hits=10):
    """-> world TCP pose, approach tilt (rad), lowest point of TCP + fingertips (m), scene points hit, and why it
    fails ('' = passes). `obstacles`: scene points (world) that are neither table nor the object itself."""
    T = T_world_obj @ c.T_obj_tcp
    tilt = tf.angle_between(T[:3, 2], DOWN)
    half = T[:3, 1] * (c.width / 2.0 + FINGER_THICK)
    lowest = min(T[2, 3], T[2, 3] + half[2], T[2, 3] - half[2])
    hits = 0 if obstacles is None else gripper_hits(T, c.width, obstacles, max_opening, finger_len)
    if tilt > np.deg2rad(max_tilt_deg):
        why = 'approach too tilted'
    elif lowest < table_z + clearance:
        why = 'fingertips hit the table'
    elif hits >= min_hits:
        why = 'gripper hits the scene'
    else:
        why = ''
    return T, tilt, lowest, hits, why


def filter_candidates(cands, T_world_obj, table_z, max_tilt_deg=50.0, clearance=0.012, obstacles=None,
                      max_opening=0.08, finger_len=0.045):
    """Candidates that pass evaluate(), with their world pose and tilt, least tilted first."""
    keep = []
    for c in cands:
        T, tilt, _, _, why = evaluate(c, T_world_obj, table_z, max_tilt_deg, clearance, obstacles, max_opening,
                                      finger_len)
        if not why:
            keep.append(GraspCandidate(c.T_obj_tcp, c.width, c.face, T, tilt))
    return sorted(keep, key=lambda g: g.tilt)
