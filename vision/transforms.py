"""Small rigid-transform toolkit (numpy only).

Convention: homogeneous transforms are 4x4 float64 arrays; T_a_b maps points in frame b to frame a.
"""
import numpy as np


def make_T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def invert(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T, pts):
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def angle_between(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return float(np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v)))


def project(K, T_cam_world, pts_world):
    """Project world points into pixels. Returns (uv Nx2, depth N)."""
    pc = transform_points(T_cam_world, pts_world)
    z = pc[:, 2]
    uv = (pc @ K.T)[:, :2] / np.maximum(z, 1e-9)[:, None]
    return uv, z
