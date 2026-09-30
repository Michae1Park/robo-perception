"""Rigid transforms (numpy). T_a_b is a 4x4 array mapping points in frame b to frame a. Quaternions are (x, y, z, w)."""
import numpy as np


def make_T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def invert(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T, pts):
    return np.asarray(pts, dtype=float) @ T[:3, :3].T + T[:3, 3]


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def angle_between(u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    return float(np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v)))


def project(K, T_cam_world, pts_world):
    """World points -> (pixels Nx2, depth N)."""
    pc = transform_points(T_cam_world, pts_world)
    z = pc[:, 2]
    return (pc @ K.T)[:, :2] / np.maximum(z, 1e-9)[:, None], z


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Pose of an OpenCV camera (+z forward, +y down) at `eye` looking at `target`."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return make_T(np.stack([x, np.cross(z, x), z], axis=1), eye)


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    """-> (x, y, z, w) with w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1.0)
        q = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], 0.25 * s * s]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[i], q[j], q[k], q[3] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s, (R[k, j] - R[j, k]) / s
    return q if q[3] >= 0 else -q
