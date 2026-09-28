"""Small rigid-transform toolkit (numpy only).

Conventions:
  * Quaternions are ROS order (x, y, z, w) unless a function says otherwise.
  * Homogeneous transforms are 4x4 float64 arrays; T_a_b maps points in frame b to frame a.
"""
import numpy as np


def quat_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def mat_to_quat(R):
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def make_T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def T_from_pq(p, q):
    return make_T(quat_to_mat(q), p)


def pq_from_T(T):
    return np.array(T[:3, 3], dtype=float), mat_to_quat(T[:3, :3])


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


def rotation_angle(Ra, Rb):
    """Geodesic angle (rad) between two rotation matrices."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def angle_between(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return float(np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v)))


def look_at_ros(eye, target, up=(0.0, 0.0, 1.0)):
    """Rotation of a ROS optical frame (z forward, x right, y down) looking at target."""
    eye = np.asarray(eye, dtype=float)
    z = np.asarray(target, dtype=float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=float))
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def intrinsics_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def project(K, T_cam_world, pts_world):
    """Project world points into pixels. Returns (uv Nx2, depth N)."""
    pc = transform_points(T_cam_world, pts_world)
    z = pc[:, 2]
    uv = (pc @ K.T)[:, :2] / np.maximum(z, 1e-9)[:, None]
    return uv, z


def depth_to_points(depth, K, stride=1):
    """Back-project a metric depth image (HxW) to camera-frame points; drops invalid pixels."""
    d = depth[::stride, ::stride]
    h, w = d.shape
    vs, us = np.mgrid[0:h, 0:w]
    us = us * stride
    vs = vs * stride
    valid = np.isfinite(d) & (d > 0)
    z = d[valid]
    x = (us[valid] - K[0, 2]) * z / K[0, 0]
    y = (vs[valid] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)
