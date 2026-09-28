"""Dominant support-plane estimation with RANSAC + least-squares refinement."""
import numpy as np


def _plane_from_points(p):
    n = np.cross(p[1] - p[0], p[2] - p[0])
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    n /= norm
    return np.append(n, -np.dot(n, p[0]))


def refine_plane(points):
    """Least-squares plane through points (SVD). Returns (a, b, c, d) with unit normal."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[-1]
    return np.append(n, -np.dot(n, c))


def orient(plane, up):
    return plane if np.dot(plane[:3], up) >= 0 else -plane


def fit_plane_ransac(points, dist_thresh=0.006, iters=300, up=(0.0, 0.0, 1.0),
                     max_tilt_deg=20.0, min_inliers=200, seed=0):
    """Largest plane whose normal is within `max_tilt_deg` of `up`.

    Returns (plane[4], inlier_mask) or (None, None). The normal points along `up`.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < max(3, min_inliers):
        return None, None
    rng = np.random.default_rng(seed)
    up = np.asarray(up, dtype=float) / np.linalg.norm(up)
    cos_tilt = np.cos(np.deg2rad(max_tilt_deg))
    best, best_count = None, 0
    for _ in range(iters):
        plane = _plane_from_points(points[rng.choice(len(points), 3, replace=False)])
        if plane is None or abs(np.dot(plane[:3], up)) < cos_tilt:
            continue
        count = int(np.count_nonzero(np.abs(points @ plane[:3] + plane[3]) < dist_thresh))
        if count > best_count:
            best, best_count = plane, count
    if best is None or best_count < min_inliers:
        return None, None
    inliers = np.abs(points @ best[:3] + best[3]) < dist_thresh
    plane = orient(refine_plane(points[inliers]), up)
    inliers = np.abs(points @ plane[:3] + plane[3]) < dist_thresh
    return plane, inliers


def height_above(plane, points):
    return np.asarray(points) @ plane[:3] + plane[3]


def plane_z_at(plane, x, y):
    a, b, c, d = plane
    return -(a * x + b * y + d) / c
