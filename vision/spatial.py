"""Stage 3: depth -> point cloud -> table plane (RANSAC) -> occupancy grid -> free-space clearance."""
import cv2
import numpy as np

FREE, OCCUPIED, UNKNOWN = 0, 100, -1


# ---------------------------------------------------------------- point cloud
def depth_to_points(depth, K, stride=1):
    """Back-project a metric depth image (HxW) to camera-frame points; drops invalid pixels."""
    d = depth[::stride, ::stride]
    h, w = d.shape
    vs, us = np.mgrid[0:h, 0:w]
    us, vs = us * stride, vs * stride
    valid = np.isfinite(d) & (d > 0)
    z = d[valid]
    x = (us[valid] - K[0, 2]) * z / K[0, 0]
    y = (vs[valid] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------- plane
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


# ---------------------------------------------------------------- occupancy grid
def grid_shape(zone, resolution):
    (x0, x1), (y0, y1) = zone
    nx = int(np.ceil((x1 - x0) / resolution - 1e-9))
    ny = int(np.ceil((y1 - y0) / resolution - 1e-9))
    return ny, nx


def occupancy_grid(points_world, plane, zone, resolution=0.01, height_thresh=0.012, max_height=0.6,
                   min_points=3):
    """Grid[iy, ix] over `zone` with FREE / OCCUPIED / UNKNOWN (no depth return, e.g. shadows).

    A cell is OCCUPIED if at least `min_points` points in it (robust to flying pixels) are between
    `height_thresh` and `max_height` above the plane, FREE if it only holds table points, UNKNOWN if
    nothing was observed.
    """
    (x0, x1), (y0, y1) = zone
    ny, nx = grid_shape(zone, resolution)
    grid = np.full((ny, nx), UNKNOWN, dtype=np.int8)
    p = np.asarray(points_world, dtype=float)
    m = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
    p = p[m]
    if len(p) == 0:
        return grid
    h = height_above(plane, p)
    ix = np.clip(((p[:, 0] - x0) / resolution).astype(int), 0, nx - 1)
    iy = np.clip(((p[:, 1] - y0) / resolution).astype(int), 0, ny - 1)
    table = np.abs(h) < height_thresh
    obstacle = (h >= height_thresh) & (h < max_height)
    grid[iy[table], ix[table]] = FREE
    counts = np.zeros((ny, nx), dtype=int)
    np.add.at(counts, (iy[obstacle], ix[obstacle]), 1)
    grid[counts >= min_points] = OCCUPIED
    return grid


def clearance_map(grid, resolution):
    """Distance (m) from each cell centre to the nearest non-free cell or the grid border (0 on non-free cells)."""
    free = np.pad((grid == FREE).astype(np.uint8), 1)             # the border counts as blocked
    d = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return np.maximum(d * resolution - resolution / 2.0, 0.0) * (grid == FREE)   # a blocked cell reaches half a cell


# ---------------------------------------------------------------- whole stage
def analyze_scene(depth_m, K, cfg):
    """Depth -> table plane + occupancy grid, in a plane-aligned "pseudo-world" frame.

    No robot calibration exists for the demo images, so the dominant plane defines the frame:
    table normal = +Z, table height = 0, X = image right projected onto the table, Y = away from the camera.
    `cfg` is the `spatial` section of config.yaml. Returns a dict (raises RuntimeError if no plane):
      T_world_cam, plane_cam, inliers, n_points, zone, grid, clearance
    """
    pts_cam = depth_to_points(depth_m, K)
    # up/tilt filter disabled (max_tilt_deg=180): camera orientation vs. gravity is unknown here
    plane_cam, inliers = fit_plane_ransac(pts_cam, dist_thresh=cfg['ransac_dist'], iters=cfg['ransac_iters'],
                                          max_tilt_deg=180, min_inliers=cfg['min_inliers'])
    if plane_cam is None:
        raise RuntimeError('RANSAC found no dominant table plane')
    if plane_cam[3] < 0:  # put the camera origin on the positive side
        plane_cam = -plane_cam
    n = plane_cam[:3]
    p0 = -plane_cam[3] * n  # a point on the plane
    u = np.array([1.0, 0.0, 0.0]) - n[0] * n  # X = camera's x (image right) projected onto the table
    u /= np.linalg.norm(u)
    R = np.stack([u, np.cross(n, u), n], axis=0)  # world_from_cam rotation
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R
    T_world_cam[:3, 3] = -R @ p0
    pts_world = (R @ (pts_cam - p0).T).T

    table_pts = pts_world[np.abs(pts_world[:, 2]) < 0.01]
    if len(table_pts) < 100:
        raise RuntimeError('too few table-height points to size the zone')
    x0, x1 = np.percentile(table_pts[:, 0], [2, 98])
    y0, y1 = np.percentile(table_pts[:, 1], [2, 98])
    zone = ((float(x0), float(x1)), (float(y0), float(y1)))
    grid = occupancy_grid(pts_world, np.array([0.0, 0.0, 1.0, 0.0]), zone, cfg['resolution'],
                          cfg['height_thresh'], cfg['max_height'])
    return {'T_world_cam': T_world_cam, 'plane_cam': plane_cam, 'inliers': inliers,
            'n_points': len(pts_cam), 'zone': zone, 'grid': grid,
            'clearance': clearance_map(grid, cfg['resolution'])}
