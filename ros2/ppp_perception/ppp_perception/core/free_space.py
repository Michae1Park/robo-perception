"""Occupancy of a rectangular table region from a world-frame point cloud."""
import numpy as np

from .plane import height_above

FREE, OCCUPIED, UNKNOWN = 0, 100, -1


def grid_shape(zone, resolution):
    (x0, x1), (y0, y1) = zone
    nx = int(np.ceil((x1 - x0) / resolution - 1e-9))
    ny = int(np.ceil((y1 - y0) / resolution - 1e-9))
    return ny, nx


def occupancy_grid(points_world, plane, zone, resolution=0.01, height_thresh=0.012, max_height=0.6,
                   min_points=3):
    """Grid[iy, ix] over `zone` with FREE / OCCUPIED / UNKNOWN (no depth return, e.g. shadows).

    A cell is OCCUPIED if at least `min_points` points in it (robust to flying pixels) are between
    `height_thresh` and `max_height` above the plane, FREE if it only holds table points, UNKNOWN if nothing was observed.
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


def cell_centers(zone, resolution, shape):
    (x0, _), (y0, _) = zone
    ny, nx = shape
    xs = x0 + (np.arange(nx) + 0.5) * resolution
    ys = y0 + (np.arange(ny) + 0.5) * resolution
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=-1)  # (ny, nx, 2)


def clearance_map(grid, zone, resolution):
    """Distance (m) from each cell centre to the nearest non-free cell or the zone border."""
    (x0, x1), (y0, y1) = zone
    centers = cell_centers(zone, resolution, grid.shape)
    border = np.minimum.reduce([centers[..., 0] - x0, x1 - centers[..., 0],
                                centers[..., 1] - y0, y1 - centers[..., 1]])
    blocked = centers[grid != FREE]
    if len(blocked) == 0:
        return border
    flat = centers.reshape(-1, 2)
    d = np.full(len(flat), np.inf)
    for i in range(0, len(blocked), 256):
        chunk = blocked[i:i + 256]
        dd = np.linalg.norm(flat[:, None, :] - chunk[None, :, :], axis=-1).min(axis=1)
        d = np.minimum(d, dd)
    # the blocked cell may extend up to half a cell towards us
    d = np.maximum(d - resolution / 2.0, 0.0).reshape(grid.shape)
    return np.minimum(d, border)
