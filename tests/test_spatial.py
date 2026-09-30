import numpy as np

from vision import spatial as fs
from vision.spatial import fit_plane_ransac, plane_z_at


def _table_scene(rng, table_z=0.0, box=None):
    xy = rng.uniform([0.2, -0.5], [0.9, 0.5], size=(200000, 2))
    pts = np.column_stack([xy, table_z + rng.normal(0, 0.001, len(xy))])
    if box is not None:
        (x0, x1), (y0, y1), h = box
        bx = rng.uniform([x0, y0], [x1, y1], size=(2000, 2))
        pts = np.vstack([pts, np.column_stack([bx, np.full(len(bx), table_z + h)])])
    outliers = rng.uniform([0.2, -0.5, 0.0], [0.9, 0.5, 0.5], size=(2000, 3))
    return np.vstack([pts, outliers])


def test_ransac_finds_table():
    rng = np.random.default_rng(0)
    pts = _table_scene(rng, table_z=0.02)
    plane, inl = fit_plane_ransac(pts)
    assert plane is not None
    assert plane[2] > 0.99
    assert abs(plane_z_at(plane, 0.5, 0.0) - 0.02) < 0.002
    assert inl.sum() > 15000


def test_occupancy_and_clearance():
    rng = np.random.default_rng(0)
    zone = ((0.4, 0.6), (-0.3, -0.1))
    pts = _table_scene(rng, box=((0.40, 0.46), (-0.3, -0.24), 0.05))
    plane, _ = fit_plane_ransac(pts)
    grid = fs.occupancy_grid(pts, plane, zone, 0.01)
    assert grid.shape == (20, 20)
    assert (grid[:5, :5] == fs.OCCUPIED).all()
    assert grid[15, 15] == fs.FREE
    clear = fs.clearance_map(grid, 0.01)
    assert clear[:5, :5].max() == 0.0
    # centre (0.505, -0.195): nearest blocked cell centre (0.455, -0.245), minus half a cell
    assert abs(clear[10, 10] - (np.hypot(0.05, 0.05) - 0.005)) < 1e-6
