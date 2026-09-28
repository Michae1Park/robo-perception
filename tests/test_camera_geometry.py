"""Synthetic depth from the configured camera -> world points -> plane + place-zone occupancy.

Checks the camera conventions shared by the simulator and the perception nodes end to end.
"""
import numpy as np

from ppp_common import transforms as tf
from ppp_common.config import SceneConfig
from ppp_perception.core import free_space as fs
from ppp_perception.core.plane import fit_plane_ransac, plane_z_at


def render_depth(cfg, boxes):
    """Ray-cast the table plane z = table_height plus axis-aligned boxes (x0, x1, y0, y1, height)."""
    c = cfg.camera
    K = cfg.camera_K()
    T = cfg.T_world_camera()
    v, u = np.mgrid[0:c['height'], 0:c['width']]
    rays_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    rays = rays_cam @ T[:3, :3].T               # world direction for depth (z_cam) = 1
    eye = T[:3, 3]
    zt = cfg.table_height
    depth = np.where(rays[..., 2] < 0, (zt - eye[2]) / rays[..., 2], 0.0)
    for x0, x1, y0, y1, h in boxes:
        t = (zt + h - eye[2]) / rays[..., 2]
        p = eye + rays * t[..., None]
        hit = (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1) & (t > 0)
        depth = np.where(hit & (t < depth), t, depth)
    return depth.astype(np.float32)


def test_camera_sees_both_zones_and_recovers_table():
    cfg = SceneConfig()
    K, T = cfg.camera_K(), cfg.T_world_camera()
    corners = []
    for zone in ('pick', 'place'):
        (x0, x1), (y0, y1) = cfg.zone(zone)
        corners += [[x, y, cfg.table_height] for x in (x0, x1) for y in (y0, y1)]
    uv, z = tf.project(K, tf.invert(T), np.array(corners))
    assert (z > 0).all()
    assert (uv[:, 0] > 0).all() and (uv[:, 0] < cfg.camera['width']).all()
    assert (uv[:, 1] > 0).all() and (uv[:, 1] < cfg.camera['height']).all()

    (px0, _), (py0, _) = cfg.zone('place')
    box = (px0 + 0.10, px0 + 0.16, py0 + 0.10, py0 + 0.16, 0.06)
    depth = render_depth(cfg, [box])
    pts = tf.transform_points(T, tf.depth_to_points(depth, K, stride=2))
    plane, _ = fit_plane_ransac(pts)
    assert plane is not None and abs(plane_z_at(plane, 0.5, 0.0) - cfg.table_height) < 1e-3

    grid = fs.occupancy_grid(pts, plane, cfg.zone('place'), 0.01)
    centers = fs.cell_centers(cfg.zone('place'), 0.01, grid.shape)
    in_box = (centers[..., 0] > box[0] + 0.01) & (centers[..., 0] < box[1] - 0.01) & \
             (centers[..., 1] > box[2] + 0.01) & (centers[..., 1] < box[3] - 0.01)
    assert (grid[in_box] == fs.OCCUPIED).all()
    far = np.hypot(centers[..., 0] - (box[0] + box[1]) / 2, centers[..., 1] - (box[2] + box[3]) / 2) > 0.15
    assert (grid[far] == fs.FREE).mean() > 0.95
