"""Placement: find a free spot in the place zone where the object's footprint fits."""
import numpy as np

from ppp_common import transforms as tf
from ppp_perception.core.free_space import cell_centers, clearance_map


def footprint(corners_obj, R_world_obj):
    """(xy radius, min z offset) of the object's box corners rotated into the world frame."""
    c = corners_obj @ R_world_obj.T
    return float(np.linalg.norm(c[:, :2], axis=1).max()), float(c[:, 2].min())


def free_spots(grid, zone, resolution, radius, margin, prefer_xy):
    """Cell centres with enough clearance, sorted by distance to `prefer_xy`."""
    clear = clearance_map(grid, zone, resolution)
    centers = cell_centers(zone, resolution, grid.shape)
    ok = clear >= radius + margin
    pts = centers[ok]
    if len(pts) == 0:
        return pts, clear
    order = np.argsort(np.linalg.norm(pts - np.asarray(prefer_xy)[None, :], axis=1))
    return pts[order], clear


def place_pose(T_world_obj, xy, yaw, table_z, corners_obj, drop_height):
    """Object pose after rotating it by `yaw` about world z and moving it above (x, y)."""
    R = tf.rot_z(yaw) @ T_world_obj[:3, :3]
    _, zmin = footprint(corners_obj, R)
    z = table_z - zmin + drop_height
    return tf.make_T(R, [xy[0], xy[1], z])
