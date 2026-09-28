import numpy as np

from ppp_common import transforms as tf
from ppp_manipulation.core import grasp as g
from ppp_manipulation.core import place as pl
from ppp_perception.core import free_space as fs
from ppp_common.mesh import box_corners


def test_candidates_respect_gripper_width():
    lo, hi = np.array([-0.045, -0.0225, -0.0875]), np.array([0.045, 0.0225, 0.0875])  # sugar box
    cands = g.generate_candidates(lo, hi, max_opening=0.08, width_margin=0.012)
    assert cands and all(c.width + 0.012 <= 0.08 for c in cands)
    for c in cands:
        R = c.T_obj_tcp[:3, :3]
        assert np.isclose(np.linalg.det(R), 1.0)


def test_top_down_grasp_selected_for_upright_box():
    lo, hi = np.array([-0.045, -0.0225, -0.0875]), np.array([0.045, 0.0225, 0.0875])
    T_obj = tf.make_T(tf.rot_z(0.3), [0.5, 0.2, 0.0875])
    cands = g.filter_candidates(g.generate_candidates(lo, hi), T_obj, table_z=0.0)
    best = cands[0]
    assert best.tilt < 1e-6
    assert best.face.startswith('+z/') and best.face.endswith('y')   # close across the 4.5 cm side
    assert best.T_world_tcp[2, 3] > 0.14                                # near the top face
    hand = g.tcp_to_hand(best.T_world_tcp, 0.1034)
    assert np.isclose(hand[2, 3] - best.T_world_tcp[2, 3], 0.1034)
    pre = g.offset_along_approach(hand, 0.1)
    assert np.isclose(pre[2, 3] - hand[2, 3], 0.1)


def test_flat_object_side_grasps_rejected_near_table():
    lo, hi = np.array([-0.045, -0.035, -0.015]), np.array([0.045, 0.035, 0.015])
    T_obj = tf.make_T(np.eye(3), [0.5, 0.0, 0.015])
    cands = g.filter_candidates(g.generate_candidates(lo, hi), T_obj, table_z=0.0, max_tilt_deg=90)
    assert all(c.face.startswith('+z') for c in cands)


def test_place_spot_avoids_obstacle():
    zone = ((0.4, 0.6), (-0.3, -0.1))
    grid = np.full((20, 20), fs.FREE, dtype=np.int8)
    grid[8:12, 8:12] = fs.OCCUPIED                          # obstacle in the middle
    lo, hi = np.array([-0.03, -0.03, -0.05]), np.array([0.03, 0.03, 0.05])
    corners = box_corners(lo, hi)
    r, zmin = pl.footprint(corners, np.eye(3))
    assert np.isclose(r, np.hypot(0.03, 0.03)) and np.isclose(zmin, -0.05)
    spots, clear = pl.free_spots(grid, zone, 0.01, 0.03, 0.01, prefer_xy=(0.5, -0.2))
    assert len(spots) > 0
    x, y = spots[0]
    assert np.hypot(x - 0.5, y - 0.2 * -1) >= 0.02 + 0.04 - 1e-9
    T = pl.place_pose(tf.make_T(np.eye(3), [0.5, 0.2, 0.05]), spots[0], np.pi / 2, 0.0, corners, 0.01)
    assert np.isclose(T[2, 3], 0.06)


def test_every_configured_object_has_a_top_down_grasp():
    """Uses the real meshes when scripts/prepare_ycb.py has been run."""
    import os
    import pytest
    from ppp_common.config import SceneConfig
    from ppp_common.mesh import read_obj_vertices
    cfg = SceneConfig()
    r, gc = cfg.robot, cfg.raw['grasp']
    for name in cfg.object_names:
        if not os.path.isfile(cfg.mesh_path(name)):
            pytest.skip('YCB meshes not prepared')
        v = read_obj_vertices(cfg.mesh_path(name))
        lo, hi = v.min(axis=0), v.max(axis=0)
        T_obj = tf.make_T(tf.rot_z(1.0), [0.5, 0.2, -lo[2]])   # resting upright on z = 0
        cands = g.filter_candidates(
            g.generate_candidates(lo, hi, r['max_opening'], gc['width_margin'], r['finger_depth']),
            T_obj, 0.0, gc['max_approach_tilt_deg'], gc['table_clearance'])
        assert cands and cands[0].tilt < 1e-6, name


def test_executor_time_scaling():
    import importlib.util
    import sys
    import types
    # executor_node imports ROS; stub the modules so the pure helpers can be tested
    for mod in ('rclpy', 'rclpy.action', 'rclpy.callback_groups', 'rclpy.executors', 'rclpy.node',
                'control_msgs', 'control_msgs.action', 'sensor_msgs', 'sensor_msgs.msg'):
        sys.modules.setdefault(mod, types.ModuleType(mod))
    for mod, names in (('rclpy.action', ['ActionServer', 'CancelResponse']),
                       ('rclpy.callback_groups', ['ReentrantCallbackGroup']),
                       ('rclpy.executors', ['MultiThreadedExecutor']), ('rclpy.node', ['Node']),
                       ('control_msgs.action', ['FollowJointTrajectory', 'GripperCommand']),
                       ('sensor_msgs.msg', ['JointState'])):
        for n in names:
            setattr(sys.modules[mod], n, getattr(sys.modules[mod], n, object))
    import os
    path = os.path.join(os.path.dirname(__file__), '..', 'ros2', 'ppp_manipulation', 'ppp_manipulation',
                        'executor_node.py')
    spec = importlib.util.spec_from_file_location('executor_node_under_test', path)
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    pos = [np.zeros(2), np.array([1.0, 0.0]), np.array([1.0, 0.5])]
    times = ex.limit_velocity([0.0, 0.1, 2.0], pos, max_vel=1.0)
    assert np.isclose(times[1], 1.0) and np.isclose(times[2], 20.0)   # slowed down 10x
    assert ex.limit_velocity([0.0, 5.0], pos[:2], 1.0) == [0.0, 5.0]   # already slow enough
    q = ex.interpolate([0.0, 1.0], [np.zeros(2), np.ones(2)], 0.25)
    assert np.allclose(q, 0.25)
    assert np.allclose(ex.retime(np.zeros(2), [np.array([0.5, 0.0])], 1.0), [0.5])
