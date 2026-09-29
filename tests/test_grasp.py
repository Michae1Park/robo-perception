import os

import numpy as np
import pytest

from vision import grasp as g, load_config, REPO, transforms as tf


def test_candidates_respect_gripper_width():
    lo, hi = np.array([-0.045, -0.0225, -0.0875]), np.array([0.045, 0.0225, 0.0875])  # sugar box
    cands = g.generate_candidates(lo, hi, max_opening=0.08, width_margin=0.012)
    assert cands and all(c.width + 0.012 <= 0.08 for c in cands)
    for c in cands:
        assert np.isclose(np.linalg.det(c.T_obj_tcp[:3, :3]), 1.0)


def test_top_down_grasp_selected_for_upright_box():
    lo, hi = np.array([-0.045, -0.0225, -0.0875]), np.array([0.045, 0.0225, 0.0875])
    T_obj = tf.make_T(tf.rot_z(0.3), [0.5, 0.2, 0.0875])
    best = g.filter_candidates(g.generate_candidates(lo, hi), T_obj, table_z=0.0)[0]
    assert best.tilt < 1e-6
    assert best.face.startswith('+z/') and best.face.endswith('y')   # close across the 4.5 cm side
    assert best.T_world_tcp[2, 3] > 0.14                                # near the top face


def test_flat_object_side_grasps_rejected_near_table():
    lo, hi = np.array([-0.045, -0.035, -0.015]), np.array([0.045, 0.035, 0.015])
    T_obj = tf.make_T(np.eye(3), [0.5, 0.0, 0.015])
    cands = g.filter_candidates(g.generate_candidates(lo, hi), T_obj, table_z=0.0, max_tilt_deg=90)
    assert all(c.face.startswith('+z') for c in cands)


def test_every_ycb_object_has_a_top_down_grasp():
    """Uses the real meshes when scripts/prepare_ycb.py has been run."""
    cfg = load_config()
    gr, gc = cfg['gripper'], cfg['grasp']
    root = os.path.join(REPO, 'assets', 'ycb')
    if not os.path.isdir(root):
        pytest.skip('YCB meshes not prepared')
    for name in sorted(os.listdir(root)):
        v = g.read_obj_vertices(os.path.join(root, name, 'textured.obj'))
        lo, hi = v.min(axis=0), v.max(axis=0)
        T_obj = tf.make_T(tf.rot_z(1.0), [0.5, 0.2, -lo[2]])   # resting upright on z = 0
        cands = g.filter_candidates(
            g.generate_candidates(lo, hi, gr['max_opening'], gc['width_margin'], gr['finger_depth']),
            T_obj, 0.0, gc['max_approach_tilt_deg'], gc['table_clearance'])
        assert cands and cands[0].tilt < 1e-6, name
