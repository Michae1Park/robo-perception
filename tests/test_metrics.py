import numpy as np

from ppp_common import transforms as tf
from ppp_eval.core import metrics as m


def test_iou():
    assert m.bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert abs(m.bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9


def test_pose_errors_and_symmetry():
    pts = np.random.default_rng(0).uniform(-0.03, 0.03, size=(300, 3))
    T_gt = tf.make_T(np.eye(3), [0.5, 0, 0.05])
    T_est = tf.make_T(tf.rot_z(np.deg2rad(40)), [0.502, 0, 0.05])
    assert abs(m.translation_error_mm(T_est, T_gt) - 2.0) < 1e-6
    assert abs(m.rotation_error_deg(T_est[:3, :3], T_gt[:3, :3]) - 40) < 1e-6
    assert m.rotation_error_deg(T_est[:3, :3], T_gt[:3, :3], 'cylinder_z') < 1e-6
    assert m.adds_mm(T_est, T_gt, pts) <= m.add_mm(T_est, T_gt, pts)
    R_flip = tf.quat_to_mat([1, 0, 0, 0])   # 180 deg about x
    assert abs(m.rotation_error_deg(R_flip, np.eye(3)) - 180) < 1e-6
    assert m.rotation_error_deg(R_flip, np.eye(3), 'box180') < 1e-6


def test_summary_markdown():
    trials = [
        {'detection_success': True, 'task_success': True, 'iou': 0.9, 'latency_ms': {'detect': 10.0}},
        {'detection_success': False, 'task_success': False, 'failed_stage': 'detect', 'latency_ms': {'detect': 12.0}},
    ]
    s = m.summarize(trials)
    assert s['rates']['detection_success'] == 0.5
    assert s['failures'] == {'detect': 1}
    assert '| detect |' in m.to_markdown(s)
