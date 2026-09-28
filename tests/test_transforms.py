import numpy as np

from ppp_common import transforms as tf


def test_quat_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = tf.quat_to_mat(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.allclose(tf.quat_to_mat(tf.mat_to_quat(R)), R, atol=1e-9)


def test_invert():
    T = tf.make_T(tf.rot_z(0.7), [1, 2, 3])
    assert np.allclose(T @ tf.invert(T), np.eye(4))


def test_look_at_ros_axes():
    R = tf.look_at_ros([1, 0, 1], [0, 0, 0])
    assert np.allclose(R[:, 2], np.array([-1, 0, -1]) / np.sqrt(2))
    assert R[2, 1] < 0          # image y points down in the world
    assert np.isclose(np.linalg.det(R), 1.0)


def test_project_backproject():
    K = tf.intrinsics_matrix(600, 600, 320, 240)
    depth = np.full((480, 640), 2.0)
    pts = tf.depth_to_points(depth, K, stride=40)
    uv, z = tf.project(K, np.eye(4), pts)
    assert np.allclose(z, 2.0)
    assert uv[:, 0].min() >= 0 and uv[:, 0].max() < 640
