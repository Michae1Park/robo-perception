import numpy as np

from vision import spatial, transforms as tf


def test_invert():
    T = tf.make_T(tf.rot_z(0.7), [1, 2, 3])
    assert np.allclose(T @ tf.invert(T), np.eye(4))


def test_project_backproject():
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    pts = spatial.depth_to_points(np.full((480, 640), 2.0), K, stride=40)
    uv, z = tf.project(K, np.eye(4), pts)
    assert np.allclose(z, 2.0)
    assert uv[:, 0].min() >= 0 and uv[:, 0].max() < 640
