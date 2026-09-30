"""Rest poses (vision.place.rest_poses): how an object can stand on a shelf board, upright first."""
import os

import numpy as np
import pytest

from vision import REPO
from vision.grasp import read_obj_vertices
from vision.place import rest_poses


def box(x, y, z):
    return np.array([[sx * x / 2, sy * y / 2, sz * z / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def test_box_all_faces_upright_first():
    p = rest_poses(box(0.06, 0.10, 0.20))
    assert p[0]['name'] == 'upright' and abs(p[0]['height'] - 0.20) < 1e-6
    assert sorted(round(q['height'], 3) for q in p[1:]) == [0.06, 0.06, 0.1, 0.1]   # 4 sides; upside down left out


def test_rotation_puts_the_face_down():
    for q in rest_poses(box(0.06, 0.10, 0.20)):
        W = box(0.06, 0.10, 0.20) @ q['R'].T
        assert np.isclose(np.linalg.det(q['R']), 1.0)
        assert np.sum(np.isclose(W[:, 2], W[:, 2].min())) == 4          # a whole face on the support


def test_tipping_com_is_unstable():
    """Centre of mass outside the base: that face is not a rest pose."""
    v = box(0.02, 0.02, 0.20)
    names = [round(q['height'], 2) for q in rest_poses(v, com=[0.05, 0, 0])]
    assert 0.2 not in names


@pytest.mark.parametrize('name, n_side', [('005_tomato_soup_can', 0), ('006_mustard_bottle', 3)])
def test_ycb(name, n_side):
    path = os.path.join(REPO, 'assets', 'ycb', name, 'textured.obj')
    if not os.path.exists(path):
        pytest.skip('no YCB meshes')
    p = rest_poses(read_obj_vertices(path))
    assert p[0]['name'] == 'upright'
    assert sum(q['name'] == 'on its side' for q in p) == n_side    # a can on its side rolls: left out
