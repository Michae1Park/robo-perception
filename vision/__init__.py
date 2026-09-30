"""Vision pipeline, one module per stage: detect (1, YOLOE), pose (2, FoundationPose), spatial (3, table plane +
occupancy), grasp (4), place (5, shelf supports + rest poses). Shared: transforms, viz."""
import os

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    with open(os.path.join(REPO, 'config.yaml')) as f:
        return yaml.safe_load(f)


def shelf_boxes(shelf, floor_z):
    """The pantry (config.yaml sim.shelf) as boxes: four corner posts from the floor to the top board, one board
    per level; open back and sides. -> [(name, centre xyz, size xyz)] in the base frame (sim/cell.py, MoveIt)."""
    (cx, cy), (w, d), t, p = shelf['center_xy'], shelf['size'], shelf['board'], shelf['post']
    top = max(shelf['levels_z'])
    boxes = [('post_%d' % k, [cx + sx * (w - p) / 2, cy + sy * (d - p) / 2, (floor_z + top) / 2], [p, p, top - floor_z])
             for k, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))]
    return boxes + [('board_%d' % k, [cx, cy, z - t / 2], [w, d, t]) for k, z in enumerate(shelf['levels_z'])]


def read_obj_vertices(path):
    """Vertices (N, 3) of a Wavefront OBJ (no mesh library needed)."""
    with open(path) as f:
        return np.array([[float(v) for v in line.split()[1:4]] for line in f if line.startswith('v ')])
