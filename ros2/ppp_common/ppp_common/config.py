"""Scene configuration shared by Isaac Sim, the ROS 2 nodes and the evaluation tools.

`scene.yaml` is the single source of truth for table/zones, camera, robot and objects.
"""
import os

import numpy as np
import yaml

from . import transforms as tf


def default_scene_path():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get('PPP_SCENE_CONFIG', ''),
        # source tree: ros2/ppp_common/ppp_common -> ros2/ppp_bringup/config
        os.path.join(here, '..', '..', 'ppp_bringup', 'config', 'scene.yaml'),
    ]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(get_package_share_directory('ppp_bringup'), 'config', 'scene.yaml'))
    except Exception:
        pass
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError('scene.yaml not found; set PPP_SCENE_CONFIG')


def assets_root(explicit=''):
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    if os.environ.get('PPP_ASSETS'):
        return os.path.abspath(os.path.expanduser(os.environ['PPP_ASSETS']))
    if os.environ.get('PPP_ROOT'):
        return os.path.join(os.path.abspath(os.path.expanduser(os.environ['PPP_ROOT'])), 'assets')
    here = os.path.dirname(os.path.abspath(__file__))
    repo_assets = os.path.abspath(os.path.join(here, '..', '..', '..', 'assets'))
    if os.path.isdir(repo_assets):
        return repo_assets
    return os.path.expanduser('~/prompt-pick-place/assets')


class SceneConfig:
    def __init__(self, path='', assets=''):
        self.path = path or default_scene_path()
        with open(self.path) as f:
            self.raw = yaml.safe_load(f)
        self.assets = assets_root(assets)
        self.objects = {o['name']: o for o in self.raw['objects']}

    # ---- geometry -------------------------------------------------------
    @property
    def world_frame(self):
        return self.raw.get('world_frame', 'world')

    @property
    def table_height(self):
        t = self.raw['table']
        return float(t['center'][2] + t['size'][2] / 2.0)

    def zone(self, name):
        z = self.raw['zones'][name]
        return (tuple(z['x']), tuple(z['y']))

    # ---- camera ---------------------------------------------------------
    @property
    def camera(self):
        return self.raw['camera']

    def camera_K(self):
        c = self.camera
        return tf.intrinsics_matrix(c['fx'], c['fy'], c['cx'], c['cy'])

    def T_world_camera(self):
        """Pose of the ROS optical camera frame in the world frame."""
        c = self.camera
        return tf.make_T(tf.look_at_ros(c['eye'], c['target']), c['eye'])

    # ---- robot ----------------------------------------------------------
    @property
    def robot(self):
        return self.raw['robot']

    # ---- assets ---------------------------------------------------------
    @property
    def object_names(self):
        return list(self.objects.keys())

    def mesh_path(self, name):
        return os.path.join(self.assets, 'ycb', name, 'textured.obj')

    def texture_path(self, name):
        return os.path.join(self.assets, 'ycb', name, 'texture_map.png')

    def usd_path(self, name):
        return os.path.join(self.assets, 'ycb', name, name + '.usd')

    def prompt_path(self, name):
        return os.path.join(self.assets, 'prompts', name + '.png')

    def symmetry(self, name):
        return self.objects[name].get('symmetry', 'none')


def zone_contains(zone, xy, margin=0.0):
    (x0, x1), (y0, y1) = zone
    return bool(x0 + margin <= xy[0] <= x1 - margin and y0 + margin <= xy[1] <= y1 - margin)


def zone_center(zone):
    (x0, x1), (y0, y1) = zone
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
