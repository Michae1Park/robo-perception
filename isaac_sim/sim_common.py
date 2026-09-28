"""Scene construction shared by scene.py (ROS 2 sim) and capture_prompts.py.

Import only after SimulationApp has been created. Targets Isaac Sim 4.5 / 5.x (`isaacsim.*` API).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ros2', 'ppp_common'))
sys.path.insert(0, HERE)

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.materials import PhysicsMaterial  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, GroundPlane  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

from ppp_common import transforms as tf  # noqa: E402
from ppp_common.mesh import read_obj_vertices  # noqa: E402
from ycb_usd import ensure_object_usd  # noqa: E402

PARK_X = -3.0  # objects not used in an episode wait on the floor behind the robot, out of view


def safe_prim_name(name):
    """USD prim names can't start with a digit (true of every YCB object name, e.g. "004_sugar_box")."""
    return name if (name[:1].isalpha() or name[:1] == '_') else '_' + name


def quat_wxyz(R):
    return tf.xyzw_to_wxyz(tf.mat_to_quat(R))


def bind_physics_material(prim, material_path):
    mat = UsdShade.Material(get_current_stage().GetPrimAtPath(material_path))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants, 'physics')


class SimScene:
    def __init__(self, cfg, with_robot=True):
        self.cfg = cfg
        raw = cfg.raw
        dt = float(raw['physics']['dt'])
        self.world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
        stage = get_current_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        GroundPlane('/World/Ground', z_position=-0.8, size=20.0, color=np.array([0.35, 0.35, 0.38]))
        f = float(raw['physics']['friction'])
        self.material = PhysicsMaterial('/World/Physics/grip', static_friction=f, dynamic_friction=f,
                                        restitution=0.0)
        t = raw['table']
        self.table = FixedCuboid('/World/Table', name='table', position=np.array(t['center'], dtype=float),
                                 scale=np.array(t['size'], dtype=float), size=1.0,
                                 color=np.array([0.62, 0.52, 0.40]))
        self.table.apply_physics_material(self.material)
        dome = UsdLux.DomeLight.Define(stage, '/World/Lights/dome')
        dome.CreateIntensityAttr(900.0)
        sun = UsdLux.DistantLight.Define(stage, '/World/Lights/sun')
        sun.CreateIntensityAttr(2500.0)
        sun.CreateAngleAttr(1.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 20.0, 0.0))

        self.robot = None
        if with_robot:
            from isaacsim.robot.manipulators.examples.franka import Franka
            self.robot = self.world.scene.add(Franka(prim_path='/World/Franka', name='franka'))
            for finger in ('panda_leftfinger', 'panda_rightfinger'):
                prim = stage.GetPrimAtPath('/World/Franka/' + finger)
                if prim.IsValid():
                    bind_physics_material(prim, '/World/Physics/grip')

        self.objects = {}
        self.half_height = {}
        for i, name in enumerate(cfg.object_names):
            self.objects[name] = self._add_object(name, i)
        self.clutter = []
        n_clutter = int(raw['spawn']['num_clutter'][1])
        rng = np.random.default_rng(0)
        for i in range(n_clutter):
            c = DynamicCuboid('/World/Clutter/box_%d' % i, name='clutter_%d' % i, size=1.0,
                              position=np.array([PARK_X, 1.0 + 0.3 * i, -0.75]),
                              scale=np.array(raw['spawn']['clutter_size_min'], dtype=float),
                              color=rng.uniform(0.2, 0.9, 3), mass=0.2)
            c.apply_physics_material(self.material)
            self.clutter.append(self.world.scene.add(c))
        self.clutter_names = ['clutter_%d' % i for i in range(n_clutter)]

    def _add_object(self, name, index):
        usd = ensure_object_usd(self.cfg, name)
        path = '/World/Objects/' + safe_prim_name(name)
        add_reference_to_stage(usd, path)
        stage = get_current_stage()
        root = stage.GetPrimAtPath(path)
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = UsdPhysics.MassAPI.Apply(root)
        mass.CreateMassAttr(float(self.cfg.objects[name]['mass']))
        geom = stage.GetPrimAtPath(path + '/geom')
        UsdPhysics.CollisionAPI.Apply(geom)
        UsdPhysics.MeshCollisionAPI.Apply(geom).CreateApproximationAttr('convexHull')
        bind_physics_material(geom, '/World/Physics/grip')
        v = read_obj_vertices(self.cfg.mesh_path(name))
        self.half_height[name] = float(-v[:, 2].min())
        prim = SingleRigidPrim(path, name=name,
                               position=np.array([PARK_X, -1.0 - 0.3 * index, -0.8 + self.half_height[name]]))
        return self.world.scene.add(prim)

    # ---- camera ----------------------------------------------------------------------------
    @staticmethod
    def add_camera(path, eye, target, width, height):
        """Camera whose ROS optical frame looks from `eye` at `target`. Call initialize() after world.reset()."""
        cam = Camera(prim_path=path, resolution=(width, height))
        cam.set_world_pose(position=np.array(eye, dtype=float), orientation=quat_wxyz(tf.look_at_ros(eye, target)),
                           camera_axes='ros')
        return cam

    @staticmethod
    def configure_intrinsics(cam, width, height, fx, fy, cx, cy):
        """Pinhole intrinsics via the USD camera attributes (square pixels, centred principal point)."""
        prim = UsdGeom.Camera(get_current_stage().GetPrimAtPath(cam.prim_path))
        ha = 20.955
        prim.GetHorizontalApertureAttr().Set(ha)
        prim.GetVerticalApertureAttr().Set(ha * height / width)
        prim.GetFocalLengthAttr().Set(fx * ha / width)
        prim.GetHorizontalApertureOffsetAttr().Set(0.0)
        prim.GetVerticalApertureOffsetAttr().Set(0.0)
        prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 10.0))
        if abs(fx - fy) > 1e-3 or abs(cx - width / 2.0) > 0.5 or abs(cy - height / 2.0) > 0.5:
            print('[ppp] WARNING: sim camera supports square pixels + centred principal point only')

    # ---- episode layout ----------------------------------------------------------------------
    @staticmethod
    def sample_positions(rng, zone, n, min_dist, margin=0.04, tries=200, restarts=100):
        """Up to n points in the zone, pairwise >= min_dist apart (greedy with random restarts)."""
        (x0, x1), (y0, y1) = zone
        best = []
        for _ in range(restarts):
            pts = []
            for _ in range(tries):
                p = rng.uniform([x0 + margin, y0 + margin], [x1 - margin, y1 - margin])
                if all(np.linalg.norm(p - q) >= min_dist for q in pts):
                    pts.append(p)
                    if len(pts) == n:
                        return pts
            best = pts if len(pts) > len(best) else best
        return best

    def park_all(self):
        for i, (name, prim) in enumerate(self.objects.items()):
            self.set_pose(prim, [PARK_X, -1.0 - 0.3 * i, -0.8 + self.half_height[name] + 0.005], 0.0)
        for i, c in enumerate(self.clutter):
            self.set_pose(c, [PARK_X, 1.0 + 0.3 * i, -0.75], 0.0)

    @staticmethod
    def set_pose(prim, pos, yaw):
        prim.set_world_pose(position=np.array(pos, dtype=float), orientation=quat_wxyz(tf.rot_z(yaw)))
        prim.set_linear_velocity(np.zeros(3))
        prim.set_angular_velocity(np.zeros(3))

    def layout(self, seed):
        """Random episode: objects in the pick zone, clutter boxes in the place zone."""
        sp = self.cfg.raw['spawn']
        rng = np.random.default_rng(seed)
        self.park_all()
        k = int(rng.integers(sp['num_objects'][0], sp['num_objects'][1] + 1))
        names = [str(n) for n in rng.choice(self.cfg.object_names, size=k, replace=False)]
        spots = self.sample_positions(rng, self.cfg.zone('pick'), k, sp['min_distance'])
        names = names[:len(spots)]
        for name, p in zip(names, spots):
            z = self.cfg.table_height + self.half_height[name] + sp['drop_height']
            self.set_pose(self.objects[name], [p[0], p[1], z], rng.uniform(-np.pi, np.pi))
        m = int(rng.integers(sp['num_clutter'][0], sp['num_clutter'][1] + 1))
        spots = self.sample_positions(rng, self.cfg.zone('place'), m, sp['clutter_min_distance'], margin=0.04)
        clutter = []
        for c, p in zip(self.clutter, spots):
            size = rng.uniform(sp['clutter_size_min'], sp['clutter_size_max'])
            c.set_local_scale(size)
            z = self.cfg.table_height + size[2] / 2.0 + sp['drop_height']
            self.set_pose(c, [p[0], p[1], z], rng.uniform(-np.pi, np.pi))
            clutter.append(c.name)
        return names, clutter

    def object_pose(self, name):
        """World pose (4x4) of an object or clutter box."""
        prim = self.objects.get(name) or next(c for c in self.clutter if c.name == name)
        p, q = prim.get_world_pose()
        return tf.make_T(tf.quat_to_mat(tf.wxyz_to_xyzw(q)), p)
