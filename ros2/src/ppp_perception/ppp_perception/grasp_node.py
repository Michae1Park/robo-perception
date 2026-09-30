"""Stage 4 node: grasp candidates for a posed object (vision/grasp.py), filtered against the table and the scene.

Service ~/plan_grasps (ppp_interfaces/PlanGrasps): object pose (from EstimatePose) ->
  1. spatial_node ~/analyze_table (C++, stage 3): table plane + the points standing on the table
  2. obstacles = those points over the table's footprint, minus the object's own (its mesh box + 1 cm); also returned,
     for the task to put into MoveIt's planning scene
  3. analytic parallel-jaw candidates on the mesh box -> keep approach tilt <= max, fingertips above the table,
     gripper not running into obstacles; least tilted first
Also publishes ~/grasps (PoseArray of the kept TCP poses) for viewing.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs_py import point_cloud2

from ppp_interfaces.msg import Grasp
from ppp_interfaces.srv import AnalyzeTable, PlanGrasps

from .common import BASE, T_to_pose, T_to_pose_stamped, call, load_config, mesh_path, pose_to_T, tf, ycb_name

FINGER_THICK = 0.01   # Franka finger pad thickness, as in pipeline/interactive_grasp.py


def rot(axis, a):
    """Rotation by angle a about the object's x, y or z axis."""
    c, s = np.cos(a), np.sin(a)
    i = 'xyz'.index(axis)
    R = np.eye(3)
    j, k = [(1, 2), (2, 0), (0, 1)][i]
    R[j, j], R[j, k], R[k, j], R[k, k] = c, -s, s, c
    return R


class GraspNode(Node):
    def __init__(self):
        super().__init__('grasp_node')
        cfg = load_config()
        self.gripper, self.gcfg, self.scfg = cfg['gripper'], cfg['grasp'], cfg['spatial']
        # the task needs side grasps to reach into the shelf, so the ROS pipeline allows more tilt than the
        # stage 4 playground (config.yaml grasp.max_approach_tilt_deg)
        self.max_grasps = self.declare_parameter('max_grasps', 12).value   # best (least tilted) ones returned
        self.max_tilt = self.declare_parameter('max_approach_tilt_deg', float(self.gcfg['max_approach_tilt_deg'])).value
        self.cb = ReentrantCallbackGroup()
        self.table = self.create_client(AnalyzeTable, '/spatial_node/analyze_table', callback_group=self.cb)
        self.pub = self.create_publisher(PoseArray, '~/grasps', 5)
        self.create_service(PlanGrasps, '~/plan_grasps', self.on_request, callback_group=self.cb)
        self.boxes = {}
        self.objects = cfg.get('objects', {})
        self.table_xy = (cfg['sim']['table_center'], cfg['sim']['table_size'][:2])
        self.get_logger().info('ready')

    def box(self, target):
        """Object's mesh bounding box (object frame) - the grasp model."""
        if target not in self.boxes:
            from vision.grasp import read_obj_vertices
            v = read_obj_vertices(mesh_path(target))
            self.boxes[target] = (v.min(axis=0), v.max(axis=0))
        return self.boxes[target]

    def on_request(self, req, res):
        from vision import grasp as g
        if req.object_pose.header.frame_id != BASE:
            res.message = 'object_pose must be in %s' % BASE
            return res
        try:
            table = call(self, self.table, AnalyzeTable.Request())
        except Exception as e:  # noqa: BLE001
            res.message = 'analyze_table failed: %s' % e
            return res
        if not table.success:
            res.message = 'no table: ' + table.message
            return res
        T_wo = pose_to_T(req.object_pose)
        lo, hi = self.box(req.target)
        pts = point_cloud2.read_points_numpy(table.above_table, field_names=('x', 'y', 'z')).astype(float)
        in_obj = tf.transform_points(tf.invert(T_wo), pts)
        own = np.all((in_obj > lo - 0.01) & (in_obj < hi + 0.01), axis=1)
        # only what stands on the table (its footprint is known cell geometry, like MoveIt's collision boxes):
        # leaves out the robot's base (the pedestal top is at table height) and the shelf
        (tx, ty), (tw, tl) = self.table_xy
        on_table = (np.abs(pts[:, 0] - tx) < tw / 2 - 0.01) & (np.abs(pts[:, 1] - ty) < tl / 2 - 0.01)
        obstacles = pts[~own & on_table]
        a, b, c, d = table.plane
        table_z = float(-(a * T_wo[0, 3] + b * T_wo[1, 3] + d) / c)   # table height under the object

        # a rotationally symmetric object's estimated roll about its axis is arbitrary, so the box model is also
        # tried turned about that axis (every 15 deg); each grasp stays expressed in the estimated object frame
        axis = self.objects.get(ycb_name(req.target), {}).get('symmetry_axis')
        turns = [tf.make_T()] if axis is None else [tf.make_T(rot(axis, np.radians(a))) for a in range(0, 360, 15)]
        cands, keep = [], []
        for S in turns:
            cs = g.generate_candidates(lo, hi, self.gripper['max_opening'], self.gcfg['width_margin'],
                                       self.gripper['finger_depth'])
            for c in cs:
                c.T_obj_tcp = S @ c.T_obj_tcp
            cands += cs
            keep += g.filter_candidates(cs, T_wo, table_z, self.max_tilt, self.gcfg['table_clearance'], FINGER_THICK,
                                        obstacles, self.gripper['max_opening'])
        keep.sort(key=lambda k: k.tilt)
        n_ok = len(keep)
        keep = keep[:self.max_grasps]
        res.obstacles = point_cloud2.create_cloud_xyz32(table.above_table.header, obstacles.astype(np.float32))
        stamp = req.object_pose.header.stamp
        for k in keep:
            res.grasps.append(Grasp(tcp=T_to_pose_stamped(k.T_world_tcp, BASE, stamp), tcp_in_object=T_to_pose(k.T_obj_tcp),
                                    width=float(k.width), tilt=float(k.tilt), face=k.face))
        res.candidates = len(cands)
        res.success = bool(keep)
        res.message = '%s: %d/%d candidates feasible%s, best %d returned (table z %.3f, %d own + %d obstacle points)%s' % (
            req.target, n_ok, len(cands), ' (%d turns about %s)' % (len(turns), axis) if axis else '', len(keep), table_z, int(own.sum()), len(obstacles),
            ', best tilt %.0f deg' % np.degrees(keep[0].tilt) if keep else '')
        self.get_logger().info(res.message)
        arr = PoseArray()
        arr.header.frame_id, arr.header.stamp = BASE, stamp
        arr.poses = [x.tcp.pose for x in res.grasps]
        self.pub.publish(arr)
        return res


def main():
    rclpy.init()
    node = GraspNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
