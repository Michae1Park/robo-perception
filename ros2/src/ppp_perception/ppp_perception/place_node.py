"""Stage 5 node: where on the shelf does the object fit, and how? Wrist-camera depth -> horizontal supports -> free
space and headroom per support -> for each stable rest pose of the object's mesh (upright first, D-031), the
placement points with room for it, most clearance first. No shelf model (D-009, D-020).

Service ~/get_placements (ppp_interfaces/GetPlacements): the task moves the arm to its "look at the shelf" pose,
then asks with not_before = the time the arm stopped; the node waits for a wrist frame at least that new. The
camera pose comes from TF at the frame's stamp (forward kinematics + wrist hand-eye calibration).
Supports are found by the C++ sequential RANSAC (ppp_geometry, D-019); the rest is vision/place.py.
"""
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener

from ppp_interfaces.msg import Placement
from ppp_interfaces.srv import GetPlacements

from .common import BASE, camera_K, depth_m, load_config, lookup_T, mesh_vertices, stamp_sec, tf


class PlaceNode(Node):
    def __init__(self):
        super().__init__('place_node')
        self.cfg = load_config()['place']
        self.wait = self.declare_parameter('frame_timeout', 5.0).value
        self.cb = ReentrantCallbackGroup()
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self, spin_thread=True)
        self.cond = threading.Condition()
        self.depth, self.info = None, None
        q = QoSProfile(depth=5)
        self.create_subscription(Image, 'depth/image_raw', self.on_depth, q, callback_group=self.cb)
        self.create_subscription(CameraInfo, 'depth/camera_info', self.on_info, q, callback_group=self.cb)
        self.pub = self.create_publisher(PoseArray, '~/placements', 5)
        self.create_service(GetPlacements, '~/get_placements', self.on_request, callback_group=self.cb)
        self.get_logger().info('ready')

    def on_depth(self, m):
        with self.cond:
            self.depth = m
            self.cond.notify_all()

    def on_info(self, m):
        self.info = m

    def on_request(self, req, res):
        from vision import place
        want = stamp_sec(req.not_before)
        with self.cond:
            if not self.cond.wait_for(lambda: self.depth is not None and stamp_sec(self.depth.header.stamp) >= want,
                                      self.wait):
                res.message = 'no wrist-camera frame newer than %.2f s within %.0f s' % (want, self.wait)
                return res
            depth_msg = self.depth
        if self.info is None:
            res.message = 'no camera_info yet'
            return res
        try:
            T_bc = lookup_T(self.tf, BASE, depth_msg.header.frame_id, depth_msg.header.stamp)
        except Exception as e:  # noqa: BLE001
            res.message = 'no camera pose: %s' % e
            return res

        V0 = mesh_vertices(req.target)
        poses = [(p['name'], p['R']) for p in place.rest_poses(V0)]   # how it can stand, upright first

        cfg = self.cfg
        P, n, valid, ok = place.points_and_normals(depth_m(depth_msg), camera_K(self.info), T_bc)
        planes = place.find_planes(P, n, ok, cfg, cpp=True)
        grid = place.Grid(P[valid][:, :2], cfg['resolution'])
        supports = place.find_supports(P, planes, grid, cfg)
        for s in supports:
            place.analyze_support(s, supports, P, valid, grid, cfg)
        found = []
        for rank, (name, R) in enumerate(poses):
            V = V0 @ R.T
            radius = float(np.linalg.norm(V[:, :2], axis=1).max() + cfg['margin'])   # footprint, any yaw
            need = float(np.ptp(V[:, 2])) + cfg['hand_clearance']
            cands = [c for s in supports for c in place.candidates(s, grid, radius, need, cfg)]
            cands.sort(key=lambda c: -c['clearance'])
            q = tf.R_to_quat(R)
            for c in cands:
                p = Placement(clearance=c['clearance'], headroom=c['headroom'], reach=c['reach'],
                              object_bottom=float(-V[:, 2].min()), rest_pose=name, rank=rank)
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)
                p.point.header.frame_id, p.point.header.stamp = BASE, depth_msg.header.stamp
                p.point.point.x, p.point.point.y, p.point.point.z = float(c['xy'][0]), float(c['xy'][1]), float(c['z'])
                res.placements.append(p)
            found.append('%s %d (r %.0f mm, needs %.0f mm)' % (name, len(cands), radius * 1000, need * 1000))
        # everything the camera sees that is not a support (items on the shelf, its panels), for MoveIt: without it
        # the arm sweeps the held object through the shelf items on its way in
        on_support = np.zeros(P.shape[:2], bool)
        for _, m in planes:
            on_support |= m
        obst = P[valid & ~on_support]
        near_board = np.zeros(len(obst), bool)   # the boards' own edges (the shelf is in MoveIt already): as voxels
        for z, _ in planes:                      # they would stick up into the space above a placement
            near_board |= np.abs(obst[:, 2] - z) < 0.03
        obst = obst[~near_board & (np.linalg.norm(obst[:, :2], axis=1) < 1.2) & (obst[:, 2] > -0.05)][::3]
        res.obstacles = point_cloud2.create_cloud_xyz32(depth_msg.header, obst.astype(np.float32))
        res.obstacles.header.frame_id = BASE
        res.supports = len(supports)
        res.success = bool(res.placements)
        res.message = '%s: %d supports at z %s -> placements: %s' % (
            req.target, len(supports), '/'.join('%.2f' % s['z'] for s in supports), ', '.join(found))
        self.get_logger().info(res.message)
        arr = PoseArray()
        arr.header.frame_id, arr.header.stamp = BASE, depth_msg.header.stamp
        for p in res.placements:
            ps = Pose()
            ps.position = p.point.point
            ps.orientation.w = 1.0
            arr.poses.append(ps)
        self.pub.publish(arr)
        return res


def main():
    rclpy.init()
    node = PlaceNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
