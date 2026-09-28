"""⑤ Placement: free spot in the place zone where the object's footprint fits, reachable with the grasp.

Service   /manipulation/plan_place     ppp_interfaces/PlanPlace
Publishes /manipulation/place_markers  visualization_msgs/MarkerArray
"""
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from ppp_common import transforms as tf
from ppp_common.config import SceneConfig, zone_center
from ppp_common.mesh import box_corners, read_obj_vertices
from ppp_common.ros_conv import T_to_pose, T_to_pose_stamped, pose_to_T
from ppp_common.ros_utils import CallError, Stopwatch
from ppp_interfaces.srv import PlanPlace

from .core import place as pl
from .moveit_client import MoveItClient


def grid_from_msg(msg):
    info = msg.info
    grid = np.array(msg.data, dtype=np.int8).reshape(info.height, info.width)
    x0, y0 = info.origin.position.x, info.origin.position.y
    zone = ((x0, x0 + info.width * info.resolution), (y0, y0 + info.height * info.resolution))
    return grid, zone, info.resolution


class PlaceNode(Node):
    def __init__(self):
        super().__init__('place_planner')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('assets_dir', '')
        self.declare_parameter('max_spots_per_yaw', 20)
        self.cfg = SceneConfig(self.get_parameter('scene_config').value, self.get_parameter('assets_dir').value)
        self.max_spots = int(self.get_parameter('max_spots_per_yaw').value)
        self.robot = self.cfg.robot
        self.pcfg = self.cfg.raw['place']
        self.boxes = {}
        for name in self.cfg.object_names:
            try:
                v = read_obj_vertices(self.cfg.mesh_path(name))
                self.boxes[name] = (v.min(axis=0), v.max(axis=0))
            except OSError:
                self.get_logger().warn('mesh missing for %s' % name)
        cb = ReentrantCallbackGroup()
        self.moveit = MoveItClient(self, self.robot['arm_joints'], hand_link=self.robot['hand_link'],
                                   frame=self.cfg.world_frame, callback_group=cb)
        self.marker_pub = self.create_publisher(MarkerArray, '/manipulation/place_markers', 1)
        self.srv = self.create_service(PlanPlace, '/manipulation/plan_place', self.on_request, callback_group=cb)
        self.get_logger().info('place planner ready')

    def on_request(self, req, res):
        sw = Stopwatch()
        if req.target not in self.boxes:
            res.success, res.message = False, 'unknown object %s' % req.target
            return res
        lo, hi = self.boxes[req.target]
        corners = box_corners(lo, hi)
        T_obj = pose_to_T(req.object_pose.pose)
        T_obj_hand = pose_to_T(req.object_to_hand)
        grid, zone, resolution = grid_from_msg(req.place_zone)
        prefer = zone_center(zone)
        r = self.robot
        tried = 0
        found = None
        try:
            for yaw_deg in self.pcfg['yaw_candidates_deg']:
                yaw = np.deg2rad(yaw_deg)
                radius, _ = pl.footprint(corners, tf.rot_z(yaw) @ T_obj[:3, :3])
                spots, _ = pl.free_spots(grid, zone, resolution, radius, self.pcfg['footprint_margin'], prefer)
                for xy in spots[:self.max_spots]:
                    tried += 1
                    T_place = pl.place_pose(T_obj, xy, yaw, req.table_height, corners, self.pcfg['drop_height'])
                    T_hand = T_place @ T_obj_hand
                    T_pre = T_hand.copy()
                    T_pre[2, 3] += r['approach_distance']
                    q_pre = self.moveit.solve_ik(T_pre, r['home'])
                    if q_pre is None or self.moveit.solve_ik(T_hand, q_pre) is None:
                        continue
                    found = (T_place, T_hand, T_pre, q_pre, yaw_deg)
                    break
                if found:
                    break
        except CallError as e:
            res.success, res.message = False, 'IK service: %s' % e
            return res
        res.latency_ms = sw.ms()
        if found is None:
            res.success = False
            res.message = 'no free & reachable spot (%d tried, %d free cells)' % (tried, int((grid == 0).sum()))
            return res
        T_place, T_hand, T_pre, q_pre, yaw_deg = found
        frame, stamp = self.cfg.world_frame, req.object_pose.header.stamp
        res.success, res.message = True, 'ok (yaw %+d deg)' % yaw_deg
        res.object_place = T_to_pose_stamped(T_place, frame, stamp)
        res.place = T_to_pose_stamped(T_hand, frame, stamp)
        res.preplace = T_to_pose_stamped(T_pre, frame, stamp)
        res.preplace_joints = JointState(name=r['arm_joints'], position=[float(v) for v in q_pre])
        self.publish_marker(T_place, lo, hi, frame, stamp)
        p = T_place[:3, 3]
        self.get_logger().info('place %s at (%.3f, %.3f) yaw %+d, %d spots tried, %.1f ms'
                               % (req.target, p[0], p[1], yaw_deg, tried, res.latency_ms))
        return res

    def publish_marker(self, T_place, lo, hi, frame, stamp):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = stamp
        m.ns, m.id, m.type = 'place_target', 0, Marker.CUBE
        center = tf.transform_points(T_place, [(lo + hi) / 2.0])[0]
        m.pose = T_to_pose(tf.make_T(T_place[:3, :3], center))
        m.scale.x, m.scale.y, m.scale.z = [float(v) for v in (hi - lo)]
        m.color.b, m.color.g, m.color.a = 1.0, 0.4, 0.4
        self.marker_pub.publish(MarkerArray(markers=[m]))


def main():
    rclpy.init()
    node = PlaceNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
