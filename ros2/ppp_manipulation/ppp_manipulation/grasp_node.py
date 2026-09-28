"""④ Grasp pose: box-face candidates from the mesh -> world via the estimated pose -> reachable best.

Service   /manipulation/plan_grasp     ppp_interfaces/PlanGrasp
Publishes /manipulation/grasp_markers  visualization_msgs/MarkerArray
"""
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from ppp_common import transforms as tf
from ppp_common.config import SceneConfig
from ppp_common.mesh import read_obj_vertices
from ppp_common.ros_conv import T_to_pose, T_to_pose_stamped, pose_to_T
from ppp_common.ros_utils import CallError, Stopwatch
from ppp_interfaces.srv import PlanGrasp

from .core import grasp as g
from .moveit_client import MoveItClient


def arrow(ns, i, T, length, color, frame, stamp):
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = stamp
    m.ns, m.id, m.type = ns, i, Marker.ARROW
    # Marker arrows point along +x; rotate so it points along the approach (+z of the TCP)
    R = T[:3, :3] @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
    start = T[:3, 3] - T[:3, 2] * length
    m.pose = T_to_pose(tf.make_T(R, start))
    m.scale.x, m.scale.y, m.scale.z = length, 0.006, 0.006
    m.color.r, m.color.g, m.color.b, m.color.a = color
    return m


class GraspNode(Node):
    def __init__(self):
        super().__init__('grasp_planner')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('assets_dir', '')
        self.cfg = SceneConfig(self.get_parameter('scene_config').value, self.get_parameter('assets_dir').value)
        self.robot = self.cfg.robot
        self.gcfg = self.cfg.raw['grasp']
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
        self.marker_pub = self.create_publisher(MarkerArray, '/manipulation/grasp_markers', 1)
        self.srv = self.create_service(PlanGrasp, '/manipulation/plan_grasp', self.on_request, callback_group=cb)
        self.get_logger().info('grasp planner ready (%d objects)' % len(self.boxes))

    def on_request(self, req, res):
        sw = Stopwatch()
        if req.target not in self.boxes:
            res.success, res.message = False, 'unknown object %s' % req.target
            return res
        lo, hi = self.boxes[req.target]
        T_obj = pose_to_T(req.object_pose.pose)
        r = self.robot
        cands = g.generate_candidates(lo, hi, r['max_opening'], self.gcfg['width_margin'], r['finger_depth'])
        feasible = g.filter_candidates(cands, T_obj, req.table_height, self.gcfg['max_approach_tilt_deg'],
                                       self.gcfg['table_clearance'])
        res.num_candidates = len(cands)
        best = None
        reachable = []
        for c in feasible:
            T_hand = g.tcp_to_hand(c.T_world_tcp, r['tcp_offset'])
            T_pre = g.offset_along_approach(T_hand, r['approach_distance'])
            try:
                q_pre = self.moveit.solve_ik(T_pre, r['home'], timeout=self.gcfg['ik_timeout'])
                q_grasp = self.moveit.solve_ik(T_hand, q_pre, timeout=self.gcfg['ik_timeout']) if q_pre else None
            except CallError as e:
                res.success, res.message = False, 'IK service: %s' % e
                return res
            if q_pre is None or q_grasp is None or g.joint_distance(q_pre, q_grasp) > 1.0:
                continue
            score = c.tilt + 0.1 * g.joint_distance(q_pre, r['home'])
            reachable.append(c)
            if best is None or score < best[0]:
                best = (score, c, T_hand, T_pre, q_pre)
        res.num_feasible = len(reachable)
        res.latency_ms = sw.ms()
        stamp = req.object_pose.header.stamp
        self.publish_markers(feasible, reachable, best[1] if best else None, stamp)
        if best is None:
            res.success = False
            res.message = 'no reachable grasp (%d candidates, %d pass geometry)' % (len(cands), len(feasible))
            return res
        _, c, T_hand, T_pre, q_pre = best
        frame = self.cfg.world_frame
        res.success, res.message = True, 'ok (%s)' % c.face
        res.grasp = T_to_pose_stamped(T_hand, frame, stamp)
        res.pregrasp = T_to_pose_stamped(T_pre, frame, stamp)
        res.pregrasp_joints = JointState(name=r['arm_joints'], position=[float(v) for v in q_pre])
        res.object_to_hand = T_to_pose(tf.invert(T_obj) @ T_hand)
        res.width = c.width
        self.get_logger().info('grasp %s: %s, %d/%d/%d cand/geom/IK, %.1f ms' % (
            req.target, c.face, len(cands), len(feasible), len(reachable), res.latency_ms))
        return res

    def publish_markers(self, feasible, reachable, chosen, stamp):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        frame = self.cfg.world_frame
        for i, c in enumerate(feasible):
            if c is chosen:
                color = (0.0, 1.0, 0.0, 1.0)
            elif any(c is r for r in reachable):
                color = (1.0, 0.8, 0.0, 0.8)
            else:
                color = (0.6, 0.6, 0.6, 0.5)
            arr.markers.append(arrow('grasp_candidates', i, c.T_world_tcp, 0.08, color, frame, stamp))
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = GraspNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
