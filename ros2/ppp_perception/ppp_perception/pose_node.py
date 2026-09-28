"""② 6-DoF pose: bridge to Isaac ROS FoundationPose (one FoundationPose instance per object mesh).

Service   /perception/estimate_pose        ppp_interfaces/EstimatePose
Forwards  /fp/<object>/pose_estimation/{image, depth_image, camera_info, segmentation}
Receives  /fp/<object>/pose_estimation/output   vision_msgs/Detection3DArray (camera frame)
Publishes /perception/target_pose (PoseStamped, world), /perception/pose_overlay (Image)
"""
import threading

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray

from ppp_common import transforms as tf
from ppp_common.config import SceneConfig
from ppp_common.mesh import box_corners, read_obj_vertices
from ppp_common.ros_conv import (T_to_pose_stamped, image_to_numpy, numpy_to_image, pose_to_T,
                                 stamp_to_sec, transform_to_T)
from ppp_common.ros_utils import Stopwatch
from ppp_interfaces.srv import EstimatePose

_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_pose(rgb, K, T_cam_obj, corners, axis_len=0.05):
    img = rgb.copy()
    uv, z = tf.project(K, np.eye(4), tf.transform_points(T_cam_obj, corners))
    if (z <= 0).any():
        return img
    pts = uv.astype(int)
    for a, b in _EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 255, 255), 1, cv2.LINE_AA)
    axes = np.array([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
    uv, _ = tf.project(K, np.eye(4), tf.transform_points(T_cam_obj, axes))
    o = tuple(uv[0].astype(int))
    for i, color in ((1, (255, 0, 0)), (2, (0, 255, 0)), (3, (0, 0, 255))):
        cv2.line(img, o, tuple(uv[i].astype(int)), color, 2, cv2.LINE_AA)
    return img


class PoseNode(Node):
    def __init__(self):
        super().__init__('pose_estimator')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('assets_dir', '')
        self.declare_parameter('fp_namespace', '/fp')
        self.declare_parameter('timeout', 5.0)
        self.declare_parameter('mask_erode_px', 2)

        self.cfg = SceneConfig(self.get_parameter('scene_config').value, self.get_parameter('assets_dir').value)
        self.timeout = float(self.get_parameter('timeout').value)
        self.erode = int(self.get_parameter('mask_erode_px').value)
        ns = self.get_parameter('fp_namespace').value.rstrip('/')
        cb = ReentrantCallbackGroup()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.io = {}
        self.corners = {}
        for name in self.cfg.object_names:
            base = '%s/%s/pose_estimation/' % (ns, name)
            self.io[name] = {
                'image': self.create_publisher(Image, base + 'image', 1),
                'depth': self.create_publisher(Image, base + 'depth_image', 1),
                'info': self.create_publisher(CameraInfo, base + 'camera_info', 1),
                'mask': self.create_publisher(Image, base + 'segmentation', 1),
                'sub': self.create_subscription(Detection3DArray, base + 'output',
                                                lambda m, n=name: self.on_output(n, m),
                                                qos_profile_sensor_data, callback_group=cb),
            }
            try:
                v = read_obj_vertices(self.cfg.mesh_path(name))
                self.corners[name] = box_corners(v.min(axis=0), v.max(axis=0))
            except OSError:
                self.get_logger().warn('mesh missing for %s (%s)' % (name, self.cfg.mesh_path(name)))

        self.lock = threading.Lock()          # one request in flight
        self.pending = None                   # (name, stamp_sec, event, [result])
        self.pose_pub = self.create_publisher(PoseStamped, '/perception/target_pose', 1)
        self.overlay_pub = self.create_publisher(Image, '/perception/pose_overlay', 1)
        self.srv = self.create_service(EstimatePose, '/perception/estimate_pose', self.on_request,
                                       callback_group=cb)
        self.get_logger().info('FoundationPose bridge ready for %d objects' % len(self.io))

    def on_output(self, name, msg):
        p = self.pending
        if p is None or p[0] != name or not msg.detections:
            return
        # FoundationPose keeps the input stamp; anything older belongs to a previous request
        if stamp_to_sec(msg.header.stamp) < p[1] - 1e-3:
            self.get_logger().debug('ignoring FoundationPose output with stale stamp')
            return
        det = max(msg.detections, key=lambda d: d.results[0].hypothesis.score if d.results else 0.0)
        if not det.results:
            return
        p[3].append((msg.header.frame_id, pose_to_T(det.results[0].pose.pose)))
        p[2].set()

    def camera_in_world(self, frame_id):
        try:
            t = self.tf_buffer.lookup_transform(self.cfg.world_frame, frame_id, rclpy.time.Time())
            return transform_to_T(t.transform)
        except Exception:
            return self.cfg.T_world_camera()

    def on_request(self, req, res):
        sw = Stopwatch()
        if req.target not in self.io:
            res.success, res.message = False, 'no FoundationPose instance for "%s"' % req.target
            return res
        with self.lock:
            header = req.rgb.header
            mask = image_to_numpy(req.mask)
            if self.erode > 0:
                mask = cv2.erode(mask, np.ones((2 * self.erode + 1,) * 2, np.uint8))
            info = req.camera_info
            info.header = header
            depth = req.depth
            depth.header = header
            event = threading.Event()
            self.pending = (req.target, stamp_to_sec(header.stamp), event, [])
            io = self.io[req.target]
            io['info'].publish(info)
            io['depth'].publish(depth)
            io['mask'].publish(numpy_to_image(mask, 'mono8', header))
            io['image'].publish(req.rgb)
            ok = event.wait(self.timeout)
            result = self.pending[3]
            self.pending = None
        res.latency_ms = sw.ms()
        if not ok or not result:
            res.success, res.message = False, 'FoundationPose timed out after %.1fs' % self.timeout
            return res
        frame_id, T_cam_obj = result[0]
        frame_id = frame_id or info.header.frame_id
        T_world_obj = self.camera_in_world(frame_id) @ T_cam_obj
        res.success, res.message = True, 'ok'
        res.pose_camera = T_to_pose_stamped(T_cam_obj, frame_id, header.stamp)
        res.pose = T_to_pose_stamped(T_world_obj, self.cfg.world_frame, header.stamp)
        self.pose_pub.publish(res.pose)
        if req.target in self.corners:
            K = np.array(info.k).reshape(3, 3)
            self.overlay_pub.publish(numpy_to_image(
                draw_pose(image_to_numpy(req.rgb), K, T_cam_obj, self.corners[req.target]), 'rgb8', header))
        p = T_world_obj[:3, 3]
        self.get_logger().info('pose %s: world xyz=(%.3f, %.3f, %.3f) %.1f ms'
                               % (req.target, p[0], p[1], p[2], res.latency_ms))
        return res


def main():
    rclpy.init()
    node = PoseNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
