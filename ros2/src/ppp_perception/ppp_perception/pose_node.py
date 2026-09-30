"""Stage 2 node: EstimatePose = detect the target, then its 6-DoF pose from that target's FoundationPose instance.

Service ~/estimate_pose (ppp_interfaces/EstimatePose):
  1. detect_node ~/detect with publish_for_pose -> the frame + mask go to /ppp/fp/<target>/...
  2. wait for FoundationPose's answer on /ppp/fp/<target>/output (vision_msgs/Detection3DArray) with that frame's
     stamp (Isaac ROS 4.5 in its container, one instance per target: D-012, D-016)
  3. camera -> base frame through TF (calibrated extrinsics), returned and published on ~/pose
"""
import threading

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray

from ppp_interfaces.srv import Detect, EstimatePose

from .common import BASE, T_to_pose_stamped, call, lookup_T, pose_to_T, stamp_sec


class PoseNode(Node):
    def __init__(self):
        super().__init__('pose_node')
        self.targets = list(self.declare_parameter('targets', ['mustard_bottle', 'tomato_soup_can']).value)
        self.timeout = self.declare_parameter('timeout', 15.0).value
        self.cb = ReentrantCallbackGroup()
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self, spin_thread=True)
        self.detect = self.create_client(Detect, '/detect_node/detect', callback_group=self.cb)
        self.cond = threading.Condition()
        self.latest = {}   # target -> Detection3DArray
        for t in self.targets:
            self.create_subscription(Detection3DArray, '/ppp/fp/%s/output' % t,
                                     lambda m, t=t: self.on_output(t, m), 5, callback_group=self.cb)
        self.pose_pub = self.create_publisher(PoseStamped, '~/pose', 5)
        self.create_service(EstimatePose, '~/estimate_pose', self.on_request, callback_group=self.cb)
        self.get_logger().info('ready: FoundationPose outputs for %s' % ', '.join(self.targets))

    def on_output(self, target, msg):
        with self.cond:
            self.latest[target] = msg
            self.cond.notify_all()

    def wait_output(self, target, stamp):
        """The FoundationPose result for the frame with this stamp (it may answer an older request first)."""
        want = stamp_sec(stamp)
        with self.cond:
            ok = self.cond.wait_for(lambda: target in self.latest and
                                    stamp_sec(self.latest[target].header.stamp) >= want - 1e-6, self.timeout)
            return self.latest[target] if ok else None

    def on_request(self, req, res):
        if req.target not in self.targets:
            res.message = 'no FoundationPose instance for "%s"' % req.target
            return res
        try:
            det = call(self, self.detect, Detect.Request(target=req.target, publish_for_pose=True))
        except Exception as e:  # noqa: BLE001  (report any failure to the caller)
            res.message = 'detect failed: %s' % e
            return res
        res.detection_score = det.score
        if not det.found:
            res.message = 'not detected: ' + det.message
            return res
        out = self.wait_output(req.target, det.stamp)
        if out is None or not out.detections:
            res.message = 'no FoundationPose result within %.0f s' % self.timeout
            return res
        d = out.detections[0]
        p = d.results[0].pose.pose if d.results else d.bbox.center
        try:
            T_base_cam = lookup_T(self.tf, BASE, out.header.frame_id, out.header.stamp)
        except Exception as e:  # noqa: BLE001
            res.message = 'no extrinsics for %s: %s' % (out.header.frame_id, e)
            return res
        T = T_base_cam @ pose_to_T(p)
        res.pose = T_to_pose_stamped(T, BASE, out.header.stamp)
        res.success = True
        res.message = '%s at (%.3f, %.3f, %.3f) m in %s, detection %.2f' % (req.target, *T[:3, 3], BASE, det.score)
        self.pose_pub.publish(res.pose)
        self.get_logger().info(res.message)
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
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
