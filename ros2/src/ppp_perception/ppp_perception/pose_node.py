"""Stage 2 node: EstimatePose = detect the target, then its 6-DoF pose from FoundationPose.

Service ~/estimate_pose (ppp_interfaces/EstimatePose), one request at a time:
  1. load the target's mesh into the one FoundationPose instance (its mesh_file_path parameter, D-043)
  2. detect_node ~/detect with publish_for_pose -> the frame + mask go to /ppp/fp/...
  3. wait for FoundationPose's answer on /ppp/fp/output (vision_msgs/Detection3DArray) with that frame's stamp
     (Isaac ROS 4.5 in its container: D-012, D-016)
  4. camera -> base frame through TF (calibrated extrinsics), returned and published on ~/pose
"""
import threading

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray

from ppp_interfaces.srv import Detect, EstimatePose

from .common import BASE, T_to_pose_stamped, call, lookup_T, mesh_path, pose_to_T, stamp_sec

FP_NODE = '/ppp/fp/foundationpose'


class PoseNode(Node):
    def __init__(self):
        super().__init__('pose_node')
        self.targets = list(self.declare_parameter('targets', ['mustard_bottle', 'tomato_soup_can']).value)
        self.timeout = self.declare_parameter('timeout', 15.0).value
        self.cb = ReentrantCallbackGroup()
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self, spin_thread=True)
        self.detect = self.create_client(Detect, '/detect_node/detect', callback_group=self.cb)
        self.set_fp_params = self.create_client(SetParameters, FP_NODE + '/set_parameters', callback_group=self.cb)
        self.mesh = None   # the mesh FoundationPose has loaded (as far as this node knows)
        self.busy = threading.Lock()   # one FoundationPose, so one request at a time
        self.cond = threading.Condition()
        self.latest = None   # Detection3DArray
        self.create_subscription(Detection3DArray, '/ppp/fp/output', self.on_output, 5, callback_group=self.cb)
        self.pose_pub = self.create_publisher(PoseStamped, '~/pose', 5)
        self.create_service(EstimatePose, '~/estimate_pose', self.on_request, callback_group=self.cb)
        self.get_logger().info('ready: one FoundationPose for %s' % ', '.join(self.targets))

    def on_output(self, msg):
        with self.cond:
            self.latest = msg
            self.cond.notify_all()

    def wait_output(self, stamp):
        """The FoundationPose result for the frame with this stamp (it may answer an older request first)."""
        want = stamp_sec(stamp)
        with self.cond:
            ok = self.cond.wait_for(lambda: self.latest is not None and
                                    stamp_sec(self.latest.header.stamp) >= want - 1e-6, self.timeout)
            return self.latest if ok else None

    def load_mesh(self, target):
        """Point FoundationPose at the target's mesh; it reloads it when the parameter changes."""
        path = mesh_path(target)
        if path == self.mesh:
            return
        r = call(self, self.set_fp_params, SetParameters.Request(
            parameters=[Parameter('mesh_file_path', value=path).to_parameter_msg()]), timeout=10.0)
        if not r.results[0].successful:
            raise RuntimeError(r.results[0].reason or 'rejected')
        self.mesh = path
        self.get_logger().info('FoundationPose mesh: %s' % target)

    def on_request(self, req, res):
        if req.target not in self.targets:
            res.message = 'no mesh for "%s" (have: %s)' % (req.target, ', '.join(self.targets))
            return res
        with self.busy:
            return self.estimate(req, res)

    def estimate(self, req, res):
        try:
            self.load_mesh(req.target)
        except Exception as e:  # noqa: BLE001
            self.mesh = None
            res.message = 'could not load the mesh into FoundationPose: %s' % e
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
        out = self.wait_output(det.stamp)
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
