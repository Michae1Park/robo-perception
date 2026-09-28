"""ROS message <-> numpy conversions (no cv_bridge dependency)."""
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from sensor_msgs.msg import Image

from . import transforms as tf

_ENC = {
    'rgb8': (np.uint8, 3), 'bgr8': (np.uint8, 3), 'rgba8': (np.uint8, 4),
    'mono8': (np.uint8, 1), '8UC1': (np.uint8, 1),
    '32FC1': (np.float32, 1), '16UC1': (np.uint16, 1),
}


def image_to_numpy(msg):
    dtype, ch = _ENC[msg.encoding]
    itemsize = np.dtype(dtype).itemsize
    row = msg.step // itemsize
    arr = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, row)
    arr = arr[:, :msg.width * ch]
    if ch > 1:
        arr = arr.reshape(msg.height, msg.width, ch)
    if msg.encoding == 'bgr8':
        arr = arr[..., ::-1]
    if msg.encoding == 'rgba8':
        arr = arr[..., :3]
    if msg.encoding == '16UC1':
        arr = arr.astype(np.float32) / 1000.0  # mm -> m
    return np.ascontiguousarray(arr)


def numpy_to_image(arr, encoding, header=None):
    msg = Image()
    if header is not None:
        msg.header = header
    arr = np.ascontiguousarray(arr)
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = arr.strides[0]
    msg.data = arr.tobytes()
    return msg


def pose_to_T(pose):
    p = pose.position
    q = pose.orientation
    return tf.T_from_pq([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])


def T_to_pose(T):
    p, q = tf.pq_from_T(T)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = [float(v) for v in p]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = [float(v) for v in q]
    return pose


def T_to_pose_stamped(T, frame_id, stamp=None):
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    if stamp is not None:
        ps.header.stamp = stamp
    ps.pose = T_to_pose(T)
    return ps


def transform_to_T(transform):
    t = transform.translation
    q = transform.rotation
    return tf.T_from_pq([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])


def T_to_transform_stamped(T, parent, child, stamp=None):
    ts = TransformStamped()
    ts.header.frame_id = parent
    if stamp is not None:
        ts.header.stamp = stamp
    ts.child_frame_id = child
    p, q = tf.pq_from_T(T)
    ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = [float(v) for v in p]
    r = ts.transform.rotation
    r.x, r.y, r.z, r.w = [float(v) for v in q]
    return ts


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9
