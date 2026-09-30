"""Shared helpers for the perception nodes: the repo's vision/ package, object names, message <-> numpy.

The nodes run on the repo's .venv Python (torch, ultralytics, numpy 2) with ROS 2 Jazzy sourced (I-012); the
launch files set PPP_REPO and use .venv/bin/python as the interpreter.
"""
import os
import sys
import threading

import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.duration import Duration
from rclpy.time import Time as RTime
from sensor_msgs.msg import Image

REPO = os.environ.get('PPP_REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                  '..', '..', '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from vision import load_config, read_obj_vertices  # noqa: E402,F401  (re-exported)
from vision import transforms as tf  # noqa: E402

BASE = 'panda_link0'
YCB = os.path.join(REPO, 'assets', 'ycb')


def ycb_name(target):
    """'mustard_bottle' -> '006_mustard_bottle' (the assets/ycb folder). YCB names are also accepted as-is."""
    if os.path.isdir(os.path.join(YCB, target)):
        return target
    for d in sorted(os.listdir(YCB)):
        if d.split('_', 1)[-1] == target:
            return d
    raise KeyError('no YCB object "%s" in %s' % (target, YCB))


def mesh_path(target):
    return os.path.join(YCB, ycb_name(target), 'textured.obj')


def mesh_vertices(target, _cache={}):
    """Vertices of the target's mesh (object frame), read once."""
    if target not in _cache:
        _cache[target] = read_obj_vertices(mesh_path(target))
    return _cache[target]


# ------------------------------------------------------------------ images
def image_to_numpy(msg):
    """sensor_msgs/Image -> HxW(xC) array (rgb8 / bgr8 / mono8 / 16UC1 / 32FC1)."""
    dtype, ch = {'rgb8': (np.uint8, 3), 'bgr8': (np.uint8, 3), 'rgba8': (np.uint8, 4), 'mono8': (np.uint8, 1),
                 '16UC1': (np.uint16, 1), 'mono16': (np.uint16, 1), '32FC1': (np.float32, 1)}[msg.encoding]
    item = np.dtype(dtype).itemsize
    a = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // item)[:, :msg.width * ch]
    return a.reshape(msg.height, msg.width, ch) if ch > 1 else a


def numpy_to_image(arr, encoding, header):
    m = Image()
    m.header = header
    m.height, m.width = arr.shape[:2]
    m.encoding, m.is_bigendian = encoding, 0
    arr = np.ascontiguousarray(arr)
    m.step = arr.strides[0]
    m.data = arr.tobytes()
    return m


def depth_m(msg):
    """Depth image -> float metres (0 = no data)."""
    d = image_to_numpy(msg)
    return d.astype(np.float64) / 1000.0 if d.dtype == np.uint16 else np.nan_to_num(d.astype(np.float64))


def camera_K(info):
    return np.array(info.k, dtype=float).reshape(3, 3)


# ------------------------------------------------------------------ poses
def quat_to_R(q):
    return tf.quat_to_R((q.x, q.y, q.z, q.w))


def pose_to_T(p):
    """geometry_msgs Pose (or PoseStamped) -> 4x4."""
    p = p.pose if isinstance(p, PoseStamped) else p
    return tf.make_T(quat_to_R(p.orientation), [p.position.x, p.position.y, p.position.z])


def T_to_pose(T):
    p = Pose()
    p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, tf.R_to_quat(T[:3, :3]))
    return p


def T_to_pose_stamped(T, frame, stamp):
    m = PoseStamped()
    m.header.frame_id, m.header.stamp = frame, stamp
    m.pose = T_to_pose(T)
    return m


def lookup_T(buffer, target, source, stamp, timeout=1.0):
    """TF lookup -> 4x4 (frame `source` in frame `target`) at a message stamp."""
    t = buffer.lookup_transform(target, source, RTime.from_msg(stamp), timeout=Duration(seconds=timeout))
    r, p = t.transform.rotation, t.transform.translation
    return tf.make_T(quat_to_R(r), [p.x, p.y, p.z])


def stamp_sec(s):
    return s.sec + s.nanosec * 1e-9


# ------------------------------------------------------------------ calling services from a service callback
def call(node, client, request, timeout=30.0):
    """Blocking service call from inside a callback. The node must spin on a MultiThreadedExecutor and the client
    must be in a reentrant (or other) callback group, so the response can arrive while this thread waits."""
    if not client.wait_for_service(timeout_sec=min(timeout, 5.0)):
        raise RuntimeError('service %s not available' % client.srv_name)
    done = threading.Event()
    fut = client.call_async(request)
    fut.add_done_callback(lambda _: done.set())
    if not done.wait(timeout):
        client.remove_pending_request(fut)
        raise TimeoutError('%s timed out after %.0f s' % (client.srv_name, timeout))
    return fut.result()
