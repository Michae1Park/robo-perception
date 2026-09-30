"""Evaluation only (D-010): the one node that reads the sim's ground truth (/sim/gt/*). Nothing in the pipeline may
depend on it - the task never calls it; test harnesses and episode runners do.

  - scores every pose_node estimate (~/pose) against the ground-truth object pose at the same time:
    rotation error, translation error, and the error of the object's z axis (the meaningful part for the
    rotationally symmetric can); published on ~/pose_error and logged
  - ~/report (std_srvs/Trigger): JSON with each target's ground-truth location now (table / shelf level k /
    elsewhere, how upright) plus all pose errors so far
"""
import json

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .common import load_config, pose_to_T, stamp_sec, tf


class EvalNode(Node):
    def __init__(self):
        super().__init__('eval_node')
        self.targets = list(self.declare_parameter('targets', ['mustard_bottle', 'tomato_soup_can']).value)
        sim = load_config()['sim']
        self.shelf, self.table = sim['shelf'], (sim['table_center'], sim['table_size'])
        self.gt = {t: [] for t in self.targets}          # target -> [(t, T)] recent history
        self.errors = {t: [] for t in self.targets}
        for t in self.targets:
            self.create_subscription(PoseStamped, '/sim/gt/%s/pose' % t, lambda m, t=t: self.on_gt(t, m), 10)
        self.create_subscription(PoseStamped, '/pose_node/pose', self.on_estimate, 10)
        self.err_pub = self.create_publisher(String, '~/pose_error', 10)
        self.create_service(Trigger, '~/report', self.on_report)
        self.get_logger().info('ready (ground truth for %s)' % ', '.join(self.targets))

    def on_gt(self, target, m):
        h = self.gt[target]
        h.append((stamp_sec(m.header.stamp), pose_to_T(m)))
        del h[:-100]

    def gt_at(self, target, t):
        h = self.gt[target]
        if not h:
            return None
        k = int(np.argmin([abs(s - t) for s, _ in h]))
        return h[k][1] if abs(h[k][0] - t) < 0.5 else None

    def on_estimate(self, m):
        t = stamp_sec(m.header.stamp)
        T = pose_to_T(m)
        # which target is it? the one whose ground truth is nearest (pose_node estimates one target per call)
        best = None
        for target in self.targets:
            G = self.gt_at(target, t)
            if G is not None:
                d = np.linalg.norm(G[:3, 3] - T[:3, 3])
                if best is None or d < best[1]:
                    best = (target, d, G)
        if best is None:
            self.get_logger().warn('no ground truth near t=%.2f' % t)
            return
        target, dist, G = best
        dR = G[:3, :3].T @ T[:3, :3]
        rot = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        axis = float(np.degrees(tf.angle_between(G[:3, 2], T[:3, 2])))
        e = {'target': target, 't': t, 'rot_deg': round(rot, 2), 'z_axis_deg': round(axis, 2),
             'trans_mm': round(float(dist) * 1000, 1)}
        self.errors[target].append(e)
        self.err_pub.publish(String(data=json.dumps(e)))
        self.get_logger().info('pose error %s: rotation %.2f deg (z axis %.2f deg), translation %.1f mm'
                               % (target, rot, axis, dist * 1000))

    def where(self, T):
        """Ground-truth location of an object pose: on the table, on a shelf level, or elsewhere."""
        x, y, z = T[:3, 3]
        s = self.shelf
        (cx, cy), (w, d) = s['center_xy'], s['size']
        if abs(x - cx) < w / 2 and abs(y - cy) < d / 2:
            k = int(np.argmin([abs(z - lz) for lz in s['levels_z']]))
            if 0 <= z - s['levels_z'][k] < 0.25:
                return 'shelf level %d' % k
        (tx, ty), (tw, tl, _) = self.table
        if abs(x - tx) < tw / 2 and abs(y - ty) < tl / 2 and -0.01 < z < 0.25:
            return 'table'
        return 'elsewhere'

    def on_report(self, _, res):
        out = {}
        for t in self.targets:
            if not self.gt[t]:
                out[t] = {'location': 'unknown (no ground truth received)'}
                continue
            T = self.gt[t][-1][1]
            out[t] = {'location': self.where(T), 'position': [round(float(v), 4) for v in T[:3, 3]],
                      'z_axis_tilt_deg': round(float(np.degrees(tf.angle_between(T[:3, 2], [0, 0, 1]))), 1),
                      'pose_errors': self.errors[t]}
        res.success, res.message = True, json.dumps(out)
        return res


def main():
    rclpy.init()
    node = EvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
