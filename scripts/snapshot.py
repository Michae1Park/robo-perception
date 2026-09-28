#!/usr/bin/env python3
"""Save the latest image of each topic as PNG (README figures, benchmark frames).

  python3 scripts/snapshot.py --out results/figures            # one shot of the default topics
  python3 scripts/snapshot.py --out frames --topics /camera/color/image_raw --count 50 --period 0.5
"""
import argparse
import os
import sys
import time

import cv2
import rclpy
from sensor_msgs.msg import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ros2', 'ppp_common'))
from ppp_common.ros_conv import image_to_numpy  # noqa: E402

DEFAULT = ['/camera/color/image_raw', '/perception/detection_overlay', '/perception/pose_overlay']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/figures')
    ap.add_argument('--topics', nargs='*', default=DEFAULT)
    ap.add_argument('--count', type=int, default=1)
    ap.add_argument('--period', type=float, default=0.5)
    ap.add_argument('--timeout', type=float, default=10.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node('snapshot')
    latest = {}
    for t in a.topics:
        node.create_subscription(Image, t, lambda m, t=t: latest.__setitem__(t, m), 2)
    for i in range(a.count):
        t0 = time.monotonic()
        latest.clear()
        while len(latest) < len(a.topics) and time.monotonic() - t0 < a.timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
        for t, msg in latest.items():
            img = image_to_numpy(msg)
            if img.ndim == 3:
                img = img[..., ::-1]
            elif img.dtype != 'uint8':
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            name = t.strip('/').replace('/', '_') + ('_%03d' % i if a.count > 1 else '') + '.png'
            cv2.imwrite(os.path.join(a.out, name), img)
        missing = set(a.topics) - set(latest)
        if missing:
            print('no message on', ', '.join(sorted(missing)))
        time.sleep(a.period if a.count > 1 else 0)
    print('saved to', a.out)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
