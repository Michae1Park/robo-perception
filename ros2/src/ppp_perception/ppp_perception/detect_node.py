"""Stage 1 node: YOLOE-seg on the fixed camera, prompted by one example image per target (or a phrase).
Default "visual+text": the example image finds candidates, a text-prompt model checks what each one is (a lookalike
- the potted meat can for the tomato can - is rejected rather than picked; see docs/DECISIONS.md).

Service ~/detect (ppp_interfaces/Detect): best detection of the requested target in the latest RGB-D frame.
With publish_for_pose, that exact frame (RGB, depth, camera_info) and the mask go out with one stamp on
  /ppp/fp/<target>/{image, depth_image, camera_info, segmentation}
- the inputs of that target's FoundationPose instance (D-016: it runs on every mask it receives, so masks are
only published on request). Also publishes ~/overlay (the detection drawn on the frame) for viewing.
"""
import os
import threading
import time

import message_filters
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo, Image

from ppp_interfaces.srv import Detect

from .common import REPO, image_to_numpy, load_config, numpy_to_image, ycb_name

REFS = os.path.join(REPO, 'data', 'multi_object_scene', 'refs')   # <ycb name>.png + .json: the example images


class DetectNode(Node):
    def __init__(self):
        super().__init__('detect_node')
        self.targets = list(self.declare_parameter('targets', ['mustard_bottle', 'tomato_soup_can']).value)
        self.mode = self.declare_parameter('prompt', 'visual+text').value       # visual | text | visual+text
        self.fp_depth = self.declare_parameter('fp_depth_encoding', '32FC1').value   # what FoundationPose gets
        self.min_score = self.declare_parameter('min_score', 0.2).value
        self.cfg = load_config()['detect']
        self.model = self.build_model()
        self.text_model, self.text_names = None, []
        if self.mode == 'visual+text':   # the checker: target names + a generic vocabulary (config.yaml detect)
            from vision.detect import build_text_prompt_model
            self.text_names = [t.replace('_', ' ') for t in self.targets] + list(self.cfg['vocabulary'])
            self.text_model = build_text_prompt_model(self.cfg['weights'], self.text_names, self.cfg['device'])
        self.lock = threading.Lock()
        self.frame = None

        q = QoSProfile(depth=5)
        subs = [message_filters.Subscriber(self, Image, 'color/image_raw', qos_profile=q),
                message_filters.Subscriber(self, Image, 'depth/image_raw', qos_profile=q),
                message_filters.Subscriber(self, CameraInfo, 'color/camera_info', qos_profile=q)]
        self.sync = message_filters.TimeSynchronizer(subs, 10)
        self.sync.registerCallback(self.on_frame)
        self.fp_pub = {t: {k: self.create_publisher(typ, '/ppp/fp/%s/%s' % (t, k), q) for k, typ in (
            ('image', Image), ('depth_image', Image), ('camera_info', CameraInfo), ('segmentation', Image))}
            for t in self.targets}
        self.overlay_pub = self.create_publisher(Image, '~/overlay', q)
        self.create_service(Detect, '~/detect', self.on_detect)
        self.get_logger().info('ready: %s prompts for %s' % (self.mode, ', '.join(self.targets)))

    def build_model(self):
        from vision.detect import build_text_prompt_model, build_visual_prompt_model, load_prompt
        if self.mode == 'text':
            return build_text_prompt_model(self.cfg['weights'], [t.replace('_', ' ') for t in self.targets],
                                           self.cfg['device'])
        prompts = [(t, *load_prompt(os.path.join(REFS, ycb_name(t) + '.png'))) for t in self.targets]
        return build_visual_prompt_model(self.cfg['weights'], prompts, self.cfg['device'])

    def fused(self, d, text_dets, target):
        """Visual score + text score for the target's name - best text score for any other name, on the same mask."""
        name = target.replace('_', ' ')
        same = [t for t in text_dets if (t['mask'] & d['mask']).sum() > 0.5 * (t['mask'] | d['mask']).sum()]
        mine = max([t['score'] for t in same if t['name'] == name], default=0.0)
        other = max([t['score'] for t in same if t['name'] != name], default=0.0)
        return d['score'] + mine - other

    def on_frame(self, rgb, depth, info):
        with self.lock:
            self.frame = (rgb, depth, info)

    def on_detect(self, req, res):
        from vision.detect import detect, detect_kwargs
        from vision.viz import draw_detection
        if req.target not in self.targets:
            res.message = 'unknown target "%s" (have: %s)' % (req.target, ', '.join(self.targets))
            return res
        with self.lock:
            frame = self.frame
        if frame is None:
            res.message = 'no camera frame yet'
            return res
        rgb_msg, depth_msg, info = frame
        bgr = np.ascontiguousarray(image_to_numpy(rgb_msg)[..., ::-1])
        t0 = time.perf_counter()
        cls = req.target.replace('_', ' ') if self.mode == 'text' else req.target
        dets = [d for d in detect(self.model, bgr, **detect_kwargs(self.cfg)) if d['name'] == cls]
        res.stamp = rgb_msg.header.stamp
        if self.text_model is not None and dets:
            text_dets = detect(self.text_model, bgr, **detect_kwargs(self.cfg))
            for d in dets:
                d['score'] = self.fused(d, text_dets, req.target)
            threshold = self.cfg['min_fused']
        else:
            threshold = self.min_score
        ms = (time.perf_counter() - t0) * 1000
        if not dets or max(d['score'] for d in dets) < threshold:
            res.message = 'no "%s" above %.2f (%d weaker), %.0f ms' % (req.target, threshold, len(dets), ms)
            self.get_logger().warn(res.message)
            return res
        best = max(dets, key=lambda d: d['score'])
        res.found, res.score = True, float(best['score'])
        res.bbox = [float(v) for v in best['bbox']]
        res.mask_pixels = int(best['mask'].sum())
        res.message = '%s score %.2f, mask %d px, %.0f ms' % (req.target, res.score, res.mask_pixels, ms)
        self.get_logger().info(res.message)
        if req.publish_for_pose:
            self.publish_for_pose(req.target, rgb_msg, depth_msg, info, best['mask'])
        vis = draw_detection(bgr, best, (255, 0, 255), '%s %.2f' % (req.target, res.score))
        self.overlay_pub.publish(numpy_to_image(vis[..., ::-1], 'rgb8', rgb_msg.header))
        return res

    def publish_for_pose(self, target, rgb, depth, info, mask):
        p = self.fp_pub[target]
        if self.fp_depth == '32FC1' and depth.encoding == '16UC1':
            d = (image_to_numpy(depth).astype(np.float32) / 1000.0)
            depth = numpy_to_image(d, '32FC1', depth.header)
        p['segmentation'].publish(numpy_to_image(mask.astype(np.uint8) * 255, 'mono8', rgb.header))
        p['camera_info'].publish(info)
        p['depth_image'].publish(depth)
        p['image'].publish(rgb)


def main():
    rclpy.init()
    node = DetectNode()
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
