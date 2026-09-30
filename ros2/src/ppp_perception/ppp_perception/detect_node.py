"""Stage 1 node: YOLOE-seg on the fixed camera, prompted by one example image per target (or a phrase).
Default "visual+text": the example image finds candidates, a text-prompt model checks what each one is (a lookalike
- the potted meat can for the tomato can - is rejected rather than picked; see docs/DECISIONS.md).

Service ~/detect (ppp_interfaces/Detect): best detection of the requested target in the latest RGB-D frame.
With publish_for_pose, that exact frame (RGB, depth, camera_info) and the mask go out with one stamp on
  /ppp/fp/{image, depth_image, camera_info, segmentation}
- FoundationPose's inputs (D-016: it runs on every mask it receives, so masks are only published on request; pose_node
has loaded the target's mesh first, D-043). Also publishes ~/overlay (the detection drawn on the frame) for viewing.

Service ~/set_prompt (ppp_interfaces/SetPrompt): a user's prompt (a phrase, or an example image + box) says WHICH
object; the library says WHAT it is. The prompt's best detection in the latest frame must overlap a target found by
its stored example (the target's mesh is what FoundationPose needs). From then on that target's Detect sends the
prompt's mask (ui/prompt_ui.py). No text and no image: back to the stored example images.
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

from ppp_interfaces.srv import Detect, SetPrompt

from .common import REPO, image_to_numpy, load_config, numpy_to_image, ycb_name

REFS = os.path.join(REPO, 'data', 'multi_object_scene', 'refs')   # <ycb name>.png + .json: the example images
CELL_REFS = os.path.join(REPO, 'data', 'cell_refs')   # the same, for objects not in that scene: taken once from the
                                                      # cell's fixed camera (rendered side views matched weakly, D-043)


def ref_image(target):
    name = ycb_name(target) + '.png'
    return next(p for p in (os.path.join(REFS, name), os.path.join(CELL_REFS, name)) if os.path.isfile(p))


MATCH_IOU = 0.5   # a prompt's mask and a library detection this similar are the same object
PROMPT_SIZES = (100, 160)   # px: object sizes an example image is also tried at (the table's objects are ~50-100 px)


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum()) / union if union else 0.0


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
        self.prompts = {}   # target -> (model, class name, description): a user's prompt (~/set_prompt)
        self.lock = threading.Lock()
        self.frame = None

        q = QoSProfile(depth=5)
        subs = [message_filters.Subscriber(self, Image, 'color/image_raw', qos_profile=q),
                message_filters.Subscriber(self, Image, 'depth/image_raw', qos_profile=q),
                message_filters.Subscriber(self, CameraInfo, 'color/camera_info', qos_profile=q)]
        self.sync = message_filters.TimeSynchronizer(subs, 10)
        self.sync.registerCallback(self.on_frame)
        self.fp_pub = {k: self.create_publisher(typ, '/ppp/fp/' + k, q) for k, typ in (
            ('image', Image), ('depth_image', Image), ('camera_info', CameraInfo), ('segmentation', Image))}
        self.overlay_pub = self.create_publisher(Image, '~/overlay', q)
        self.create_service(Detect, '~/detect', self.on_detect)
        self.create_service(SetPrompt, '~/set_prompt', self.on_set_prompt)
        self.get_logger().info('ready: %s prompts for %s' % (self.mode, ', '.join(self.targets)))

    def build_model(self):
        from vision.detect import build_text_prompt_model, build_visual_prompt_model, load_prompt
        if self.mode == 'text':
            return build_text_prompt_model(self.cfg['weights'], [t.replace('_', ' ') for t in self.targets],
                                           self.cfg['device'])
        prompts = [(t, *load_prompt(ref_image(t))) for t in self.targets]
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

    def latest_bgr(self):
        with self.lock:
            frame = self.frame
        if frame is None:
            return None, None
        return frame, np.ascontiguousarray(image_to_numpy(frame[0])[..., ::-1])

    def library(self, bgr, kw):
        """Every target found by its stored example image (+ text check): target -> best detection or None, and the
        text check's detections (for naming other things)."""
        from vision.detect import detect
        dets = detect(self.model, bgr, **kw)
        text_dets = detect(self.text_model, bgr, **kw) if self.text_model is not None else []
        found = {}
        for t in self.targets:
            cls = t.replace('_', ' ') if self.mode == 'text' else t
            mine = [d for d in dets if d['name'] == cls]
            if self.text_model is not None:
                for d in mine:
                    d['score'] = self.fused(d, text_dets, t)
            threshold = self.cfg['min_fused'] if self.text_model is not None else self.min_score
            ok = [d for d in mine if d['score'] >= threshold]
            found[t] = max(ok, key=lambda d: d['score']) if ok else None
        return found, text_dets

    def on_detect(self, req, res):
        from vision.detect import detect, detect_kwargs
        from vision.viz import draw_detection
        if req.target not in self.targets:
            res.message = 'unknown target "%s" (have: %s)' % (req.target, ', '.join(self.targets))
            return res
        frame, bgr = self.latest_bgr()
        if frame is None:
            res.message = 'no camera frame yet'
            return res
        rgb_msg, depth_msg, info = frame
        res.stamp = rgb_msg.header.stamp
        t0 = time.perf_counter()
        kw = detect_kwargs(self.cfg)
        known = self.library(bgr, kw)[0][req.target]
        best, label = known, req.target
        if known is not None and req.target in self.prompts:
            # the user's prompt picks the pixels; it must be the object the library knows as this target
            model, _, what = self.prompts[req.target]
            best = max([d for d in detect(model, bgr, **kw) if iou(d['mask'], known['mask']) > MATCH_IOU],
                       key=lambda d: d['score'], default=None)
            label = '%s -> %s' % (what, req.target)
        ms = (time.perf_counter() - t0) * 1000
        if best is None:
            res.message = ('no "%s" (%s), %.0f ms' % (req.target, 'prompt: ' + self.prompts[req.target][2]
                           if known is not None else 'not in the frame', ms))
            self.get_logger().warn(res.message)
            return res
        res.found, res.score = True, float(best['score'])
        res.bbox = [float(v) for v in best['bbox']]
        res.mask_pixels = int(best['mask'].sum())
        res.message = '%s score %.2f, mask %d px, %.0f ms' % (label, res.score, res.mask_pixels, ms)
        self.get_logger().info(res.message)
        if req.publish_for_pose:
            self.publish_for_pose(req.target, rgb_msg, depth_msg, info, best['mask'])
        vis = draw_detection(bgr, best, (255, 0, 255), label)
        self.overlay_pub.publish(numpy_to_image(vis[..., ::-1], 'rgb8', rgb_msg.header))
        return res

    def on_set_prompt(self, req, res):
        """The prompt says which object; the library (stored example + mesh per target) says what it is."""
        from vision.detect import (build_text_prompt_model, build_visual_prompt_model, detect, detect_kwargs,
                                   pad_prompt)
        from vision.viz import draw_detection
        text = req.text.strip()
        if not text and not req.image.data:
            self.prompts.clear()
            res.message = 'back to the stored example images'
            self.get_logger().info(res.message)
            return res
        frame, bgr = self.latest_bgr()
        if frame is None:
            res.message = 'no camera frame yet'
            return res
        t0 = time.perf_counter()
        if req.image.data:   # visual prompt: YOLOE pools its own features inside the box
            img = image_to_numpy(req.image)[..., :3]
            img = np.ascontiguousarray(img[..., ::-1] if req.image.encoding.startswith('rgb') else img)
            h, w = img.shape[:2]
            bbox = [float(v) for v in req.bbox] if any(req.bbox) else [0.0, 0.0, float(w - 1), float(h - 1)]
            what = 'example image'
            kw = detect_kwargs(self.cfg)
            model, dets = None, []   # as given, and padded to two object sizes (pad_prompt): keep the best match
            for im, bb in [(img, bbox)] + [pad_prompt(img, bbox, size) for size in PROMPT_SIZES]:
                m = build_visual_prompt_model(self.cfg['weights'], [('prompt', im, bb)], self.cfg['device'])
                d = detect(m, bgr, **kw)
                if d and (not dets or max(x['score'] for x in d) > max(x['score'] for x in dets)):
                    model, dets = m, d
        else:
            what = '"%s"' % text
            model = build_text_prompt_model(self.cfg['weights'], [text], self.cfg['device'])
            kw = detect_kwargs(self.cfg)
            dets = detect(model, bgr, **kw)
        if not dets:
            res.message = 'nothing matches %s (%.0f ms)' % (what, (time.perf_counter() - t0) * 1000)
            self.get_logger().warn(res.message)
            return res
        best = max(dets, key=lambda d: d['score'])
        known, text_dets = self.library(bgr, kw)
        overlap = {t: iou(d['mask'], best['mask']) for t, d in known.items() if d is not None}
        target = max(overlap, key=overlap.get, default='')
        if overlap.get(target, 0.0) <= MATCH_IOU:
            target = ''
        named = [t for t in text_dets if iou(t['mask'], best['mask']) > MATCH_IOU]
        res.label = target.replace('_', ' ') or (max(named, key=lambda t: t['score'])['name'] if named
                                                 else 'unknown object')
        res.found, res.score, res.target = True, float(best['score']), target
        res.bbox = [float(v) for v in best['bbox']]
        ms = (time.perf_counter() - t0) * 1000
        if target:
            self.prompts[target] = (model, 'prompt', what)
            res.message = '%s -> %s (score %.2f, overlap %.2f), %.0f ms' % (what, target, res.score, overlap[target], ms)
            colour, caption = (255, 0, 255), '%s -> %s' % (what, target)
        else:
            res.message = '%s -> %s, not one I know (%s), %.0f ms' % (
                what, res.label, ', '.join(t.replace('_', ' ') for t in self.targets), ms)
            colour, caption = (0, 140, 255), '%s -> %s?' % (what, res.label)
        self.get_logger().info(res.message)
        vis = draw_detection(bgr, best, colour, caption)
        self.overlay_pub.publish(numpy_to_image(vis[..., ::-1], 'rgb8', frame[0].header))
        return res

    def publish_for_pose(self, target, rgb, depth, info, mask):
        p = self.fp_pub
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
