"""① Visual-prompt target detection (YOLOE-seg, PyTorch or TensorRT).

Service  /perception/detect   ppp_interfaces/DetectTarget
Publishes /perception/detection_overlay  sensor_msgs/Image (rgb8) for inspection / README figures
"""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from ppp_common.config import SceneConfig
from ppp_common.ros_conv import image_to_numpy, numpy_to_image
from ppp_common.ros_utils import Stopwatch
from ppp_interfaces.srv import DetectTarget

from .yoloe_prompt import detect, load_detector


def draw_detections(rgb, dets, target, chosen):
    img = rgb.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d['bbox']]
        is_chosen = d is chosen
        color = (0, 255, 0) if is_chosen else ((255, 200, 0) if d['name'] == target else (160, 160, 160))
        if is_chosen and d['mask'] is not None:
            overlay = img.copy()
            overlay[d['mask']] = (0, 255, 0)
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, '%s %.2f' % (d['name'], d['score']), (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('assets_dir', '')
        self.declare_parameter('weights', 'yoloe-11l-seg.pt')
        self.declare_parameter('engine', '')          # .engine path -> TensorRT backend
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('half', False)
        self.declare_parameter('conf', 0.05)
        self.declare_parameter('min_mask_pixels', 150)

        cfg = SceneConfig(self.get_parameter('scene_config').value, self.get_parameter('assets_dir').value)
        self.device = self.get_parameter('device').value
        self.half = bool(self.get_parameter('half').value)
        self.conf = float(self.get_parameter('conf').value)
        self.min_mask_pixels = int(self.get_parameter('min_mask_pixels').value)
        engine = self.get_parameter('engine').value
        self.backend = 'tensorrt' if engine else 'pytorch'

        self.names = cfg.object_names
        self.model = load_detector(self.get_parameter('weights').value, engine, self.names,
                                   cfg.prompt_path, self.device)
        detect(self.model, np.zeros((cfg.camera['height'], cfg.camera['width'], 3), np.uint8),
               self.conf, self.device, self.half)  # warm-up
        self.overlay_pub = self.create_publisher(Image, '/perception/detection_overlay', 1)
        self.srv = self.create_service(DetectTarget, '/perception/detect', self.on_detect)
        self.get_logger().info('YOLOE visual-prompt detector ready (%s), %d prompts: %s'
                               % (self.backend, len(self.names), ', '.join(self.names)))

    def on_detect(self, req, res):
        sw = Stopwatch()
        if req.target not in self.names:
            res.success = False
            res.message = 'no visual prompt registered for "%s"' % req.target
            return res
        rgb = image_to_numpy(req.rgb)
        dets = detect(self.model, np.ascontiguousarray(rgb[..., ::-1]), self.conf, self.device, self.half)
        cands = [d for d in dets if d['name'] == req.target and d['mask'] is not None
                 and d['mask'].sum() >= self.min_mask_pixels]
        chosen = max(cands, key=lambda d: d['score']) if cands else None
        res.latency_ms = sw.ms()
        if chosen is None:
            res.success = False
            res.message = 'target "%s" not found (%d detections)' % (req.target, len(dets))
        else:
            res.success = True
            res.bbox = chosen['bbox']
            res.score = chosen['score']
            res.mask = numpy_to_image((chosen['mask'] * 255).astype(np.uint8), 'mono8', req.rgb.header)
            res.message = 'ok'
        self.overlay_pub.publish(numpy_to_image(draw_detections(rgb, dets, req.target, chosen),
                                                'rgb8', req.rgb.header))
        self.get_logger().info('detect %s: %s score=%.2f %.1f ms' % (
            req.target, res.message, res.score, res.latency_ms))
        return res


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
