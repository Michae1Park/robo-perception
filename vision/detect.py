"""Stage 1: YOLOE-seg detection, prompted by an example image (visual) or a phrase (text).

YOLOE has no fixed class list: each class is an embedding vector, and set_classes() swaps them in.
  text   : MobileCLIP encodes the phrase                          -> build_text_prompt_model()
  visual : YOLOE pools its own features inside a box on a ref image -> build_visual_prompt_model()
  none   : *-seg-pf.pt weights ship a fixed 4585-class vocabulary   -> build_prompt_free_model()
One forward pass then scores every image region against every class.
"""
import json
import os
import warnings

import numpy as np

from . import REPO

MODELS_DIR = os.path.join(REPO, 'models')  # yoloe-*.pt + mobileclip_blt.ts (text encoder); auto-downloaded here


def weights_path(weights):
    """Bare file names (config.yaml style) resolve to models/; paths are used as given."""
    return weights if os.path.dirname(weights) else os.path.join(MODELS_DIR, weights)


def load_prompt(image_path):
    """<name>.png + <name>.json ({"bbox": [x0, y0, x1, y1]}) -> (bgr image, bbox)."""
    import cv2
    with open(os.path.splitext(image_path)[0] + '.json') as f:
        return cv2.imread(image_path), [float(v) for v in json.load(f)['bbox']]


def pad_prompt(img, bbox, size, canvas=(480, 640)):
    """A tight example crop -> the object `size` px across, centred on a canvas of the crop's border colour.
    YOLOE pools the prompt's features at its own scale: an object that fills the whole prompt image matches objects
    that fill the scene, not ones 50-100 px across."""
    import cv2
    h, w = img.shape[:2]
    s = size / max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    fill = np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]), axis=0)
    ch, cw = max(canvas[0], img.shape[0]), max(canvas[1], img.shape[1])
    out = np.empty((ch, cw, 3), np.uint8)
    out[:] = fill
    y, x = (ch - img.shape[0]) // 2, (cw - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out, [bbox[0] * s + x, bbox[1] * s + y, bbox[2] * s + x, bbox[3] * s + y]


def build_visual_prompt_model(weights, prompts, device='cuda:0'):
    """prompts: list of (name, bgr_image, bbox). Returns a YOLOE model whose classes are the names."""
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    model = YOLOE(weights_path(weights))
    vpes = []
    for _, img, bbox in prompts:  # one predict per ref image; it leaves the box's embedding in model.model.pe
        vp = dict(bboxes=np.array([bbox], dtype=np.float32), cls=np.array([0]))
        model.predict(img, refer_image=img, visual_prompts=vp, predictor=YOLOEVPSegPredictor,
                      device=device, verbose=False)
        vpes.append(model.model.pe.detach().cpu().reshape(1, 1, -1))
    model.set_classes([p[0] for p in prompts], F.normalize(torch.cat(vpes, dim=1), dim=-1))
    model.predictor = None  # drop the visual-prompt predictor; the next predict() builds a normal one
    return model


def build_text_prompt_model(weights, texts, device='cuda:0'):
    """texts: list of class-name phrases. Returns a YOLOE model whose classes are those phrases."""
    from ultralytics import YOLOE

    model = YOLOE(weights_path(weights))
    cwd = os.getcwd()
    os.chdir(MODELS_DIR)  # ultralytics looks for / downloads mobileclip_blt.ts in the cwd
    try:
        with warnings.catch_warnings():  # newer torch deprecates the jit.load used for mobileclip_blt.ts
            warnings.filterwarnings('ignore', message='.*torch.jit.load.*', category=FutureWarning)
            model.set_classes(texts, model.get_text_pe(texts))
    finally:
        os.chdir(cwd)
    return model


def build_prompt_free_model(weights):
    """weights: a *-seg-pf.pt file. Returns a YOLOE model with its built-in vocabulary; set_classes() is not allowed."""
    from ultralytics import YOLOE

    if not weights.endswith('-pf.pt'):
        raise ValueError('prompt-free needs *-seg-pf.pt weights, got %s' % weights)
    return YOLOE(weights_path(weights))


def detect_kwargs(cfg):
    """The `detect` section of config.yaml -> keyword args for detect()."""
    return {k: cfg[k] for k in ('conf', 'device', 'half', 'iou', 'imgsz', 'max_det')}


def detect(model, bgr, **predict_kwargs):
    """Returns list of dicts: name, score, bbox (xyxy), mask (HxW bool, image resolution).
    predict_kwargs go straight to ultralytics predict(): conf, iou, imgsz, max_det, half, device."""
    res = model.predict(bgr, retina_masks=True, verbose=False, **predict_kwargs)[0]  # retina: masks at image res
    if not len(res.boxes):
        return []
    b = res.boxes
    return [{'name': res.names[int(c)], 'score': float(s), 'bbox': box.tolist(), 'mask': m > 0.5}
            for c, s, box, m in zip(b.cls.cpu(), b.conf.cpu(), b.xyxy.cpu().numpy(), res.masks.data.cpu().numpy())]
