"""YOLOE-seg with visual prompts: one example image per object -> class embeddings.

Each object is registered from a single reference image (and a box around the object in it).
The per-object visual prompt embeddings (VPE) are stacked and set as the model's classes, so a
single forward pass scores every registered object. The same model can then be exported to a
TensorRT engine with the embeddings baked in (see scripts/yoloe_trt.py).
"""
import json
import os

import numpy as np


def load_prompt(image_path):
    """Returns (bgr image, bbox xyxy). bbox comes from <name>.json next to the image, else full image."""
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError('visual prompt not found: %s' % image_path)
    meta = os.path.splitext(image_path)[0] + '.json'
    if os.path.isfile(meta):
        with open(meta) as f:
            bbox = [float(v) for v in json.load(f)['bbox']]
    else:
        h, w = img.shape[:2]
        bbox = [0.0, 0.0, float(w - 1), float(h - 1)]
    return img, bbox


def _visual_pe(model, image, bbox, device):
    """Visual prompt embedding (1, 1, D) for one object box in one reference image."""
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    vp = dict(bboxes=np.array([bbox], dtype=np.float32), cls=np.array([0]))
    try:
        # ultralytics >= 8.3.1xx: predictor keeps the embedding when return_vpe=True
        model.predict(image, visual_prompts=vp, predictor=YOLOEVPSegPredictor,
                      return_vpe=True, device=device, verbose=False)
        vpe = model.predictor.vpe
    except Exception:
        # fallback: refer_image path sets the embedding as the model's class embedding
        model.predict(image, refer_image=image, visual_prompts=vp, predictor=YOLOEVPSegPredictor,
                      device=device, verbose=False)
        vpe = model.model.pe
    model.predictor = None
    return vpe.detach().clone()


def build_visual_prompt_model(weights, prompts, device='cuda:0'):
    """prompts: list of (name, bgr_image, bbox). Returns a YOLOE model whose classes are the names."""
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLOE

    model = YOLOE(weights)
    names, vpes = [], []
    for name, img, bbox in prompts:
        vpes.append(_visual_pe(model, img, bbox, device).to('cpu').reshape(1, 1, -1))
        names.append(name)
    vpe = F.normalize(torch.cat(vpes, dim=1), dim=-1, p=2)
    model.set_classes(names, vpe)
    model.predictor = None
    return model


def build_text_prompt_model(weights, texts, device='cuda:0'):
    """texts: list of open-vocabulary class-name strings (no reference image needed). Returns a
    YOLOE model whose classes are those texts -- same detect()-compatible model as
    build_visual_prompt_model(), just prompted by text instead of an example image."""
    from ultralytics import YOLOE

    model = YOLOE(weights)
    model.set_classes(texts, model.get_text_pe(texts))
    model.predictor = None
    return model


def load_detector(weights, engine, names, prompt_path_fn, device='cuda:0'):
    """TensorRT engine (classes baked in at export) if given, else build from the prompt images."""
    if engine:
        from ultralytics import YOLO
        return YOLO(engine, task='segment')
    prompts = []
    for n in names:
        img, bbox = load_prompt(prompt_path_fn(n))
        prompts.append((n, img, bbox))
    return build_visual_prompt_model(weights, prompts, device)


def detect(model, bgr, conf=0.05, device='cuda:0', half=False):
    """Returns list of dicts: name, score, bbox (xyxy), mask (HxW bool, image resolution)."""
    res = model.predict(bgr, conf=conf, retina_masks=True, device=device, half=half, verbose=False)[0]
    out = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)
    masks = res.masks.data.cpu().numpy() > 0.5 if res.masks is not None else None
    h, w = bgr.shape[:2]
    for i in range(len(boxes)):
        m = masks[i] if masks is not None else None
        if m is not None and m.shape != (h, w):
            import cv2
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        out.append({'name': res.names[int(classes[i])], 'score': float(scores[i]),
                    'bbox': [float(v) for v in boxes[i]], 'mask': m})
    return out
