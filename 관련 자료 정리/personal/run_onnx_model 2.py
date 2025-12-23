import onnxruntime as ort
import numpy as np
import cv2
import os
from typing import List, Tuple

"""
Run YOLO-style ONNX model and visualize detections.
- Works with common YOLOv5/v8 ONNX exports.
- Handles output shaped as (1, N, 85) or (1, 84, N) / (1, 85, N).
- Applies confidence filter + NMS, rescales boxes to original image, and saves 'result.jpg'.

Usage:
  python run_onnx_model.py --image test.jpg --model last.onnx --conf 0.25 --iou 0.45
"""

import argparse


def load_class_names(names_path: str, num_classes: int) -> List[str]:
    """Load class names from a .names or .txt file; fallback to generic labels."""
    if names_path and os.path.isfile(names_path):
        with open(names_path, 'r', encoding='utf-8') as f:
            names = [ln.strip() for ln in f.readlines() if ln.strip()]
        if names:
            return names
    return [f'class_{i}' for i in range(num_classes)]


def letterbox(im: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color=(114, 114, 114)):
    """Resize + pad image to fit new_shape while keeping aspect ratio (YOLO-style)."""
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def xywh2xyxy(x):
    # Convert [x, y, w, h] -> [x1, y1, x2, y2]
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    # Basic NMS; returns indices to keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def postprocess_yolo(raw: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45):
    """
    Convert raw YOLO outputs to (boxes, scores, class_ids).
    Accepts shapes like:
      - (1, N, 85)  -> [x, y, w, h, conf, cls0..clsK]
      - (1, 85, N) or (1, 84, N) -> transpose to (N, C)
      - (1, 84, N) (no objectness) -> treat max class prob as conf
    """
    if raw.ndim == 3:
        raw = raw.squeeze(0)  # (N, C) or (C, N)
    if raw.shape[0] < raw.shape[1] and raw.shape[0] in (84, 85):
        raw = raw.transpose(1, 0)  # (N, C)

    N, C = raw.shape  # C = 85 (x,y,w,h, conf, classes) or 84 (x,y,w,h, classes)

    has_obj = (C == 85)
    box = raw[:, :4]
    if has_obj:
        obj_conf = raw[:, 4:5]
        cls_scores = raw[:, 5:]
        cls_conf = cls_scores.max(axis=1, keepdims=True)
        cls_ids = cls_scores.argmax(axis=1)
        conf = (obj_conf * cls_conf).squeeze(1)
    else:
        # No objectness column (some exports):
        cls_scores = raw[:, 4:]
        cls_conf = cls_scores.max(axis=1)
        cls_ids = cls_scores.argmax(axis=1)
        conf = cls_conf

    # Filter by confidence
    mask = conf >= conf_thres
    box = box[mask]
    conf = conf[mask]
    cls_ids = cls_ids[mask]

    if box.size == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    # YOLO boxes are center-based (x,y,w,h)
    box_xyxy = xywh2xyxy(box)

    # NMS per class
    keep_all = []
    for c in np.unique(cls_ids):
        idxs = np.where(cls_ids == c)[0]
        kept = nms_boxes(box_xyxy[idxs], conf[idxs], iou_thres)
        keep_all.extend(idxs[k] for k in kept)

    keep_all = np.array(keep_all, dtype=int)
    return box_xyxy[keep_all], conf[keep_all], cls_ids[keep_all]


def scale_boxes_to_original(boxes: np.ndarray, ratio: float, pad: Tuple[float, float], orig_w: int, orig_h: int):
    # Reverse letterbox scaling back to original image coordinates
    if boxes.size == 0:
        return boxes
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, orig_w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, orig_h)
    return boxes


def draw_detections(im: np.ndarray, boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray, class_names: List[str]):
    for (x1, y1, x2, y2), sc, c in zip(boxes.astype(int), scores, cls_ids):
        label = f"{class_names[int(c)] if int(c) < len(class_names) else c}: {sc:.2f}"
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(im, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(im, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return im


def infer_image(model_path: str, image_path: str, img_size: int = 640, conf: float = 0.25, iou: float = 0.45, names: str = ""):
    # 1) Load image
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    orig_h, orig_w = orig.shape[:2]

    # 2) Preprocess (letterbox to keep ratio)
    padded, ratio, pad = letterbox(orig, (img_size, img_size))
    inp = padded[:, :, ::-1]  # BGR->RGB if needed
    inp = inp.transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # (1,3,H,W)

    # 3) Inference
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])  # add CUDA if available
    input_name = session.get_inputs()[0].name
    outs = session.run(None, {input_name: inp})

    # 4) Postprocess (assume single output tensor)
    raw = outs[0]
    boxes, scores, cls_ids = postprocess_yolo(raw, conf_thres=conf, iou_thres=iou)

    # 5) Rescale boxes back to original image
    boxes = scale_boxes_to_original(boxes, ratio, pad, orig_w, orig_h)

    # 6) Class names
    num_classes_guess = 80  # default to COCO
    if raw.ndim == 3:
        cdim = raw.shape[-1] if raw.shape[-1] > raw.shape[-2] else raw.shape[-2]
        if cdim in (84, 85):
            num_classes_guess = cdim - 5 if cdim == 85 else cdim - 4
    class_names = load_class_names(names, num_classes_guess)

    # 7) Draw & save
    vis = orig.copy()
    vis = draw_detections(vis, boxes, scores, cls_ids, class_names)
    out_path = 'result.jpg'
    cv2.imwrite(out_path, vis)

    print(f"Saved visualization -> {out_path}")
    print(f"Detections: {len(boxes)}")
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        print(f"  #{i+1}: cls={class_names[int(cls_ids[i])] if int(cls_ids[i]) < len(class_names) else int(cls_ids[i])}, conf={scores[i]:.3f}, box=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='last.onnx')
    parser.add_argument('--image', type=str, default='test.jpg')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--names', type=str, default='', help='path to class names file (.names/.txt)')
    args = parser.parse_args()

    infer_image(args.model, args.image, args.img_size, args.conf, args.iou, args.names)
