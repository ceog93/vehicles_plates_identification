# archivo: src/utils/mpd_utils.py
# Funciones de ayuda: resize+pad, conversión de cajas y NMS en numpy
# Comentarios detallados en español.

import cv2
import numpy as np

def resize_pad(img, size):
    """
    Redimensiona manteniendo aspecto y aplica padding centrado.
    Color de padding: (114,114,114) (gris neutro).
    Retorna: img_padded, scale, top, left
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h = size - nh
    pad_w = size - nw
    top = pad_h // 2
    left = pad_w // 2
    bottom = pad_h - top
    right = pad_w - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img_padded, scale, top, left

def xywhn_to_xyxy(boxes, img_w, img_h):
    """
    Convierte cajas normalizadas [cx,cy,w,h] (valores 0..1) a [x1,y1,x2,y2] en pixeles.
    boxes: array Nx4 o Nx5 (si tiene obj en columna 4 se ignora)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0,4))
    bx = boxes[:,0]
    by = boxes[:,1]
    bw = boxes[:,2]
    bh = boxes[:,3]
    cx = bx * img_w
    cy = by * img_h
    w = bw * img_w
    h = bh * img_h
    x1 = cx - w/2; y1 = cy - h/2
    x2 = cx + w/2; y2 = cy + h/2
    return np.stack([x1,y1,x2,y2], axis=1)

def iou_numpy(box1, boxes):
    """
    IoU entre box1 (1x4) y boxes (Nx4) formato [x1,y1,x2,y2]
    """
    x1 = np.maximum(box1[:,0], boxes[:,0])
    y1 = np.maximum(box1[:,1], boxes[:,1])
    x2 = np.minimum(box1[:,2], boxes[:,2])
    y2 = np.minimum(box1[:,3], boxes[:,3])
    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = (box1[:,2]-box1[:,0])*(box1[:,3]-box1[:,1])
    area2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    union = area1 + area2 - inter
    return inter / (union + 1e-7)

def nms_numpy(boxes, scores, iou_thresh=0.45, score_thresh=0.25, max_detections=200):
    """
    Non-max suppression simple en Numpy.
    Devuelve índices de boxes seleccionadas.
    """
    # filtrar por score
    if boxes.shape[0] == 0:
        return np.array([], dtype=int)
    mask = scores >= score_thresh
    if not np.any(mask):
        return np.array([], dtype=int)
    boxes = boxes[mask]
    scores = scores[mask]
    idxs = np.argsort(scores)[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < max_detections:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou_numpy(boxes[i:i+1], boxes[idxs[1:]])
        rem = np.where(ious[0] <= iou_thresh)[0]
        idxs = idxs[rem + 1]
    # mapear indices de regreso a arreglo original con mask
    orig_idx = np.where(mask)[0]
    keep_orig = orig_idx[keep]
    return keep_orig
