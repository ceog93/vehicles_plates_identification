# src/utils/mpd_utils.py
import cv2
import numpy as np
import tensorflow as tf

def resize_pad(img, size):
    """
    Se usa en el inferencia para:
        Mantener aspecto original
        Evitar deformar la placa
        Escalar correctamente
        Recuperar luego coordenadas con scale, top, left
    Resize manteniendo aspecto y apply padding (letterbox).
    img: numpy array HxWx3 (RGB)
    size: int (ej. 416 o 640)
    Retorna: img_padded (size,size,3), scale, top, left
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

def denormalize_box(box, img_w, img_h):
    """
    Se usa después de predecir para convertir las cajas normalizadas 0–1 a coordenadas de píxeles.
    box: [xmin, ymin, xmax, ymax] normalizados 0..1
    devuelve coords en píxeles int (x1,y1,x2,y2)
    """
    xmin = int(box[0] * img_w)
    ymin = int(box[1] * img_h)
    xmax = int(box[2] * img_w)
    ymax = int(box[3] * img_h)
    return [xmin, ymin, xmax, ymax]

def nms_numpy(boxes, scores, iou_thresh=0.45, score_thresh=0.25):
    """
    Se usa en la fase de post-procesamiento, después de obtener todas las cajas y scores.
    NMS simple usando TF backend (pero devuelve indices numpy)
    boxes: Nx4 en formato [x1,y1,x2,y2] (píxeles o floats)
    scores: N
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    boxes_tf = tf.convert_to_tensor(boxes, dtype=tf.float32)
    scores_tf = tf.convert_to_tensor(scores, dtype=tf.float32)
    selected = tf.image.non_max_suppression(boxes_tf, scores_tf, max_output_size=200, iou_threshold=iou_thresh, score_threshold=score_thresh)
    selected_np = selected.numpy().astype(int)
    return selected_np
