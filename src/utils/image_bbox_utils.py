"""
src/utils/image_bbox_utils.py

Utilidades para el preprocesamiento de imágenes y el postprocesamiento de cajas delimitadoras.
Este módulo fue creado para reemplazar el antiguo nombre `mpd_utils.py` (mpd = Multi Plate Detector).
Proporciona funciones de ayuda utilizadas durante la inferencia: `resize_pad`, `denormalize_box` y `nms_numpy`.
"""
import cv2
import numpy as np
import tensorflow as tf

def resize_pad(img, size):
    """
    Redimensiona la imagen manteniendo la relación de aspecto y aplica relleno (letterbox).
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
    Convierte una caja normalizada [xmin,ymin,xmax,ymax] en el rango 0..1 a coordenadas de píxeles.
    Retorna enteros [x1,y1,x2,y2].
    """
    xmin = int(box[0] * img_w)
    ymin = int(box[1] * img_h)
    xmax = int(box[2] * img_w)
    ymax = int(box[3] * img_h)
    return [xmin, ymin, xmax, ymax]


def nms_numpy(boxes, scores, iou_thresh=0.45, score_thresh=0.25):
    """
    Wrapper para la supresión de no máximos (NMS) usando el backend de TensorFlow; retorna índices de numpy.
    boxes: Lista de cajas en formato Nx4 [x1,y1,x2,y2]
    scores: Lista de puntuaciones de confianza N
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    boxes_tf = tf.convert_to_tensor(boxes, dtype=tf.float32)
    scores_tf = tf.convert_to_tensor(scores, dtype=tf.float32)
    selected = tf.image.non_max_suppression(boxes_tf, scores_tf, max_output_size=200, iou_threshold=iou_thresh, score_threshold=score_thresh)
    selected_np = selected.numpy().astype(int)
    return selected_np
