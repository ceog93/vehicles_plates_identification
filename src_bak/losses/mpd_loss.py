# archivo: src/losses/mpd_loss.py
# Implementación de la pérdida compuesta (CIoU + objectness + no-object)

import tensorflow as tf
import math

def ciou_tf(boxes1, boxes2):
    """
    CIoU entre boxes1 y boxes2.
    boxes en formato [cx, cy, w, h] (normalizados o en la misma escala).
    """
    x1, y1, w1, h1 = tf.split(boxes1, 4, axis=-1)
    x2, y2, w2, h2 = tf.split(boxes2, 4, axis=-1)

    x1_min = x1 - w1/2; y1_min = y1 - h1/2
    x1_max = x1 + w1/2; y1_max = y1 + h1/2
    x2_min = x2 - w2/2; y2_min = y2 - h2/2
    x2_max = x2 + w2/2; y2_max = y2 + h2/2

    inter_xmin = tf.maximum(x1_min, x2_min)
    inter_ymin = tf.maximum(y1_min, y2_min)
    inter_xmax = tf.minimum(x1_max, x2_max)
    inter_ymax = tf.minimum(y1_max, y2_max)

    inter_w = tf.maximum(inter_xmax - inter_xmin, 0.0)
    inter_h = tf.maximum(inter_ymax - inter_ymin, 0.0)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    iou = inter_area / (union + 1e-7)

    enc_xmin = tf.minimum(x1_min, x2_min)
    enc_ymin = tf.minimum(y1_min, y2_min)
    enc_xmax = tf.maximum(x1_max, x2_max)
    enc_ymax = tf.maximum(y1_max, y2_max)
    enc_w = enc_xmax - enc_xmin
    enc_h = enc_ymax - enc_ymin
    c2 = enc_w**2 + enc_h**2

    rho2 = (x2 - x1)**2 + (y2 - y1)**2
    v = (4 / (math.pi**2)) * tf.square(tf.atan(w2 / (h2 + 1e-7)) - tf.atan(w1 / (h1 + 1e-7)))
    ciou = iou - (rho2 / (c2 + 1e-7)) - (v / (1 - iou + 1e-7))
    return ciou

class PerdidaMPD:
    """
    Wrapper de pérdida para MPD-Net.
    anchors: lista de anchors por escala en pixeles (relativos al input_size de entrenamiento)
    y_true_list: lista de tensores por escala [B,S,S,A,5]
    y_pred_list: lista de predicciones por escala [B,S,S,A*5] (antes de reshape)
    """
    def __init__(self, anchors, lambda_box=5.0, lambda_obj=1.0, lambda_noobj=0.5):
        self.anchors = anchors
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj

    def __call__(self, y_true_list, y_pred_list):
        total_loss = 0.0
        # Iterar por cada escala
        for y_true, y_pred, anchors in zip(y_true_list, y_pred_list, self.anchors):
            # Reshape pred a (B,S,S,A,5)
            pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], [-1,5]], axis=0))
            pred_box = pred[..., :4]   # tx,ty,tw,th codificados
            pred_obj = tf.sigmoid(pred[..., 4])

            true_box = y_true[..., :4]
            true_obj = y_true[..., 4]

            # Pérdida de caja: 1 - CIoU (solo para anchors con objeto)
            ciou = ciou_tf(pred_box, true_box)
            loss_box = tf.reduce_sum((1.0 - ciou) * true_obj)

            # Pérdida de obj (BCE)
            loss_obj = tf.reduce_sum(tf.keras.losses.binary_crossentropy(true_obj, pred_obj))

            # Pérdida de no-obj: penalizar falsos positivos
            mask_noobj = 1.0 - true_obj
            loss_noobj = tf.reduce_sum(tf.keras.losses.binary_crossentropy(tf.zeros_like(pred_obj), pred_obj) * mask_noobj)

            total_loss += self.lambda_box * loss_box + self.lambda_obj * loss_obj + self.lambda_noobj * loss_noobj
        return total_loss
