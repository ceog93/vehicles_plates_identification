# src/losses/mpd_loss.py
import tensorflow as tf
'''
Funciones de pérdida personalizadas para la detección de placas vehiculares.
Incluye pérdida de regresión Huber para bounding boxes.
Se usa en el modelo de detección (efficient_detector.py, mobile_net_detector.py).
'''
def regression_huber_loss(y_true, y_pred, delta=1.0):
    return tf.keras.losses.Huber(delta=delta)(y_true, y_pred)

# Ejemplo de wrapper si se desea incluir objectness más adelante:
def combined_loss(y_true_boxes, y_pred_boxes, lambda_box=1.0):
    # Por ahora identidad a Huber:
    return regression_huber_loss(y_true_boxes, y_pred_boxes) * lambda_box
