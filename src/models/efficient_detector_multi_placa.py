import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import numpy as np

from src.config import IMG_SIZE, LEARNING_RATE


def tf_maybe_print(*args, **kwargs):
    """tf.print solo en modo eager para evitar ops de string en grafos XLA."""
    if tf.executing_eagerly():
        tf.print(*args, **kwargs)

# ===============================
# CONFIGURACIÓN MULTI-PLACA REAL
# ===============================
GRID_SIZE = 13
NUM_ANCHORS = 3
NUM_CLASSES = 1
BBOX_DIM = 5

OUTPUT_DIM = NUM_ANCHORS * (BBOX_DIM + NUM_CLASSES)  # 18


# ==================================================
#                     CIOU
# ==================================================
def bbox_ciou(b1, b2):
    b1_x1 = b1[..., 0] - b1[..., 2] / 2
    b1_y1 = b1[..., 1] - b1[..., 3] / 2
    b1_x2 = b1[..., 0] + b1[..., 2] / 2
    b1_y2 = b1[..., 1] + b1[..., 3] / 2

    b2_x1 = b2[..., 0] - b2[..., 2] / 2
    b2_y1 = b2[..., 1] - b2[..., 3] / 2
    b2_x2 = b2[..., 0] + b2[..., 2] / 2
    b2_y2 = b2[..., 1] + b2[..., 3] / 2

    x1 = tf.maximum(b1_x1, b2_x1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x2 = tf.minimum(b1_x2, b2_x2)
    y2 = tf.minimum(b1_y2, b2_y2)

    inter = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union = area1 + area2 - inter + 1e-6
    iou = inter / union

    c_dist = tf.square(b1[..., 0] - b2[..., 0]) + tf.square(b1[..., 1] - b2[..., 1])

    cw = tf.maximum(b1_x2, b2_x2) - tf.minimum(b1_x1, b2_x1)
    ch = tf.maximum(b1_y2, b2_y2) - tf.minimum(b1_y1, b2_y1)
    c_diag = tf.square(cw) + tf.square(ch) + 1e-6

    v = (4 / (np.pi ** 2)) * tf.square(
        tf.atan(b1[..., 2] / (b1[..., 3] + 1e-6)) -
        tf.atan(b2[..., 2] / (b2[..., 3] + 1e-6))
    )
    alpha = v / (1 - iou + v + 1e-6)

    ciou = iou - (c_dist / c_diag) - alpha * v
    return ciou


# ==================================================
#                   LOSS COMPLETA
# ==================================================
def yolo_ciou_loss(y_true, y_pred):

    # Usar shapes dinámicas en tiempo de ejecución para evitar errores
    batch = tf.shape(y_pred)[0]
    grid_h = tf.shape(y_pred)[1]
    grid_w = tf.shape(y_pred)[2]
    channels = tf.shape(y_pred)[3]

    per_anchor = BBOX_DIM + NUM_CLASSES

    # Comprobar que el número de canales es divisible por lo esperado (por ancla)
    mod = tf.math.floormod(channels, per_anchor)
    # Evitar mensajes de debug que introduzcan ops no soportadas por XLA (StringFormat)
    with tf.control_dependencies([
        tf.debugging.assert_equal(mod, 0),
    ]):
        num_anchors_tensor = tf.math.floordiv(channels, per_anchor)
    # Imprimir shapes solo en modo eager (no crea ops en grafos compilados)
    tf_maybe_print("[yolo_loss] shapes -> batch:", batch, "grid:", grid_h, grid_w,
                   "channels:", channels, "anchors:", num_anchors_tensor)

    pred = tf.reshape(
        y_pred,
        (batch, grid_h, grid_w, num_anchors_tensor, per_anchor)
    )
    true = tf.reshape(
        y_true,
        (batch, grid_h, grid_w, num_anchors_tensor, per_anchor)
    )

    pred_conf = pred[..., 0]
    pred_bbox = pred[..., 1:5]
    pred_cls  = pred[..., 5:6]

    true_conf = true[..., 0]
    true_bbox = true[..., 1:5]
    true_cls  = true[..., 5:6]

    # IoU para asignación
    ious = bbox_ciou(pred_bbox, true_bbox)
    best_anchor = tf.argmax(ious, axis=-1)  # (B,13,13)

    # máscara objeto
    obj_mask = true_conf

    # máscara del anchor asignado
    best_mask = tf.one_hot(best_anchor, NUM_ANCHORS)
    best_mask = tf.cast(best_mask, tf.float32)

    # --------------- 1) LOSS LOCALIZACIÓN ----------------
    ciou = bbox_ciou(pred_bbox, true_bbox)
    ciou_term = best_mask * (1 - ciou)  # [B,13,13,3]

    loc_loss = obj_mask * tf.reduce_sum(ciou_term, axis=-1, keepdims=True)

    # ---------------- 2) LOSS CONF -----------------------
    conf_loss = tf.keras.losses.binary_crossentropy(true_conf, pred_conf)

    # ---------------- 3) LOSS CLASE ----------------------
    class_loss = obj_mask * tf.keras.losses.binary_crossentropy(true_cls, pred_cls)

    total = (
        tf.reduce_sum(loc_loss) +
        tf.reduce_sum(conf_loss) +
        tf.reduce_sum(class_loss)
    )

    return total


# ==================================================
#                  MODELO COMPLETO
# ==================================================
def build_multishot_detector_from_scratch(img_size=IMG_SIZE, learning_rate=LEARNING_RATE):

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = layers.Rescaling(1./255)(inputs)

    base = EfficientNetB0(include_top=False, weights=None, input_tensor=x)
    base.trainable = True

    x = layers.Resizing(GRID_SIZE, GRID_SIZE)(base.output)

    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Conv2D(
        OUTPUT_DIM,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        name="detection_output"
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=yolo_ciou_loss,
    )

    return model


if __name__ == "__main__":
    m = build_multishot_detector_from_scratch()
    x = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
    out = m.predict(x)
    print("Salida:", out.shape)
    print("Modelo EfficientDet Multi-Placa creado correctamente.")
