#src/models/efficient_detector_multi_placa.py

"""
Módulo del detector multi-placa basado en EfficientNetB0.

Este archivo define:
- Constantes de la arquitectura (tamaño de grid, número de anclas, dimensiones de caja).
- La función `bbox_ciou` que calcula la métrica CIoU entre dos cajas (útil como término de localización).
- La función de pérdida `yolo_ciou_loss` que combina CIoU (localización),
    binary crossentropy para la confianza y la clase, y que está diseñada para
    trabajar con salidas en formato YOLO-like: (batch, grid_h, grid_w, anchors*(5+num_classes)).
- `build_multishot_detector_from_scratch`: construye y devuelve el modelo Keras.

Notas de diseño:
- Evitamos ops que generen cadenas de texto dentro del grafo (p.ej. mensajes de
    `tf.print` o `message=` en asserts) porque cuando TensorFlow intenta compilar
    el grafo con XLA para GPU algunas operaciones de formato de strings no están
    disponibles y provocan errores de compilación. Para depuración mantenemos
    impresiones condicionadas a `tf.executing_eagerly()`.
- Las dimensiones de la salida están sincronizadas con el generador de datos
    `ImageSequence` en `src/utils/image_seguence.py` importando las mismas
    constantes (GRID_SIZE, NUM_ANCHORS, etc.) cuando sea necesario.

Todo el texto de los docstrings y comentarios está en español para facilitar
la lectura y mantenimiento por el equipo.
"""

import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.applications import EfficientNetB0 # type: ignore
import numpy as np

from src.config import IMG_SIZE, LEARNING_RATE


def tf_maybe_print(*args, **kwargs):
        """Imprime con `tf.print` solo cuando TensorFlow está en modo eager.

        Razón: `tf.print` crea ops relacionadas con strings cuando se inserta en el
        grafo. Si el grafo se compila con XLA (GPU), dichas ops pueden no estar
        soportadas y provocan errores de compilación (StringFormat). Con este
        helper mantenemos la información de depuración disponible en ejecuciones
        interactivas sin afectar a la ejecución/compilación en GPU.
        """
        if tf.executing_eagerly():
                tf.print(*args, **kwargs)

# ===============================
# CONFIGURACIÓN MULTI-PLACA REAL
# ===============================
# ======= Parámetros de salida y anchoring =======
# `GRID_SIZE` define la resolución de la cuadrícula final (p. ej. 13 en YOLO)
GRID_SIZE = 13
# Número de anclas por celda. En este proyecto usamos 3 anclas.
NUM_ANCHORS = 3
# Número de clases a predecir. Aquí se entrena para detectar la presencia de una placa.
NUM_CLASSES = 1
# Dimensión por bbox: [conf, cx, cy, w, h] = 5
BBOX_DIM = 5

# OUTPUT_DIM es el número de canales de la capa de salida del detector:
# anchors * (bbox_dim + num_classes). Para los valores actuales: 3 * (5 + 1) = 18
OUTPUT_DIM = NUM_ANCHORS * (BBOX_DIM + NUM_CLASSES)  # 18


# ==================================================
#                     CIOU
# ==================================================
def bbox_ciou(b1, b2):
    # Convertir desde formato (cx, cy, w, h) a (x1, y1, x2, y2)
    b1_x1 = b1[..., 0] - b1[..., 2] / 2
    b1_y1 = b1[..., 1] - b1[..., 3] / 2
    b1_x2 = b1[..., 0] + b1[..., 2] / 2
    b1_y2 = b1[..., 1] + b1[..., 3] / 2

    b2_x1 = b2[..., 0] - b2[..., 2] / 2
    b2_y1 = b2[..., 1] - b2[..., 3] / 2
    b2_x2 = b2[..., 0] + b2[..., 2] / 2
    b2_y2 = b2[..., 1] + b2[..., 3] / 2

    # Coordenadas de la intersección
    x1 = tf.maximum(b1_x1, b2_x1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x2 = tf.minimum(b1_x2, b2_x2)
    y2 = tf.minimum(b1_y2, b2_y2)

    # Área de intersección (clamp a 0 para evitar negativas)
    inter = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)

    # Áreas individuales
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # Unión + epsilon para estabilidad numérica
    union = area1 + area2 - inter + 1e-6
    iou = inter / union

    # Distancia entre centros
    c_dist = tf.square(b1[..., 0] - b2[..., 0]) + tf.square(b1[..., 1] - b2[..., 1])

    # Caja englobante que contiene ambas cajas
    cw = tf.maximum(b1_x2, b2_x2) - tf.minimum(b1_x1, b2_x1)
    ch = tf.maximum(b1_y2, b2_y2) - tf.minimum(b1_y1, b2_y1)
    c_diag = tf.square(cw) + tf.square(ch) + 1e-6

    # Componente para la diferencia de relación de aspecto
    v = (4 / (np.pi ** 2)) * tf.square(
        tf.atan(b1[..., 2] / (b1[..., 3] + 1e-6)) -
        tf.atan(b2[..., 2] / (b2[..., 3] + 1e-6))
    )
    alpha = v / (1 - iou + v + 1e-6)

    # CIoU: IoU penalizado por distancia de centros y discrepancia de aspecto
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
    # Usar el número de anclas calculado dinámicamente (`num_anchors_tensor`) para
    # evitar discrepancias entre la forma real de salida y la constante del módulo.
    best_mask = tf.one_hot(best_anchor, tf.cast(num_anchors_tensor, tf.int32))
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

        """Construye y devuelve el modelo de detección.

        Flujo del modelo:
        - `inputs` acepta imágenes con la forma `img_size` definida en `src.config`.
        - Normalizamos los píxeles con `Rescaling(1./255)`.
        - Usamos `EfficientNetB0` como *backbone* sin la cabeza de clasificación
            (`include_top=False`) y sin pesos pre-entrenados en este proyecto
            (weights=None). Esto facilita entrenar desde cero con tu dataset.
        - Reducimos la salida del backbone a la resolución de la cuadrícula final
            usando `Resizing(GRID_SIZE, GRID_SIZE)` para obtener un mapa espacial
            compatible con el formato YOLO-like.
        - Añadimos un par de capas convolucionales y BatchNorm para procesar
            características antes de la capa final de predicción.
        - La capa final es una `Conv2D` con `OUTPUT_DIM` filtros, activación
            `sigmoid` y sin reshape explícito: la salida tiene forma
            `(batch, GRID_SIZE, GRID_SIZE, OUTPUT_DIM)`.

        La función compila el modelo con Adam y la función de pérdida `yolo_ciou_loss`.
        Puedes ajustar `learning_rate` pasando un valor distinto al llamar a esta
        función.

        Retorna:
                model (tf.keras.Model): modelo compilado listo para entrenar.
        """

        # Entrada: tamaño de imagen definido por la configuración
        inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
        # Normalización simple de píxeles (0..1)
        x = layers.Rescaling(1./255)(inputs)

        # Backbone EfficientNetB0 sin cabeza; `input_tensor=x` enlaza la normalización
        base = EfficientNetB0(include_top=False, weights=None, input_tensor=x)
        base.trainable = True

        # Ajustar el mapa de características a la resolución de la cuadrícula
        x = layers.Resizing(GRID_SIZE, GRID_SIZE)(base.output)

        # Capas intermedias para procesar características antes de la salida
        x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        # Capa final: por cada celda se predice OUTPUT_DIM valores
        outputs = layers.Conv2D(
                OUTPUT_DIM,
                kernel_size=1,
                padding="same",
                activation="sigmoid",
                name="detection_output"
        )(x)

        model = models.Model(inputs, outputs)

        # Compilamos el modelo con la loss definida más arriba. No aplicamos
        # métricas aquí (puedes añadir métricas personalizadas si las necesitas).
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
