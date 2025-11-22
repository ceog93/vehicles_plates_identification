# src/models/efficient_detector.py
# Modelo EfficientNetB0 desde cero para detección de placas vehiculares
# Compatible al 100% con src/train.py

import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.applications import EfficientNetB0 # type: ignore

from src.config import IMG_SIZE, LEARNING_RATE


def build_efficient_detector(img_size=IMG_SIZE, learning_rate=LEARNING_RATE):
    """
    Detector de placas vehiculares usando EfficientNetB0
    ENTRENADO DESDE CERO (weights=None), cumpliendo la restricción del usuario.
    100% compatible con el pipeline actual.
    """

    # ============================================
    # 1. Backbone EfficientNetB0 (SIN PREENTRENAR)
    # ============================================
    base = EfficientNetB0(
        include_top=False,
        weights=None,                                # entrenamiento desde cero
        input_shape=(img_size[0], img_size[1], 3)
    )

    # ENTRENAMIENTO COMPLETO DESDE CERO
    base.trainable = True

    # ============================================
    # 2. Construcción del modelo
    # ============================================
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = base(inputs)

    # Pooling global + regularización equivalente
    x = layers.GlobalAveragePooling2D()(x)

    # Capas densas profundas
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)          # sustituto de dropout del backbone

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Regresión bounding box (xmin, ymin, xmax, ymax)
    outputs = layers.Dense(
        4,
        activation="sigmoid",
        name="bounding_box_output"
    )(x)

    model = models.Model(inputs, outputs)

    # ============================================
    # 3. Compilación del modelo
    # ============================================
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
        metrics=["mae", "mse"]
    )

    return model
