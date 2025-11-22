# src/models/efficient_detector_multi_placa.py

import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.applications import EfficientNetB0  # type: ignore
import numpy as np

# Se asume que IMG_SIZE = (W, H)
from src.config import IMG_SIZE, LEARNING_RATE

# ============================================
# PARÁMETROS CRÍTICOS para el modelo Multiplaca
# ============================================
GRID_SIZE = 7       # La imagen se reduce a una cuadrícula S x S (ej. 7x7)
BBOX_ANCHORS = 3    # Número de 'bounding boxes' (o anclas) a predecir por celda
NUM_CLASSES = 1     # Solo la clase 'Placa'
OUTPUT_DIM = (5 * BBOX_ANCHORS) + NUM_CLASSES # 16 = (5*3) + 1

def yolo_like_loss(y_true, y_pred):
    """
    Función de pérdida personalizada (Custom Loss) tipo YOLO/SSD simplificada.
    Combina la pérdida de localización, la pérdida de confianza y la pérdida de clasificación.
    
    y_true shape: (Batch, GRID_SIZE, GRID_SIZE, OUTPUT_DIM)
    y_pred shape: (Batch, GRID_SIZE, GRID_SIZE, OUTPUT_DIM)
    """
    
    # Parámetros de penalización (se ajustan durante el entrenamiento)
    lambda_coord = 5.0
    lambda_noobj = 0.5
    
    # 1. Reformatear y_true y y_pred para facilitar el acceso a las predicciones
    # El modelo predice (BBOX_ANCHORS) cajas, cada una con [confianza, cx, cy, w, h]
    
    # Dividimos la salida en Confianza (C), Localización (L) y Clases
    
    # Cajas predichas (cx, cy, w, h)
    pred_boxes = tf.reshape(y_pred[..., 1:5], (-1, GRID_SIZE, GRID_SIZE, BBOX_ANCHORS, 4))
    # Confianza predicha (P(Obj) * IOU)
    pred_conf = tf.reshape(y_pred[..., 0:1], (-1, GRID_SIZE, GRID_SIZE, BBOX_ANCHORS, 1))

    # Cajas verdaderas (cx, cy, w, h)
    true_boxes = tf.reshape(y_true[..., 1:5], (-1, GRID_SIZE, GRID_SIZE, BBOX_ANCHORS, 4))
    # Máscara de presencia de objeto (1 si hay objeto, 0 si no)
    response_mask = y_true[..., 0:1] # La primera dimensión de y_true es la máscara (1 o 0)

    # ==================================
    # 2. Pérdida de Confianza (Confidence Loss)
    # ==================================
    
    # Pérdida para celdas con objeto (Binary Crossentropy)
    obj_loss = response_mask * tf.square(pred_conf - y_true[..., 0:1])
    
    # Pérdida para celdas sin objeto (Penalización más baja: lambda_noobj)
    noobj_loss = lambda_noobj * (1 - response_mask) * tf.square(pred_conf - y_true[..., 0:1])

    confidence_loss = tf.reduce_sum(obj_loss + noobj_loss)
    
    # ==================================
    # 3. Pérdida de Localización (Localization Loss)
    # ==================================
    
    # Solo aplicamos la pérdida de localización donde hay un objeto (response_mask == 1)
    
    # Pérdida para (x, y) (Distancia euclidiana cuadrada)
    coord_xy_loss = response_mask * tf.square(true_boxes[..., 0:2] - pred_boxes[..., 0:2])
    
    # Pérdida para (w, h) (Raíz cuadrada para estabilizar gradientes)
    # Evita log(0)
    pred_wh_sqrt = tf.sign(pred_boxes[..., 2:4]) * tf.sqrt(tf.abs(pred_boxes[..., 2:4]) + 1e-6)
    true_wh_sqrt = tf.sign(true_boxes[..., 2:4]) * tf.sqrt(tf.abs(true_boxes[..., 2:4]) + 1e-6)
    
    coord_wh_loss = response_mask * tf.square(true_wh_sqrt - pred_wh_sqrt)
    
    localization_loss = lambda_coord * tf.reduce_sum(coord_xy_loss + coord_wh_loss)
    
    # ==================================
    # 4. Pérdida Total
    # ==================================
    total_loss = localization_loss + confidence_loss 
    
    return total_loss


def build_multishot_detector_from_scratch(img_size=IMG_SIZE, learning_rate=LEARNING_RATE):
    """
    Detector Multiplaca basado en Grid (SSD simplificado)
    ENTRENADO 100% DESDE CERO (weights=None).
    """

    # ============================================
    # 1. Backbone EfficientNetB0 (SIN PREENTRENAR)
    # ============================================
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    
    # Preprocesamiento simple (escala 0-1)
    x = layers.Rescaling(1./255)(inputs) 

    base = EfficientNetB0(
        include_top=False,
        weights=None,                   # ! Restricción: entrenamiento desde cero
        input_tensor=x                  # Conectar al input preprocesado
    )
    base.trainable = True

    # ============================================
    # 2. Cabeza de Detección (Detection Head)
    # ============================================
    
    x = base.output 

    # Capas convolucionales para refinar las características del backbone
    # Es crucial que la salida espacial de este punto sea S x S (7x7)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x) # Recomendado al entrenar desde cero
    x = layers.Dropout(0.3)(x)

    # Última capa Conv: La salida final debe ser (S, S, OUTPUT_DIM) -> (7, 7, 16)
    outputs = layers.Conv2D(
        OUTPUT_DIM, 
        kernel_size=1, 
        padding='same', 
        name="detection_output",
        activation='sigmoid' # Usar Sigmoid en la salida para las predicciones normalizadas [0, 1]
    )(x) 
    
    model = models.Model(inputs, outputs)

    # ============================================
    # 3. Compilación del modelo
    # ============================================
    
    # Aquí usamos la función de pérdida personalizada definida arriba
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=yolo_like_loss, 
        metrics=['accuracy'] # Métricas simples. Deben añadirse métricas IOU personalizadas.
    )

    return model

if __name__ == "__main__":
    # Test rápido de la forma de salida (debe ser (1, 7, 7, 16))
    m = build_multishot_detector_from_scratch()
    x = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
    out = m.predict(x)
    print("Modelo Multiplaca cargado — salida shape:", out.shape)
    
    # Verificar la forma esperada
    expected_shape = (1, GRID_SIZE, GRID_SIZE, OUTPUT_DIM)
    assert out.shape == expected_shape, f"La forma de salida es {out.shape}, se esperaba {expected_shape}"