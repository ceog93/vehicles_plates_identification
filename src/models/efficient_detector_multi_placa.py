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
    
    y_true shape: (Batch, GRID_SIZE, GRID_SIZE, OUTPUT_DIM)
    y_pred shape: (Batch, GRID_SIZE, GRID_SIZE, OUTPUT_DIM)
    """
    
    # Parámetros de penalización (GRID_SIZE=7, BBOX_ANCHORS=3)
    lambda_coord = 5.0  # Peso para la localización
    lambda_noobj = 0.5  # Peso para celdas sin objeto
    
    # response_mask: 1.0 si hay objeto en esa celda, 0.0 si no. (Batch, S, S, 1)
    # response_mask es el canal 0 de y_true
    response_mask = y_true[..., 0:1] 
    
    # =====================================================================
    # 1. Pérdida de Confianza (Confidence Loss) - Calculada sobre las 3 Anclas
    # =====================================================================
    
    # Separar las confianzas predichas para las 3 anclas (Canales 0, 5, 10)
    pred_conf_list = [y_pred[..., i * 5 : i * 5 + 1] for i in range(BBOX_ANCHORS)]
    pred_all_conf = tf.concat(pred_conf_list, axis=-1) # Shape (Batch, S, S, 3)
    
    # Crear el tensor de confianza Ground Truth (1.0 en Anchor 1, 0.0 en Anchor 2 y 3)
    true_conf_1 = y_true[..., 0:1]
    true_conf_rest = tf.zeros_like(true_conf_1)
    
    # true_all_conf: Shape (Batch, S, S, 3) -> [C1_GT, 0, 0]
    true_all_conf = tf.concat([true_conf_1, true_conf_rest, true_conf_rest], axis=-1) 

    # Máscara de respuesta expandida: 1.0 en Anchor 1 si hay objeto. 0.0 en Anchor 2 y 3.
    response_mask_expanded = tf.concat([response_mask, tf.zeros_like(response_mask), tf.zeros_like(response_mask)], axis=-1) # Shape (Batch, S, S, 3)

    # a) Pérdida para celdas CON objeto (Solo Anchor 1 tiene 1.0 en response_mask_expanded)
    obj_loss = response_mask_expanded * tf.square(pred_all_conf - true_all_conf)
    
    # b) Pérdida para celdas SIN objeto
    noobj_mask = 1.0 - response_mask_expanded
    
    # Aplicamos lambda_noobj al error en todas las anclas donde no hay objeto.
    noobj_loss = lambda_noobj * noobj_mask * tf.square(pred_all_conf - true_all_conf)

    confidence_loss = tf.reduce_sum(obj_loss + noobj_loss)
    
    # =====================================================================
    # 2. Pérdida de Localización (Localization Loss) - Solo para la Ancla 1
    # =====================================================================
    
    # Los boxes predichos y verdaderos para Anchor 1 están en los canales 1:5 (cx1, cy1, w1, h1)
    pred_boxes = y_pred[..., 1:5] # (Batch, S, S, 4)
    true_boxes = y_true[..., 1:5] # (Batch, S, S, 4)
    
    # Máscara para localización: replicar response_mask 4 veces (para cx, cy, w, h)
    response_mask_boxes = tf.concat([response_mask] * 4, axis=-1) # (Batch, S, S, 4)
    
    # Pérdida para (cx, cy)
    coord_xy_loss = response_mask_boxes[..., 0:2] * tf.square(true_boxes[..., 0:2] - pred_boxes[..., 0:2])
    
    # Pérdida para (w, h) (Usando raíz cuadrada para estabilización)
    pred_wh_sqrt = tf.sign(pred_boxes[..., 2:4]) * tf.sqrt(tf.abs(pred_boxes[..., 2:4]) + 1e-6)
    true_wh_sqrt = tf.sign(true_boxes[..., 2:4]) * tf.sqrt(tf.abs(true_boxes[..., 2:4]) + 1e-6)
    
    coord_wh_loss = response_mask_boxes[..., 2:4] * tf.square(true_wh_sqrt - pred_wh_sqrt)
    
    # Multiplicar por lambda_coord
    localization_loss = lambda_coord * tf.reduce_sum(coord_xy_loss + coord_wh_loss)
    
    # =====================================================================
    # 3. Pérdida de Clasificación (Classification Loss) - Último Canal (15)
    # =====================================================================
    
    # El canal de clase está en la última posición.
    # 🛑 FIX: Usamos [-1:] para mantener la dimensión de canal 1: [B, S, S, 1]
    true_class = y_true[..., -1:] 
    pred_class = y_pred[..., -1:]
    
    # response_mask tiene forma [B, S, S, 1], ahora true/pred_class también.
    class_loss = response_mask * tf.square(true_class - pred_class)
    classification_loss = tf.reduce_sum(class_loss)
    
    # =====================================================================
    # 4. Pérdida Total (Se re-incluye la pérdida de clasificación)
    # =====================================================================
    total_loss = localization_loss + confidence_loss + classification_loss
    
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
    
    # 🛑 FIX ARQUITECTÓNICO: Insertar Resizing para forzar 7x7
    # La salida de EfficientNetB0 es 10x10 o 13x13 para input 320x320.
    # Necesitamos 7x7 para que coincida con GRID_SIZE.
    x = layers.Resizing(GRID_SIZE, GRID_SIZE, interpolation='bilinear')(x) # <-- ¡ESTA ES LA LÍNEA CRÍTICA!

    # Capas convolucionales para refinar las características del backbone
    # Ahora esta capa ya recibe 7x7 y opera sobre 7x7.
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x) 
    x = layers.Dropout(0.3)(x)

    # Última capa Conv: La salida final debe ser (S, S, OUTPUT_DIM) -> (7, 7, 16)
    outputs = layers.Conv2D(
        OUTPUT_DIM, 
        kernel_size=1, 
        padding='same', 
        name="detection_output",
        activation='sigmoid' 
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