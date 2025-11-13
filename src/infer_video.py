# src/infer_video.py

import cv2
import numpy as np
import tensorflow as tf
import os

# ============================
# CONFIGURACIÓN (Debe coincidir con train.py)
# ============================
MODEL_PATH = "models/detector.h5"
IMG_SIZE = (224, 224)
# Puedes ajustar el color y grosor del recuadro
BOX_COLOR = (0, 255, 0)  # Verde BGR
BOX_THICKNESS = 2

# ============================
# CARGA DEL MODELO (MODIFICADA)
# ============================
MODEL_PATH = "models/detector.h5"

try:
    # Definimos las métricas que el modelo espera, aunque no las usemos para inferencia.
    # Usamos las funciones predefinidas de Keras/TensorFlow.
    custom_objects = {
        'mse': tf.keras.losses.MeanSquaredError(), # Usa la clase de pérdida para 'mse'
        'mae': tf.keras.metrics.MeanAbsoluteError() # Usa la clase de métrica para 'mae'
    }

    # Cargar el modelo con custom_objects
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects=custom_objects,
        compile=False # Deshabilitar la recompilación puede ayudar a evitar errores
    )
    print(f"✅ Modelo de detección cargado exitosamente desde: {MODEL_PATH}")

except Exception as e:
    # Si sigue fallando, la arquitectura del modelo podría ser la causa
    print(f"❌ Error al cargar el modelo: {e}")
    print("Asegúrate de que la versión de TensorFlow en tu venv sea la misma que la de entrenamiento.")
    exit()

# ============================
# PREPROCESAMIENTO
# ============================
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocesa un frame para que coincida con la entrada del modelo.
    """
    # 1. Redimensionar al tamaño de entrenamiento (224x224)
    img_resized = cv2.resize(frame, IMG_SIZE)
    # 2. Normalizar a [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    # 3. Añadir la dimensión del batch (1, 224, 224, 3)
    return np.expand_dims(img_normalized, axis=0)

# ============================
# DIBUJAR CAJA DELIMITADORA
# ============================
def draw_bbox(frame: np.ndarray, normalized_box: np.ndarray) -> np.ndarray:
    """
    Convierte las coordenadas normalizadas (x_c, y_c, w, h) a coordenadas
    absolutas de píxeles y dibuja el recuadro en el frame.
    """
    # 1. Dimensiones originales del frame
    h, w = frame.shape[:2]
    
    # 2. Desnormalizar las predicciones
    x_center_norm, y_center_norm, box_width_norm, box_height_norm = normalized_box

    # 3. Convertir a coordenadas absolutas de la caja (xmin, ymin, xmax, ymax)
    
    # Centro
    center_x = int(x_center_norm * w)
    center_y = int(y_center_norm * h)
    
    # Ancho y Alto
    box_w = int(box_width_norm * w)
    box_h = int(box_height_norm * h)

    # Coordenadas de esquina (xmin, ymin) y (xmax, ymax)
    xmin = int(center_x - box_w / 2)
    ymin = int(center_y - box_h / 2)
    xmax = int(center_x + box_w / 2)
    ymax = int(center_y + box_h / 2)
    
    # Asegurar que las coordenadas estén dentro del frame
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    # 4. Dibujar el rectángulo
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), BOX_COLOR, BOX_THICKNESS)
    
    # 5. Opcional: Añadir una etiqueta
    label = "PLACA DETECTADA"
    # Posicionar el texto encima de la caja
    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLOR, 2)
    
    return frame

# ============================
# INFERENCIA PRINCIPAL
# ============================
def run_video_detection(video_path: str):
    """
    Ejecuta el detector de placas en un archivo de video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error al abrir el video en la ruta: {video_path}")
        return

    print(f"\n🎥 Iniciando detección en el video: {video_path}")

    cv2.namedWindow("Detección de Placas (Regresion)", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break # Fin del video

        # 1. Preprocesar y Predecir
        input_frame = preprocess_frame(frame)
        # El modelo predice [x_c, y_c, w, h] normalizados
        predictions = model.predict(input_frame, verbose=0)[0] 
        
        # 2. Dibujar la caja delimitadora en el frame original
        frame_with_box = draw_bbox(frame, predictions)

        # 3. Mostrar el resultado
        cv2.imshow("Detección de Placas (Regresion)", frame_with_box) # EL MISMO NOMBRE!

        # 4. Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Detección de video finalizada.")