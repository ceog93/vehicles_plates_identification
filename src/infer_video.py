# src/infer_video.py

"""
Inferencia sobre un video para detección Multiplaca:
- Lee frames, aplica letterbox (resize_pad)
- Predice el tensor de salida (7, 7, 16)
- Decodifica el tensor, filtra por THRESHOLD, mapea coords y dibuja BBoxes con confianza.
- Guarda output video.
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR, THRESHOLD, ROOT_MODEL_DIR, LATEST_MODEL_PATH
from src.utils.mpd_utils import resize_pad
# Importar parámetros del modelo Multiplaca y custom loss para cargar el modelo
from src.models.efficient_detector_multi_placa import GRID_SIZE, NUM_ANCHORS, yolo_ciou_loss
 

os.makedirs(OUTPUT_FEED_DIR, exist_ok=True)

# =======================================================================
# 1. FUNCIÓN DE DECODIFICACIÓN Y POST-PROCESAMIENTO
# =======================================================================

def process_predictions(output_tensor, img_size=IMG_SIZE[0], confidence_threshold=THRESHOLD):
    """
    Decodifica la salida (1, 7, 7, 16) del modelo Multiplaca, filtra por umbral de confianza 
    y retorna BBoxes normalizados ([xmin, ymin, xmax, ymax]) y sus scores.
    """
    
    final_boxes_norm = [] # [xmin, ymin, xmax, ymax] normalizado a [0, 1] del frame padded
    final_scores = []
    
    # Reducir el tensor de (1, S, S, D) a (S, S, D)
    output_tensor = output_tensor[0] 

    for i in range(GRID_SIZE): # cell_y
        for j in range(GRID_SIZE): # cell_x
            for b in range(NUM_ANCHORS): # ancla
                
                # Índice de inicio para la ancla 'b'. Cada ancla tiene 5 valores [C, cx, cy, w, h]
                start_idx = b * 5 
                prediction = output_tensor[i, j, start_idx : start_idx + 5]
                
                confidence = prediction[0] # El primer valor es la confianza
                
                # 🚨 APLICAR UMBRAL DE CONFIANZA: Muestra BBox SOLAMENTE si está seguro.
                if confidence >= confidence_threshold:
                    
                    # 1. Decodificación de Coordenadas (de local a normalizado [0, 1] de la imagen)
                    cx_local, cy_local, w_norm, h_norm = prediction[1:]
                    
                    # Coordenadas del centro normalizadas [0, 1] respecto a la imagen
                    cx_norm = (j + cx_local) / GRID_SIZE 
                    cy_norm = (i + cy_local) / GRID_SIZE 
                    
                    # 2. De (cx, cy, w, h) a (xmin, ymin, xmax, ymax) normalizado
                    xmin_norm = cx_norm - (w_norm / 2)
                    ymin_norm = cy_norm - (h_norm / 2)
                    xmax_norm = cx_norm + (w_norm / 2)
                    ymax_norm = cy_norm + (h_norm / 2)
                    
                    final_boxes_norm.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
                    final_scores.append(confidence)

    # Nota: Aquí se aplicaría Non-Maximum Suppression (NMS) si fuera necesario para múltiples cajas.
    return final_boxes_norm, final_scores

# =======================================================================
# 2. CARGA Y EJECUCIÓN DEL MODELO
# =======================================================================

def find_latest_model_in_models_dir():
    # Buscar la última carpeta en ROOT_MODEL_DIR y localizar un archivo detector_model.keras o similar
    if not os.path.isdir(ROOT_MODEL_DIR):
        return None
    subdirs = [d for d in os.listdir(ROOT_MODEL_DIR) if os.path.isdir(os.path.join(ROOT_MODEL_DIR, d))]
    if not subdirs:
        return None
    latest = max(subdirs)
    candidate_dir = os.path.join(ROOT_MODEL_DIR, latest)
    # buscar archivos comunes de checkpoint
    for name in ("detector_model.keras", "detector.keras", "detector.h5", "detector"):
        p = os.path.join(candidate_dir, name)
        if os.path.exists(p):
            return p
    # si no encontramos nombres exactos, buscar cualquier archivo .keras o .h5
    for f in os.listdir(candidate_dir):
        if f.endswith('.keras') or f.endswith('.h5'):
            return os.path.join(candidate_dir, f)
    return None


def load_model_safe(model_path=None):
    # Prioridad: model_path (argumento) -> MODEL_PATH (config) -> LATEST_MODEL_PATH -> buscar en ROOT_MODEL_DIR
    candidates = []
    if model_path:
        candidates.append(model_path)
    candidates.append(MODEL_PATH)
    if LATEST_MODEL_PATH:
        candidates.append(LATEST_MODEL_PATH)
    # última oportunidad: buscar automáticamente
    auto = find_latest_model_in_models_dir()
    if auto:
        candidates.append(auto)

    last_err = None
    for p in candidates:
        if not p:
            continue
        if not os.path.exists(p):
            continue
        try:
            model = tf.keras.models.load_model(p, custom_objects={'yolo_ciou_loss': yolo_ciou_loss}, compile=False)
            print(f"Modelo cargado desde: {p}")
            return model
        except Exception as e:
            last_err = e
            print(f"Intento de carga falló para {p}: {e}")

    # Si no se pudo cargar ninguno
    msg = f"No se pudo cargar ningún modelo. Intentos: {candidates}. Ultimo error: {last_err}"
    raise FileNotFoundError(msg)

def run_video_detection(video_path, out_video_path=None, img_size=IMG_SIZE[0], display=False):
    ''' Función principal para correr la detección en un video. '''
    
    model = load_model_safe(MODEL_PATH) # Cargar el modelo guardado por train.py

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if out_video_path is None:
        base = os.path.basename(video_path)
        from src.config import OUTPUT_FEED_DIR # Importar OUTPUT_FEED_DIR
        out_video_path = os.path.join(OUTPUT_FEED_DIR, f"det_{base}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Procesando frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Preprocesamiento (Resize + Pad)
        img_pad, scale, top, left = resize_pad(rgb, img_size)
        inp = (img_pad.astype(np.float32) / 255.0)[None, ...]

        # 2. Predicción (Salida: 1, 7, 7, 16)
        pred_tensor = model.predict(inp, verbose=0)
        
        # 3. Decodificación y Filtrado por Umbral (process_predictions)
        boxes_norm_padded, scores = process_predictions(pred_tensor, img_size=img_size)

        out_frame = frame.copy()

        # 4. Iterar sobre las cajas detectadas que superaron el umbral y dibujar
        for box_norm, score in zip(boxes_norm_padded, scores):
            
            # Mapear de BBox normalizado a píxeles del frame padded
            x1_p = int(box_norm[0] * img_size)
            y1_p = int(box_norm[1] * img_size)
            x2_p = int(box_norm[2] * img_size)
            y2_p = int(box_norm[3] * img_size)

            # Mapear de padded -> original
            x1_orig = int(max(0, (x1_p - left) / scale))
            y1_orig = int(max(0, (y1_p - top) / scale))
            x2_orig = int(min(width, (x2_p - left) / scale))
            y2_orig = int(min(height, (y2_p - top) / scale))

            # Dibujar BBox (Solo las que pasaron el umbral)
            cv2.rectangle(out_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
            
            # Dibujar Etiqueta y Confianza (mostrar % confianza)
            label = f"Placa: {score*100:.0f}%"
            cv2.putText(out_frame, label, (x1_orig, max(0, y1_orig - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        # 5. Escritura y Display
        writer.write(out_frame)
        if display:
            cv2.imshow("Detección Multiplaca", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()
    if display:
        cv2.destroyAllWindows()

    print(f"Video de salida guardado en: {out_video_path}")
    return out_video_path


if __name__ == "__main__":
    from src.config import TEST_VIDEO_PATH
    run_video_detection(TEST_VIDEO_PATH)