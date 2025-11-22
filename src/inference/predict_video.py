# src/inference/predict_video.py

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR, THRESHOLD 
from src.utils.mpd_utils import resize_pad
from src.models.efficient_detector_multi_placa import GRID_SIZE, BBOX_ANCHORS, yolo_like_loss # Importar para decodificación y carga

# ... (otras configuraciones de directorio - SIN CAMBIOS) ...

# =======================================================================
# 1. FUNCIÓN DE DECODIFICACIÓN Y POST-PROCESAMIENTO
# =======================================================================

def process_predictions(output_tensor, img_size=IMG_SIZE[0], confidence_threshold=THRESHOLD):
    """
    Decodifica la salida (7, 7, 16) del modelo, filtra por umbral de confianza 
    y retorna BBoxes normalizados.
    """
    
    final_boxes_norm = [] # [xmin, ymin, xmax, ymax] normalizado a [0, 1] del frame padded
    final_scores = []
    
    # Reducir el tensor de (1, 7, 7, 16) a (7, 7, 16)
    output_tensor = output_tensor[0] 

    for i in range(GRID_SIZE): # cell_y
        for j in range(GRID_SIZE): # cell_x
            for b in range(BBOX_ANCHORS): # ancla
                
                start_idx = b * 5 
                prediction = output_tensor[i, j, start_idx : start_idx + 5]
                
                confidence = prediction[0]
                
                # APLICAR UMBRAL DE CONFIANZA
                if confidence >= confidence_threshold:
                    
                    # 1. Decodificación de Coordenadas (de local a normalizado [0, 1] de la imagen)
                    cx_local, cy_local, w_norm, h_norm = prediction[1:]
                    
                    cx_norm = (j + cx_local) / GRID_SIZE 
                    cy_norm = (i + cy_local) / GRID_SIZE 
                    
                    # 2. De (cx, cy, w, h) a (xmin, ymin, xmax, ymax) normalizado
                    xmin_norm = cx_norm - (w_norm / 2)
                    ymin_norm = cy_norm - (h_norm / 2)
                    xmax_norm = cx_norm + (w_norm / 2)
                    ymax_norm = cy_norm + (h_norm / 2)
                    
                    final_boxes_norm.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
                    final_scores.append(confidence)

    # Nota: Aquí se aplicaría la NMS si la función apply_nms estuviera implementada.
    return final_boxes_norm, final_scores

# =======================================================================
# 2. CARGA DEL MODELO (Añadir custom_objects)
# =======================================================================

def load_model_safe(model_path):
    try:
        # Importar la pérdida personalizada para cargar el modelo
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'yolo_like_loss': yolo_like_loss}, 
            compile=False # No re-compilar
        )
        print(f"Modelo cargado desde: {model_path}")
        return model
    except Exception as e:
        print("Error cargando el modelo:", e)
        raise

# =======================================================================
# 3. FUNCIÓN PRINCIPAL DE PROCESAMIENTO (process_video)
# =======================================================================

def process_video(model, video_path, out_video_path=None, img_size=IMG_SIZE[0], display=False):
    # ... (Carga de video, FPS, width, height, writer - SIN CAMBIOS) ...
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ... (Definición de out_video_path, fourcc, writer - SIN CAMBIOS) ...
    if out_video_path is None:
        base = os.path.basename(video_path)
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
        pred_tensor = model.predict(inp)
        
        # 3. Decodificación y Filtrado por Umbral
        boxes_norm_padded, scores = process_predictions(pred_tensor, img_size=img_size)

        out_frame = frame.copy()

        # 4. Iterar sobre las cajas detectadas y dibujar
        for box_norm, score in zip(boxes_norm_padded, scores):
            
            # BBox normalizado [0, 1] -> BBox en píxeles del frame padded
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
            
            # Dibujar Etiqueta y Confianza
            label = f"Placa: {score:.2f}"
            cv2.putText(out_frame, label, (x1_orig, max(0, y1_orig - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        # ... (Escritura y display - SIN CAMBIOS) ...
        writer.write(out_frame)
        if display:
            cv2.imshow("Detección", out_frame)
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

def main():
    parser = argparse.ArgumentParser(description="Inferir un video con detector de placas")
    parser.add_argument("--video", required=True, help="Ruta a archivo de video")
    parser.add_argument("--model", default=MODEL_PATH, help="Ruta al modelo Keras")
    parser.add_argument("--out", default=None, help="Ruta de salida mp4 opcional")
    parser.add_argument("--display", action="store_true", help="Mostrar video en pantalla")
    args = parser.parse_args()

    model = load_model_safe(args.model)
    out = process_video(model, args.video, out_video_path=args.out, display=args.display)
    print("Hecho.")

if __name__ == "__main__":
    main()
