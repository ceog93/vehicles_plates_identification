# src/inference/predict_video.py

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR, THRESHOLD, ROOT_MODEL_DIR, LATEST_MODEL_PATH
from src.utils.mpd_utils import resize_pad, nms_numpy
from src.models.efficient_detector_multi_placa import GRID_SIZE, NUM_ANCHORS, NUM_CLASSES, yolo_ciou_loss
import numpy as np

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
    
    # Reducir el tensor si viene con batch dim: (1, G, G, C) -> (G, G, C)
    arr = np.asarray(output_tensor)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    gh, gw, channels = arr.shape[:3]
    per_anchor = 5 + NUM_CLASSES
    anchors = int(channels // per_anchor)
    arr = arr.reshape(gh, gw, anchors, per_anchor)

    for i in range(gh):
        for j in range(gw):
            for a in range(anchors):
                cell = arr[i, j, a]
                confidence = float(cell[0])
                if confidence < confidence_threshold:
                    continue
                cx_local = float(cell[1])
                cy_local = float(cell[2])
                w_norm = float(cell[3])
                h_norm = float(cell[4])

                cx_norm = (j + cx_local) / gh
                cy_norm = (i + cy_local) / gh

                xmin_norm = cx_norm - (w_norm / 2)
                ymin_norm = cy_norm - (h_norm / 2)
                xmax_norm = cx_norm + (w_norm / 2)
                ymax_norm = cy_norm + (h_norm / 2)

                final_boxes_norm.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
                final_scores.append(confidence)

    # Aplicar NMS para eliminar duplicados cercanos
    if len(final_boxes_norm) > 0:
        selected = nms_numpy(final_boxes_norm, final_scores, iou_thresh=0.45, score_thresh=0.2)
        final_boxes = [final_boxes_norm[i] for i in selected]
        final_scores = [final_scores[i] for i in selected]
    else:
        final_boxes = []

    return final_boxes, final_scores

# =======================================================================
# 2. CARGA DEL MODELO (Añadir custom_objects)
# =======================================================================

def find_latest_model_in_models_dir():
    if not os.path.isdir(ROOT_MODEL_DIR):
        return None
    subdirs = [d for d in os.listdir(ROOT_MODEL_DIR) if os.path.isdir(os.path.join(ROOT_MODEL_DIR, d))]
    if not subdirs:
        return None
    latest = max(subdirs)
    candidate_dir = os.path.join(ROOT_MODEL_DIR, latest)
    for name in ("detector_model.keras", "detector.keras", "detector.h5", "detector"):
        p = os.path.join(candidate_dir, name)
        if os.path.exists(p):
            return p
    for f in os.listdir(candidate_dir):
        if f.endswith('.keras') or f.endswith('.h5'):
            return os.path.join(candidate_dir, f)
    return None


def load_model_safe(model_path=None):
    candidates = []
    if model_path:
        candidates.append(model_path)
    candidates.append(MODEL_PATH)
    if LATEST_MODEL_PATH:
        candidates.append(LATEST_MODEL_PATH)
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
    raise FileNotFoundError(f"No se encontró modelo válido. Intentos: {candidates}. Ultimo error: {last_err}")

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
            
                # Dibujar Etiqueta y Confianza (mostrar %)
            label = f"Placa: {score*100:.0f}%"
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
