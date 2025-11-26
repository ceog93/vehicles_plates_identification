# src/inference/predict_video.py
# Versión Final Definitiva: Etiquetas Completas y Lógica de Tracking
import re
import os
import argparse
from datetime import datetime
import csv
import queue
import threading
import time
import traceback

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ---------------------------
# CONFIGURACIÓN DEL ENTORNO TF
# ---------------------------
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# OCR - Importación segura
try:
    import easyocr
    # Inicializamos el lector solo una vez para evitar recargas
    # Ajuste de mag_ratio para detectar placas (texto grande/compacto)
    READER = easyocr.Reader(['es', 'en'], gpu=False)
    # Parámetros predeterminados para la lectura: solo letras y números
    OCR_DEFAULTS = {'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'mag_ratio': 1.0}
except Exception:
    print("Advertencia: EasyOCR no está instalado o falló la inicialización. OCR deshabilitado.")
    READER = None
    OCR_DEFAULTS = {} # Definir por si se usa en run_ocr aunque READER sea None

# Algoritmo Húngaro para tracking
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_HUNGARIAN = True
except Exception:
    linear_sum_assignment = None
    _HAS_HUNGARIAN = False

# Importaciones del proyecto
from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR, THRESHOLD, ROOT_MODEL_DIR, LATEST_MODEL_PATH
from src.utils.mpd_utils import resize_pad, nms_numpy
from src.models.efficient_detector_multi_placa import NUM_CLASSES, yolo_ciou_loss

# ---------------------------
# UTILIDAD DE DIBUJADO (NUEVO - CENTRALIZADO)
# ---------------------------
def draw_labels(image, bbox, track_id, score, ocr_text="", color=(0, 255, 0)):
    """
    Dibuja bounding box y etiquetas con fondo oscuro para máxima legibilidad.
    Garantiza que la imagen guardada y el video tengan la misma info visual.
    Formatos:
      - Arriba: ID:<id> | <score>%
      - Abajo: PLACA: <ocr>
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_img, w_img = image.shape[:2]

    # 1. Dibujar rectángulo de la placa (Bounding Box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # 2. Preparar etiqueta superior (ID + Confianza)
    # Formato solicitado: ID:<track_id> | porcentaje
    label_top = f"ID:{track_id} | {score*100:.0f}%"
    (w_top, h_top), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    # Fondo negro superior para legibilidad
    # Dibujamos encima de la caja (y1)
    cv2.rectangle(image, (x1, max(0, y1 - 25)), (x1 + w_top + 10, max(0, y1)), (0, 0, 0), -1)
    # Texto superior (Blanco)
    cv2.putText(image, label_top, (x1 + 5, max(0, y1 - 7)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 3. Preparar etiqueta inferior (OCR) solo si existe texto
    if ocr_text:
        # Formato explícito solicitado: "PLACA: <texto>"
        label_bot = f"PLACA: {ocr_text}"
        (w_bot, h_bot), _ = cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Calcular posición inferior asegurando que no se salga de la imagen
        y_bot_start = min(h_img, y2)
        y_bot_end = min(h_img, y2 + 30)
        
        # Fondo negro inferior
        cv2.rectangle(image, (x1, y_bot_start), (x1 + w_bot + 10, y_bot_end), (0, 0, 0), -1)
        # Texto inferior (Amarillo Cyan para resaltar)
        cv2.putText(image, label_bot, (x1 + 5, min(h_img - 5, y2 + 22)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return image

def draw_bbox_safe(image, bbox, track_id, score, ocr_text="", color=(0, 255, 0)):
    """
    Dibuja SOLO el bounding box, ignorando errores de etiquetas/OCR.
    Garantiza que el bbox SIEMPRE aparezca, independientemente de problemas con texto.
    """
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Asegurar que los valores están dentro de la imagen
        h_img, w_img = image.shape[:2]
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(x1 + 1, min(x2, w_img))
        y2 = max(y1 + 1, min(y2, h_img))
        
        # DIBUJAR BBOX (lo más importante)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Intentar agregar etiquetas (si falla, el bbox ya está dibujado)
        try:
            label_top = f"ID:{track_id} | {score*100:.0f}%"
            (w_top, h_top), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, max(0, y1 - 25)), (x1 + w_top + 10, max(0, y1)), (0, 0, 0), -1)
            cv2.putText(image, label_top, (x1 + 5, max(0, y1 - 7)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except:
            pass  # Si las etiquetas fallan, el bbox ya está dibujado
        
        # Intentar agregar OCR (si falla, el bbox ya está dibujado)
        if ocr_text:
            try:
                label_bot = f"PLACA: {ocr_text}"
                (w_bot, h_bot), _ = cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                y_bot_start = min(h_img, y2)
                y_bot_end = min(h_img, y2 + 30)
                cv2.rectangle(image, (x1, y_bot_start), (x1 + w_bot + 10, y_bot_end), (0, 0, 0), -1)
                cv2.putText(image, label_bot, (x1 + 5, min(h_img - 5, y2 + 22)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            except:
                pass  # Si OCR falla, el bbox ya está dibujado
    except Exception:
        pass  # Si todo falla, devolver la imagen sin cambios
    
    return image

# ---------------------------
# FUNCIONES AUXILIARES
# ---------------------------
def process_predictions(output_tensor, img_size=IMG_SIZE[0], confidence_threshold=THRESHOLD):
    """Decodifica la salida del modelo a cajas [x1, y1, x2, y2] normalizadas."""
    arr = np.asarray(output_tensor)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    gh, gw, channels = arr.shape[:3]
    per_anchor = 5 + NUM_CLASSES
    anchors = int(channels // per_anchor)
    arr = arr.reshape(gh, gw, anchors, per_anchor)

    final_boxes_norm = []
    final_scores = []
    
    for i in range(gh):
        for j in range(gw):
            for a in range(anchors):
                cell = arr[i, j, a]
                conf = float(cell[0])
                if conf < confidence_threshold:
                    continue
                cx_local, cy_local = float(cell[1]), float(cell[2])
                w_norm, h_norm = float(cell[3]), float(cell[4])
                cx_norm = (j + cx_local) / gw
                cy_norm = (i + cy_local) / gh
                xmin = cx_norm - (w_norm / 2.0)
                ymin = cy_norm - (h_norm / 2.0)
                xmax = cx_norm + (w_norm / 2.0)
                ymax = cy_norm + (h_norm / 2.0)
                final_boxes_norm.append([xmin, ymin, xmax, ymax])
                final_scores.append(conf)

    if len(final_boxes_norm) > 0:
        selected = nms_numpy(final_boxes_norm, final_scores, iou_thresh=0.45, score_thresh=0.0)
        final_boxes = [final_boxes_norm[i] for i in selected]
        final_scores = [final_scores[i] for i in selected]
    else:
        final_boxes, final_scores = [], []

    return final_boxes, final_scores


def find_latest_model_in_models_dir():
    if not os.path.isdir(ROOT_MODEL_DIR):
        return None
    subdirs = [d for d in os.listdir(ROOT_MODEL_DIR) if os.path.isdir(os.path.join(ROOT_MODEL_DIR, d))]
    if not subdirs:
        return None
    subdirs.sort(reverse=True)
    for sd in subdirs:
        candidate_dir = os.path.join(ROOT_MODEL_DIR, sd)
        for name in ("detector_model.keras", "detector.keras", "detector.h5", "detector"):
            p = os.path.join(candidate_dir, name)
            if os.path.exists(p):
                return p
        for f in os.listdir(candidate_dir):
            if f.endswith(".keras") or f.endswith(".h5"):
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
        if not p: continue
        if not os.path.exists(p): continue
        try:
            model = tf.keras.models.load_model(p, custom_objects={'yolo_ciou_loss': yolo_ciou_loss}, compile=False)
            print(f"✔ Modelo cargado desde: {p}")
            return model
        except Exception as e:
            last_err = e
            print(f"Intento de carga falló para {p}: {e}")
    raise FileNotFoundError(f"No se encontró modelo válido. Intentos: {candidates}. Ultimo error: {last_err}")


import re # Asegúrate de que esta librería esté importada en la parte superior del archivo

def run_ocr(cropped_img):
    # OCR_DEFAULTS = {'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'mag_ratio': 1.0}
    # Asegúrate de que OCR_DEFAULTS y READER están definidos correctamente
    if READER is None:
        return "[OCR NO INSTALADO]"
    try:
        # 0. Conversión de color (necesario para EasyOCR)
        if len(cropped_img.shape) == 3 and cropped_img.shape[2] == 3:
            rgb_crop = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_crop = cropped_img
        
        # 1. Ejecutar OCR usando la configuración optimizada (mag_ratio, allowlist)
        results = READER.readtext(rgb_crop, **OCR_DEFAULTS, detail=0, paragraph=True)
        
        if results:
            # 2. Limpieza de texto:
            # Elimina todos los espacios en blanco (incluyendo \ua0, \xa0, tabs, etc.) y guiones
            # Usa 're.sub' para una limpieza de espacios más robusta
            raw_text = "".join(results)
            raw_text = re.sub(r'\s+', '', raw_text).replace("-", "").upper()
            
            # 3. **POST-PROCESAMIENTO ESTRICTO PARA FORMATO COLOMBIANO (6 caracteres)**
            if len(raw_text) == 6:
                corrected_plate = ""
                
                # --- Bloque Alfabético (Posiciones 0, 1, 2) ---
                for i in range(3):
                    char = raw_text[i]
                    # Forzar a LETRA: Reemplazar 0 por O, 1 por I
                    if char == '0':
                        corrected_plate += 'O'
                    elif char == '1' or char == '4':
                        corrected_plate += 'I'
                    else:
                        corrected_plate += char

                # --- Bloque Numérico/Alfanumérico (Posiciones 3, 4, 5) ---
                for i in range(3, 6):
                    char = raw_text[i]
                    # Forzar a NÚMERO (si estamos en pos 3 o 4)
                    if i <= 4:
                        if char == 'O' or char == 'Q':
                            corrected_plate += '0'
                        elif char == 'I' or char == 'L':
                            corrected_plate += '1'
                        elif char == 'S':
                            corrected_plate += '5'
                        else:
                            corrected_plate += char
                    else: # Posición 5 (puede ser N o L)
                        corrected_plate += char
                
                if len(corrected_plate) == 6:
                    return corrected_plate
                
                return raw_text # Fallback si no se logra la corrección de 6 caracteres

            return raw_text # Si no es de 6, devolver el texto limpio
        return ""
    except Exception as e:
        return f"[OCR ERROR: {e}]"


def _iou_numpy(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    denom = areaA + areaB - interArea
    return interArea / denom if denom > 0 else 0.0


def _center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])

def select_best_detection_from_history(track):
    """
    Selecciona la mejor detección del histórico del track.
    Prioridad:
    1. OCR exitoso con mejor score
    2. Score más alto
    3. Frame más reciente
    """
    history = track.get('detection_history', [])
    if not history:
        return track.get('best_frame'), track.get('best_bbox'), track.get('best_score'), track.get('ocr_text', '')
    
    best_det = history[0]
    # Preferir OCR que cumpla el formato colombiano (AAA999)
    import re as _re
    col_plate_re = _re.compile(r'^[A-Z]{3}[0-9]{3}$')

    # 1) buscar coincidencias válidas por formato
    col_matches = [d for d in history if d.get('ocr_text') and col_plate_re.match(d.get('ocr_text'))]
    if col_matches:
        best_det = max(col_matches, key=lambda x: x['score'])
        return (best_det['frame'], best_det['bbox'], best_det['score'], best_det.get('ocr_text', ''))

    # 2) buscar OCR exitoso (sin ERR)
    ocr_success = [d for d in history if d.get('ocr_text') and 'ERR' not in d.get('ocr_text') and '[OCR' not in d.get('ocr_text')]
    if ocr_success:
        best_det = max(ocr_success, key=lambda x: x['score'])
        return (best_det['frame'], best_det['bbox'], best_det['score'], best_det.get('ocr_text', ''))

    # 3) fallback: usar la detección con mayor score
    best_det = max(history, key=lambda x: x['score'])
    
    return (
        best_det['frame'],
        best_det['bbox'],
        best_det['score'],
        best_det.get('ocr_text', '')
    )

def match_detections_to_tracks(detections, tracks, iou_thr=0.30, dist_thr=0.35):
    if len(tracks) == 0:
        return [], [], list(range(len(detections)))
    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    T = len(tracks); D = len(detections)
    cost = np.ones((T, D), dtype=np.float32)

    for ti, tr in enumerate(tracks):
        box_t = tr['bbox']; ct = _center(box_t)
        ref = max(1.0, np.linalg.norm(np.array([box_t[2] - box_t[0], box_t[3] - box_t[1]])))
        for di, det in enumerate(detections):
            box_d = det['bbox']; cd = _center(box_d)
            iou_v = _iou_numpy(box_t, box_d)
            dist = np.linalg.norm(ct - cd)
            dist_norm = np.clip(1.0 - (dist / (ref + 1e-6)), 0.0, 1.0)
            score = 0.70 * iou_v + 0.30 * dist_norm 
            cost[ti, di] = 1.0 - score

    cost_matrix = cost.copy()
    matches = []
    matched_t = set(); matched_d = set()

    if _HAS_HUNGARIAN and linear_sum_assignment is not None:
        try:
            t_idx, d_idx = linear_sum_assignment(cost_matrix)
            for ti, di in zip(t_idx, d_idx):
                if _iou_numpy(tracks[ti]['bbox'], detections[di]['bbox']) >= iou_thr:
                    matches.append((ti, di))
                    matched_t.add(ti); matched_d.add(di)
        except Exception:
            pass

    if len(matches) == 0:
        flat = []
        for ti in range(T):
            for di in range(D):
                flat.append((cost_matrix[ti, di], ti, di))
        flat.sort(key=lambda x: x[0])
        used_t = set(); used_d = set()
        for c, ti, di in flat:
            if ti in used_t or di in used_d: continue
            if _iou_numpy(tracks[ti]['bbox'], detections[di]['bbox']) >= iou_thr:
                matches.append((ti, di))
                used_t.add(ti); used_d.add(di)
        matched_t.update(used_t); matched_d.update(used_d)

    unmatched_t = [i for i in range(T) if i not in matched_t]
    unmatched_d = [j for j in range(D) if j not in matched_d]
    return matches, unmatched_t, unmatched_d


# ---------------------------
# SAVER THREAD
# ---------------------------
def start_saver_thread(q, csv_path):
    stop_token = object()

    def worker():
        rows = {} 
        if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
            try:
                with open(csv_path, 'r', newline='') as cf:
                    r = csv.DictReader(cf)
                    for rec in r:
                        tid = rec.get('track_id', '')
                        if tid: rows[tid] = rec
            except Exception: pass

        fieldnames = ['track_id', 'filepath', 'score', 'ocr_text', 'timestamp', 'video_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']

        def write_csv():
            try:
                with open(csv_path, 'w', newline='') as cf:
                    writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    writer.writeheader()
                    for v in rows.values(): writer.writerow(v)
            except Exception as e: print("Error CSV:", e)

        while True:
            job = q.get()
            if job is stop_token:
                write_csv()
                q.task_done()
                break

            try:
                if job.get('type') == 'image':
                    path = job['path']
                    # La imagen job['image'] ya viene dibujada desde el proceso principal
                    saved = cv2.imwrite(path, job['image'])
                    
                    if saved:
                        tid = str(job.get('track_id', ''))
                        bbox = job.get('bbox', [0,0,0,0])
                        rows[tid] = {
                            'track_id': tid,
                            'filepath': path,
                            'score': f"{job.get('score', 0):.4f}",
                            'ocr_text': job.get('ocr_text', ''),
                            'timestamp': job.get('timestamp', ''),
                            'video_path': job.get('video_path', ''),
                            'bbox_x1': str(bbox[0]), 'bbox_y1': str(bbox[1]),
                            'bbox_x2': str(bbox[2]), 'bbox_y2': str(bbox[3]),
                        }
                        write_csv()
            except Exception:
                pass
            finally:
                q.task_done()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t, stop_token

# ---------------------------
# OCR THREAD (NUEVO PARA RENDIMIENTO)
# ---------------------------
def start_ocr_thread(q, results_dict):
    """
    Inicia un hilo de trabajo para procesar OCR de forma asíncrona.
    - q: La cola de la que se leen los trabajos de OCR.
    - results_dict: Un diccionario compartido donde se almacenan los resultados.
    """
    stop_token = object()

    def ocr_worker():
        while True:
            job = q.get()
            if job is stop_token:
                q.task_done()
                break

            track_id = job.get('track_id')
            crop_img = job.get('image')

            if track_id is None or crop_img is None or crop_img.size == 0:
                q.task_done()
                continue
            
            try:
                # Ejecutar el OCR (la parte lenta)
                ocr_text = run_ocr(crop_img)
                # Guardar el resultado en el diccionario compartido
                if ocr_text:
                    results_dict[track_id] = ocr_text
            finally:
                q.task_done()

    t = threading.Thread(target=ocr_worker, daemon=True)
    t.start()
    return t, stop_token

# ---------------------------
# PROCESO PRINCIPAL
# ---------------------------
def process_video(model, video_path, out_video_path=None, img_size=IMG_SIZE[0], display=False,
                  max_missed=10, iou_thresh=0.30, confirm_frames=3, min_area=400,
                  aspect_ratio_min=1.8, aspect_ratio_max=8.0, dampening_factor=0.85,
                  ocr_padding_ratio=0.15):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise FileNotFoundError(f"❌ No se pudo abrir: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    base = os.path.basename(video_path)
    UNIQUE_OUTPUT_DIR = os.path.join(OUTPUT_FEED_DIR, ts)
    os.makedirs(UNIQUE_OUTPUT_DIR, exist_ok=True)

    if out_video_path is None:
        out_video_path = os.path.join(UNIQUE_OUTPUT_DIR, f"det_{base}.mp4")

    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    pbar = tqdm(total=total_frames, desc="Procesando")

    try:
        dummy = np.zeros((1, img_size, img_size, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)
    except Exception: pass

    saver_q = queue.Queue()
    saver_thread, stop_token = start_saver_thread(saver_q, os.path.join(UNIQUE_OUTPUT_DIR, 'metadata.csv'))

    ocr_q = queue.Queue()
    ocr_results = {} # Diccionario para almacenar resultados de OCR {track_id: text}
    ocr_thread, ocr_stop_token = start_ocr_thread(ocr_q, ocr_results)

    tracks = []
    next_track_id = 1

    if display:
        try:
            cv2.namedWindow("Detección", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detección", min(1280, width), min(720, height))
        except: pass

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        #if frame_idx <= 3:
            #print(f"[DEBUG] Leyendo frame {frame_idx}...", flush=True)

        bgr = frame.copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_pad, scale, top, left = resize_pad(rgb, img_size)
        inp = (img_pad.astype(np.float32) / 255.0)[None, ...]

        pred_tensor = model.predict(inp, verbose=0)
        boxes_norm, scores = process_predictions(pred_tensor, img_size=img_size)

        if frame_idx <= 3:
            try:
                arr = np.asarray(pred_tensor)
                #print(f"[DEBUG] pred_tensor shape: {arr.shape}, scores_len={len(scores)}", flush=True)
            except Exception:
                pass
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                all_conf = arr[:,:,:,0].flatten()
                max_conf = np.max(all_conf) if len(all_conf) > 0 else 0
                mean_conf = np.mean(all_conf) if len(all_conf) > 0 else 0
                #print(f"[DEBUG Frame {frame_idx}] Raw model output shape: {arr.shape}, max_conf: {max_conf:.4f}, mean_conf: {mean_conf:.4f}, boxes_after_filter: {len(boxes_norm)}", flush=True)

        out_frame = bgr.copy()
        detections = []

        for box_norm, score in zip(boxes_norm, scores):
            x1_p = int(box_norm[0] * img_size); y1_p = int(box_norm[1] * img_size)
            x2_p = int(box_norm[2] * img_size); y2_p = int(box_norm[3] * img_size)
            x1 = int(max(0, (x1_p - left) / scale)); y1 = int(max(0, (y1_p - top) / scale))
            x2 = int(min(width, (x2_p - left) / scale)); y2 = int(min(height, (y2_p - top) / scale))
            w = max(1, x2 - x1); h = max(1, y2 - y1)
            area = w * h; aspect = float(w) / float(h) if h > 0 else 0
            if area < min_area or not (aspect_ratio_min <= aspect <= aspect_ratio_max): continue
            detections.append({'bbox': [x1, y1, x2, y2], 'score': float(score)})

        matches, unmatched_tracks_idx, unmatched_dets_idx = match_detections_to_tracks(
            detections, tracks, iou_thr=iou_thresh, dist_thr=0.35
        )
        # Umbral para intentar OCR en detecciones de alta confianza
        OCR_CANDIDATE_SCORE = 0.65
        # -------------------------------------------------------------------------------------
        # Actualizar Tracks
        # -------------------------------------------------------------------------------------
        
        # Actualizar Tracks (mejorado para preservar bbox y evitar drift)
        for (t_idx, d_idx) in matches:
            if t_idx >= len(tracks) or d_idx >= len(detections): continue
            tr = tracks[t_idx]; det = detections[d_idx]
            
            # --- LÓGICA DE ACTUALIZACIÓN DE TRACK SIMPLIFICADA ---
            # El track se mueve suavemente hacia la nueva detección.
            # La 'bbox' del track es la única fuente de verdad para su posición.
            
            # Usamos un factor de amortiguación (alpha) para suavizar el movimiento.
            alpha = float(dampening_factor)
            
            # Caja actual del track (la que se usa para matching y dibujo)
            tb = tr['bbox']
            # Caja de la nueva detección
            db = det['bbox']

            # --- PREDICCIÓN DE MOVIMIENTO (VELOCIDAD) ---
            # 1. Calcular la velocidad actual basada en el cambio de centro
            current_velocity = np.array(_center(db)) - np.array(_center(tb))
            
            # Actualización exponencial: la nueva bbox es una mezcla de la anterior y la nueva detección.
            tr['bbox'] = [
                    int(alpha * tb[0] + (1 - alpha) * db[0]),
                    int(alpha * tb[1] + (1 - alpha) * db[1]),
                    int(alpha * tb[2] + (1 - alpha) * db[2]),
                    int(alpha * tb[3] + (1 - alpha) * db[3]),
            ]

            # 2. Suavizar la velocidad del track
            # Esto estabiliza la predicción si hay detecciones con jitter.
            v_alpha = 0.7 # Más peso a la nueva velocidad
            old_velocity = tr.get('velocity', np.array([0.0, 0.0]))
            tr['velocity'] = v_alpha * current_velocity + (1 - v_alpha) * old_velocity
            
            # Guardamos la última detección real por si se necesita
            tr['last_detection_bbox'] = det['bbox'].copy()
            tr['current_score'] = det['score']  # Score para dibujar en tiempo real
            
            # Actualizar metadatos
            tr['missed'] = 0 # Reiniciar contador de fallos
            tr['seen'] = tr.get('seen', 0) + 1
            tr['age'] = tr.get('age', 0) + 1
            tr['last_frame'] = frame_idx
            tr['last_seen_conf'] = det['score']
            
            # Calidad basada en la bbox * score (usada para decidir best_frame)
            area_prod = (tr['bbox'][2] - tr['bbox'][0]) * (tr['bbox'][3] - tr['bbox'][1])
            quality = det['score'] * max(1, area_prod)
            is_new_best = False
            
            # Condición de Actualización:
            # 1. Si la calidad (score * area) mejora sustancialmente (x 1.05) O
            # 2. Si el score es mucho mejor (ej: 0.10) O
            # 3. Si se logra un OCR exitoso (si la placa ya está confirmada)
            
            current_ocr_fail = ('ERR' in tr.get('ocr_text', '')) or not tr.get('ocr_text')
            
            if (quality > tr.get('best_area', 0) * 1.05) or \
               (det['score'] > tr.get('best_score', 0) + 0.10) or \
               (current_ocr_fail and det['score'] > 0.70): # Intentar mejorar la captura si el OCR falló
                
                tr['best_area'] = quality
                tr['best_score'] = det['score']
                tr['best_frame'] = bgr.copy()
                #  Usamos la caja de la DETECCIÓN (det['bbox']) para la mejor foto, NO la suavizada
                tr['best_bbox'] = det['bbox'].copy() 
                is_new_best = True
            
            # Guardar en histórico de detecciones (agregar OCR si la detección es suficientemente buena)
            tr.setdefault('detection_history', [])
            
            # En lugar de ejecutar OCR aquí, lo enviamos a la cola si es un buen candidato
            if det['score'] >= OCR_CANDIDATE_SCORE:
                # Solo enviar a la cola si aún no tenemos un resultado para este track
                if tr['id'] not in ocr_results:
                    bx1, by1, bx2, by2 = [int(v) for v in det['bbox']]
                    crop = bgr[by1:by2, bx1:bx2].copy()
                    if crop.size > 0:
                        ocr_q.put({'track_id': tr['id'], 'image': crop})

            tr['detection_history'].append({
                'bbox': det['bbox'].copy(),
                'score': det['score'],
                'frame': bgr.copy(),
                'frame_idx': frame_idx,
                # El ocr_text se consultará del diccionario de resultados
            })
            
            # Actualizar el texto del track si el resultado del OCR ya está listo
            if tr['id'] in ocr_results:
                tr['ocr_text'] = ocr_results[tr['id']]

            # Limitar el histórico a últimas 30 detecciones por memoria
            if len(tr['detection_history']) > 30:
                tr['detection_history'] = tr['detection_history'][-30:]

            should_save = False
            
            # 1. Lógica de Confirmación (mismo criterio)
            if not tr.get('confirmed', False):
                if tr.get('seen', 0) >= confirm_frames and tr.get('best_score', 0) >= 0.35:
                    tr['confirmed'] = True
                    # Forzar un primer intento de OCR/guardado al confirmar
                    should_save = True 
            
            # 2. Re-OCR y Re-Guardado (Solo si está confirmado)
            if tr.get('confirmed', False) and is_new_best:
                # 2a. Guardar si el SCORE es mucho mejor que el guardado previamente
                score_improved_significantly = det['score'] > (tr.get('saved_score', 0) + 0.10)
                
                # 2b. O si el OCR falló o no se ha intentado
                ocr_needed = ('ERR' in tr.get('ocr_text', '')) or not tr.get('ocr_text')
                
                if score_improved_significantly or ocr_needed:
                    
                    # Enviar la mejor imagen hasta ahora a la cola de OCR
                    if tr['id'] not in ocr_results or ocr_needed:
                        best_bbox = tr.get('best_bbox', tr['bbox'])
                        bx1, by1, bx2, by2 = [int(v) for v in best_bbox]
                        crop = tr['best_frame'][by1:by2, bx1:bx2].copy()
                        if crop.size > 0:
                            ocr_q.put({'track_id': tr['id'], 'image': crop})

                    # Actualizar el texto del track si el resultado ya llegó
                    if tr['id'] in ocr_results:
                        new_ocr_text = ocr_results[tr['id']]
                        tr['ocr_text'] = new_ocr_text

                        # ----------------------------------------------------
                        # Lógica de Guardado DEFINITIVA
                        # ----------------------------------------------------
                        
                        ocr_is_successful = new_ocr_text and "ERR" not in new_ocr_text and "[OCR NO" not in new_ocr_text
                        
                        # Guardar si:
                        # a) Se logró un OCR exitoso (prioridad 1)
                        # b) O si el score es significativamente mejor y no había OCR antes (prioridad 2)
                        
                        if ocr_is_successful:
                            should_save = True
                        elif score_improved_significantly and not tr.get('saved_ocr_success', False):
                            should_save = True

            if should_save:
                # Asegurarse de que el ocr_text está actualizado antes de guardar
                if tr['id'] in ocr_results:
                    tr['ocr_text'] = ocr_results[tr['id']]

                best_bbox = tr.get('best_bbox', tr['bbox'])

                img_to_save = tr['best_frame'].copy()
                img_to_save = draw_labels(img_to_save, best_bbox, tr['id'], tr['best_score'], tr['ocr_text'])

                clean_text = tr['ocr_text'].replace('[OCR-', '').replace(']', '')
                suffix = clean_text if clean_text and "ERR" not in clean_text else "placa"
                img_name = f"{ts}_ID{tr['id']}_{suffix}.jpg"

                tr['saved_score'] = tr['best_score']
                saver_q.put({'type': 'image', 'path': os.path.join(UNIQUE_OUTPUT_DIR, img_name), 'image': img_to_save,
                             'track_id': tr['id'], 'score': tr.get('best_score', 0),
                             'ocr_text': tr['ocr_text'], 'timestamp': datetime.now().isoformat(),
                             'video_path': out_video_path, 'bbox': best_bbox})


        # Tracks no emparejados: mantener bbox y edad, incrementar misses
        for ti in unmatched_tracks_idx:
            if 0 <= ti < len(tracks):
                tracks[ti]['missed'] = tracks[ti].get('missed', 0) + 1
                tracks[ti]['age'] = tracks[ti].get('age', 0) + 1
                
                # --- PREDICCIÓN DE POSICIÓN PARA TRACKS PERDIDOS ---
                # En lugar de congelar la caja, la movemos según su última velocidad conocida.
                # Esto permite que el track "flote" y se re-enganche si la placa reaparece.
                velocity = tracks[ti].get('velocity', np.array([0.0, 0.0]))
                for i in range(4):
                    tracks[ti]['bbox'][i] += int(velocity[i % 2])

        # Detecciones nuevas -> crear track solo si cumplen requisitos mínimos
        for di in unmatched_dets_idx:
            if di < 0 or di >= len(detections): continue
            det = detections[di]
            # Filtrado: evitar tracks por detecciones muy pequeñas o muy débiles
            w = det['bbox'][2] - det['bbox'][0]; h = det['bbox'][3] - det['bbox'][1]
            area = max(1, w*h)
            if det['score'] < 0.25 or area < (min_area * 0.5):
                # Ignorar detecciones muy débiles / pequeñas (ruido)
                continue

            tnew = {
                'id': next_track_id,
                'bbox': det['bbox'].copy(),
                'last_detection_bbox': det['bbox'].copy(),
                'best_frame': bgr.copy(),
                'best_bbox': det['bbox'].copy(),
                'best_score': det['score'],
                'best_area': det['score'] * area,
                'missed': 0,
                'seen': 1 if det['score'] >= THRESHOLD else 0,
                'confirmed': False,
                'ocr_text': '',
                'last_frame': frame_idx,
                'saved_score': 0.0,
                'age': 1,
                'last_seen_conf': det['score']
            }
            # Incluir OCR inicial en el histórico si la detección es lo suficientemente buena
            if tnew['best_score'] >= OCR_CANDIDATE_SCORE:
                bx1, by1, bx2, by2 = [int(v) for v in det['bbox']]
                crop = bgr[by1:by2, bx1:bx2].copy()
                if crop.size > 0:
                    ocr_q.put({'track_id': tnew['id'], 'image': crop})
            
            # Consultar si el OCR ya terminó (poco probable, pero posible)
            initial_ocr = ocr_results.get(tnew['id'], '')
            tnew['ocr_text'] = initial_ocr
            
            tnew['detection_history'] = [
                {'bbox': det['bbox'].copy(), 'score': det['score'], 'frame': bgr.copy(), 'frame_idx': frame_idx, 'ocr_text': initial_ocr}
            ]
            tnew['current_score'] = det['score']
            tracks.append(tnew); next_track_id += 1

            # Confirmar solo si la detección es suficientemente clara y grande
            if (tnew['best_score'] >= 0.50) or (tnew['best_score'] >= 0.35 and (w*h) >= 800):
                tnew['confirmed'] = True
                # Guardamos una imagen inicial para metadata, pero NO sobrescribimos saved_score
                img_to_save = tnew['best_frame'].copy()
                img_to_save = draw_labels(img_to_save, tnew['best_bbox'], tnew['id'], tnew['best_score'], tnew['ocr_text'])
                clean_text = tnew['ocr_text'].replace('[OCR-', '').replace(']', '')
                suffix = clean_text if clean_text else "placa"
                saver_q.put({'type': 'image', 'path': os.path.join(UNIQUE_OUTPUT_DIR, f"{ts}_ID{tnew['id']}_{suffix}.jpg"),
                             'image': img_to_save, 'track_id': tnew['id'], 'score': tnew['best_score'],
                             'ocr_text': tnew['ocr_text'], 'timestamp': datetime.now().isoformat(),
                             'video_path': out_video_path, 'bbox': tnew['best_bbox']})
                # dejar saved_score en 0.0 para forzar un primer guardado más tarde si hay mejora

        # NO FILTRAR TRACKS: mantener todos para poder pintarlos
        # El filtrado final (eliminación) sucede al final después de escribir el frame
        # Esto garantiza que incluso tracks "viejos" se pinten si tienen detecciones recientes

        # --- DIBUJO PARA EL VIDEO DE SALIDA ---
        # Pintar TODOS los tracks activos (sin restricciones adicionales)
        drawn_count = 0
        for tr in tracks:
            # Usar score y bbox actualizado en tiempo real
            # Comprobar si hay un nuevo resultado de OCR para este track
            if tr['id'] in ocr_results:
                tr['ocr_text'] = ocr_results[tr['id']]

            display_bbox = tr['bbox'] # Usar siempre la bbox principal del track
            display_score = tr.get('current_score', tr.get('best_score', 0))
            color = (0, 255, 0) if tr.get('confirmed', False) else (0, 165, 255)
            ocr_disp = tr.get('ocr_text', '')
            
            # DIBUJAR SIEMPRE si el track está activo
            out_frame = draw_bbox_safe(out_frame, display_bbox, tr['id'], display_score, ocr_disp, color)
            drawn_count += 1
        
        #if frame_idx <= 5:  # Debug: solo primeros 5 frames
            #print(f"[Frame {frame_idx}] Tracks: {len(tracks)}, Dibujados: {drawn_count}, Detecciones: {len(detections)}")
        
        # DESPUÉS de dibujar: eliminar tracks expirados
        preserved = []
        for t in tracks:
            mm = t.get('missed', 0)
            limit = int(max_missed * 2.0) if t.get('confirmed', False) else max_missed
            if mm <= limit and t.get('age', 0) <= 0x7fffffff:
                preserved.append(t)
        tracks = preserved
        writer.write(out_frame)
        if display:
            cv2.imshow("Detección", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        pbar.update(1)

    cap.release(); writer.release(); pbar.close()
    if display: cv2.destroyAllWindows()
    saver_q.put(stop_token); saver_thread.join(timeout=15)
    ocr_q.put(ocr_stop_token); ocr_thread.join(timeout=15)
    # Esperar a que la cola de OCR se vacíe antes del guardado final
    ocr_q.join()
    
    # Guardar la mejor detección de cada track desde el histórico
    print("📊 Procesando histórico de detecciones para guardar las mejores...")
    for tr in tracks:
        if not tr.get('confirmed', False) or tr.get('age', 0) < 2:
            continue
    
        best_frame, best_bbox, best_score, best_ocr = select_best_detection_from_history(tr)

        # Asegurarse de que el texto final del OCR esté actualizado desde el diccionario
        if tr['id'] in ocr_results:
            best_ocr = ocr_results[tr['id']]
    
        if best_frame is None:
            continue
    
        # Dibujar etiqueta
        img_to_save = best_frame.copy()
        img_to_save = draw_labels(img_to_save, best_bbox, tr['id'], best_score, best_ocr)
    
        # Nombre del archivo con OCR
        clean_text = best_ocr.replace('[OCR-', '').replace(']', '').replace('[OCR', '').replace('NO', '').replace('INSTALADO', '')
        suffix = clean_text.strip() if clean_text and "ERR" not in clean_text else "placa"
        img_name = f"{ts}_FINAL_ID{tr['id']}_{suffix}.jpg"
    
        saver_q.put({
            'type': 'image',
            'path': os.path.join(UNIQUE_OUTPUT_DIR, img_name),
            'image': img_to_save,
            'track_id': tr['id'],
            'score': best_score,
            'ocr_text': best_ocr,
            'timestamp': datetime.now().isoformat(),
            'video_path': out_video_path,
            'bbox': best_bbox
        })

    print(f"✔ Resultados guardados en: {UNIQUE_OUTPUT_DIR}")
    return out_video_path

def main():
    parser = argparse.ArgumentParser(description="Inferir video con etiquetas completas (ID, Confianza, Placa)")
    parser.add_argument("--video", required=False, default=None, help="Video entrada o 'input_feed' para procesar la carpeta 03_production/input_feed/")
    parser.add_argument("--model", default=MODEL_PATH, help="Modelo Keras")
    parser.add_argument("--display", action="store_true", help="Ver ventana")
    parser.add_argument("--confirm_frames", type=int, default=3)
    parser.add_argument("--iou_thresh", type=float, default=0.30)
    parser.add_argument("--min_area", type=int, default=800)
    parser.add_argument("--max_missed", type=int, default=10)
    parser.add_argument("--dampening_factor", type=float, default=0.8)
    parser.add_argument("--ocr_padding_ratio", type=float, default=0.15)
    args = parser.parse_args()

    model = load_model_safe(args.model)

    # Si no se pasa --video o se pasa 'input_feed', procesar todos los videos en 03_production/input_feed/
    if args.video is None or args.video == 'input_feed':
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        input_dir = os.path.join(repo_root, '03_production', 'input_feed')
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"No se encontró la carpeta de entrada: {input_dir}")

        # Extensiones de video soportadas
        exts = ('.mp4', '.avi', '.mov', '.mkv')
        videos = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(exts)]
        if not videos:
            raise FileNotFoundError(f"No se encontraron videos en: {input_dir}")

        for vid in videos:
            print(f"Procesando video: {vid}")
            try:
                process_video(model, vid, img_size=IMG_SIZE[0], display=args.display,
                              max_missed=args.max_missed, iou_thresh=args.iou_thresh,
                              confirm_frames=args.confirm_frames, min_area=args.min_area,
                              dampening_factor=args.dampening_factor, ocr_padding_ratio=args.ocr_padding_ratio)
            except Exception as e:
                print(f"Error procesando {vid}: {e}")
    else:
        process_video(model, args.video, img_size=IMG_SIZE[0], display=args.display,
                      max_missed=args.max_missed, iou_thresh=args.iou_thresh,
                      confirm_frames=args.confirm_frames, min_area=args.min_area,
                      dampening_factor=args.dampening_factor, ocr_padding_ratio=args.ocr_padding_ratio)

if __name__ == "__main__":
    main()