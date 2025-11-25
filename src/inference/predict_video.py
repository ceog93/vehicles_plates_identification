# src/inference/predict_video.py
# Versión final corregida y optimizada
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
# Reducir la actividad de autoajuste de XLA / cuDNN que puede ser inestable en algunas GPUs
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# OCR - Importación segura
try:
    import easyocr
    # Inicializamos el lector solo una vez
    READER = easyocr.Reader(['es', 'en'], gpu=False)
except Exception:
    print("Advertencia: EasyOCR no está instalado o falló la inicialización. OCR deshabilitado.")
    READER = None

# Algoritmo Húngaro para tracking
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_HUNGARIAN = True
except Exception:
    linear_sum_assignment = None
    _HAS_HUNGARIAN = False

# Importaciones del proyecto (Asegurar que estos módulos existan)
from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR, THRESHOLD, ROOT_MODEL_DIR, LATEST_MODEL_PATH
from src.utils.mpd_utils import resize_pad, nms_numpy
from src.models.efficient_detector_multi_placa import NUM_CLASSES, yolo_ciou_loss

# ---------------------------
# FUNCIONES AUXILIARES: DECODIFICACIÓN Y CARGA
# ---------------------------

def process_predictions(output_tensor, img_size=IMG_SIZE[0], confidence_threshold=THRESHOLD):
    """
    Decodifica la salida del modelo (estilo YOLO) a cajas delimitadoras normalizadas.
    """
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
                
                # Coordenadas locales de la celda
                cx_local, cy_local = float(cell[1]), float(cell[2])
                w_norm, h_norm = float(cell[3]), float(cell[4])
                
                # Convertir a coordenadas normalizadas globales
                cx_norm = (j + cx_local) / gw
                cy_norm = (i + cy_local) / gh
                
                xmin = cx_norm - (w_norm / 2.0)
                ymin = cy_norm - (h_norm / 2.0)
                xmax = cx_norm + (w_norm / 2.0)
                ymax = cy_norm + (h_norm / 2.0)
                
                final_boxes_norm.append([xmin, ymin, xmax, ymax])
                final_scores.append(conf)

    if len(final_boxes_norm) > 0:
        # Aplicar Non-Maximum Suppression (NMS)
        selected = nms_numpy(final_boxes_norm, final_scores, iou_thresh=0.45, score_thresh=0.0)
        final_boxes = [final_boxes_norm[i] for i in selected]
        final_scores = [final_scores[i] for i in selected]
    else:
        final_boxes, final_scores = [], []

    return final_boxes, final_scores


def find_latest_model_in_models_dir():
    """Busca automáticamente el modelo más reciente en el directorio de modelos."""
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
    """Carga el modelo Keras manejando errores y rutas alternativas."""
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
            # Importante: pasar custom_objects para la función de pérdida si se guardó con ella
            model = tf.keras.models.load_model(p, custom_objects={'yolo_ciou_loss': yolo_ciou_loss}, compile=False)
            print(f"✔ Modelo cargado desde: {p}")
            return model
        except Exception as e:
            last_err = e
            print(f"Intento de carga falló para {p}: {e}")
    raise FileNotFoundError(f"No se encontró modelo válido. Intentos: {candidates}. Ultimo error: {last_err}")


# ---------------------------
# OCR & UTILS
# ---------------------------

def run_ocr(cropped_img):
    """Ejecuta EasyOCR sobre un recorte de imagen."""
    if READER is None:
        return "[OCR NO INSTALADO]"
    try:
        # Convertir BGR (OpenCV) a RGB para EasyOCR
        if len(cropped_img.shape) == 3 and cropped_img.shape[2] == 3:
            rgb_crop = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_crop = cropped_img
        
        # Lectura restringida a caracteres alfanuméricos
        results = READER.readtext(rgb_crop, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', detail=0, paragraph=True)
        if results:
            text = "".join(results).replace(" ", "").replace("-", "")
            return text.upper()
        return ""
    except Exception as e:
        return f"[OCR ERROR: {e}]"


def _iou_numpy(boxA, boxB):
    """Calcula la Intersección sobre Unión (IoU) entre dos cajas."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    denom = areaA + areaB - interArea
    return interArea / denom if denom > 0 else 0.0


def _center(box):
    """Calcula el centro de una caja [x1, y1, x2, y2]."""
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def match_detections_to_tracks(detections, tracks, iou_thr=0.30, dist_thr=0.35):
    """
    Asocia detecciones nuevas con tracks existentes usando Algoritmo Húngaro o Greedy.
    Combina IoU y Distancia Euclidiana para robustez.
    """
    if len(tracks) == 0:
        return [], [], list(range(len(detections)))
    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    T = len(tracks)
    D = len(detections)
    cost = np.ones((T, D), dtype=np.float32)

    for ti, tr in enumerate(tracks):
        box_t = tr['bbox']
        ct = _center(box_t)
        # Referencia de normalización basada en el tamaño de la caja (diagonal)
        ref = max(1.0, np.linalg.norm(np.array([box_t[2] - box_t[0], box_t[3] - box_t[1]])))
        
        for di, det in enumerate(detections):
            box_d = det['bbox']
            cd = _center(box_d)
            
            iou_v = _iou_numpy(box_t, box_d)
            dist = np.linalg.norm(ct - cd)
            
            # Distancia normalizada e invertida (1.0 es cerca, 0.0 es lejos)
            dist_norm = 1.0 - (dist / (ref + 1e-6))
            dist_norm = np.clip(dist_norm, 0.0, 1.0)
            
            # Score combinado: Damos peso al IoU y a la cercanía
            # Higher score = mejor coincidencia
            score = 0.70 * iou_v + 0.30 * dist_norm 
            
            # Matriz de costo (Húngaro minimiza costo, por eso 1 - score)
            cost[ti, di] = 1.0 - score

    cost_matrix = cost.copy()
    matches = []
    matched_t = set()
    matched_d = set()

    # 1. Intentar Algoritmo Húngaro
    if _HAS_HUNGARIAN and linear_sum_assignment is not None:
        try:
            t_idx, d_idx = linear_sum_assignment(cost_matrix)
            for ti, di in zip(t_idx, d_idx):
                # Validar con un umbral mínimo de IoU para evitar saltos absurdos
                if _iou_numpy(tracks[ti]['bbox'], detections[di]['bbox']) >= iou_thr:
                    matches.append((ti, di))
                    matched_t.add(ti)
                    matched_d.add(di)
        except Exception as e:
            print("⚠ Hungarian falló:", e)

    # 2. Fallback Greedy (Voraz) para lo que sobró o si falló Húngaro
    if len(matches) == 0:
        flat = []
        for ti in range(T):
            for di in range(D):
                flat.append((cost_matrix[ti, di], ti, di))
        # Ordenar por menor costo (mejor match)
        flat.sort(key=lambda x: x[0])
        
        used_t = set()
        used_d = set()
        for c, ti, di in flat:
            if ti in used_t or di in used_d: continue
            iou_v = _iou_numpy(tracks[ti]['bbox'], detections[di]['bbox'])
            
            if iou_v >= iou_thr:
                matches.append((ti, di))
                used_t.add(ti)
                used_d.add(di)
        
        matched_t.update(used_t)
        matched_d.update(used_d)

    unmatched_t = [i for i in range(T) if i not in matched_t]
    unmatched_d = [j for j in range(D) if j not in matched_d]

    return matches, unmatched_t, unmatched_d


# ---------------------------
# HILO DE GUARDADO (SAVER THREAD)
# ---------------------------
def start_saver_thread(q, csv_path):
    """
    Hilo dedicado a guardar imágenes y escribir el CSV para no bloquear el procesamiento de video.
    Recibe diccionarios 'job' con la imagen YA preparada (con cuadros dibujados si es necesario).
    """
    stop_token = object()

    def worker():
        rows = {}  # Cache de filas por track_id para evitar duplicados en el CSV final
        
        # Leer CSV existente si hay (para continuar tracks si se reanuda)
        if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
            try:
                with open(csv_path, 'r', newline='') as cf:
                    r = csv.DictReader(cf)
                    for rec in r:
                        tid = rec.get('track_id', '')
                        if tid:
                            rows[tid] = rec
            except Exception as e:
                print("Advertencia: fallo leyendo CSV existente:", e)

        fieldnames = ['track_id', 'filepath', 'score', 'ocr_text', 'timestamp', 'video_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']

        def write_csv():
            try:
                with open(csv_path, 'w', newline='') as cf:
                    writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    writer.writeheader()
                    for v in rows.values():
                        writer.writerow(v)
            except Exception as e:
                print("Error escribiendo CSV:", e)

        while True:
            job = q.get()
            if job is stop_token:
                try:
                    write_csv()
                except Exception as e:
                    print("Error final escribiendo CSV:", e)
                q.task_done()
                break

            try:
                if job.get('type') == 'image':
                    # Guardar imagen en disco
                    path = job['path']
                    img_data = job['image']
                    saved = False
                    try:
                        # cv2.imwrite guarda en BGR, aseguramos que img_data sea BGR
                        saved = cv2.imwrite(path, img_data)
                    except Exception as e:
                        print(f"Error guardando imagen {path}: {e}")
                        saved = False
                    
                    if saved:
                        # Actualizar registro en memoria y CSV
                        tid = str(job.get('track_id', ''))
                        bbox = job.get('bbox', [0,0,0,0])
                        rows[tid] = {
                            'track_id': tid,
                            'filepath': path,
                            'score': f"{job.get('score', 0):.4f}",
                            'ocr_text': job.get('ocr_text', ''),
                            'timestamp': job.get('timestamp', ''),
                            'video_path': job.get('video_path', ''),
                            'bbox_x1': str(bbox[0]),
                            'bbox_y1': str(bbox[1]),
                            'bbox_x2': str(bbox[2]),
                            'bbox_y2': str(bbox[3]),
                        }
                        write_csv() # Escribir frecuentemente para evitar perdida de datos
            except Exception:
                print("Excepción en Saver Thread:", traceback.format_exc())
            finally:
                q.task_done()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t, stop_token


# ---------------------------
# PROCESO PRINCIPAL DE VIDEO
# ---------------------------
def process_video(model, video_path, out_video_path=None, img_size=IMG_SIZE[0], display=False,
                  max_missed=5, iou_thresh=0.45, confirm_frames=3, min_area=800,
                  aspect_ratio_min=1.8, aspect_ratio_max=8.0, dampening_factor=0.85,
                  ocr_padding_ratio=0.1):
    """
    Procesa un video completo: detección, tracking, OCR y generación de video de salida con anotaciones.
    """
    # Abrir captura de video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Directorio de salida único basado en timestamp
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    base = os.path.basename(video_path)
    UNIQUE_OUTPUT_DIR = os.path.join(OUTPUT_FEED_DIR, ts)
    os.makedirs(UNIQUE_OUTPUT_DIR, exist_ok=True)

    if out_video_path is None:
        out_video_path = os.path.join(UNIQUE_OUTPUT_DIR, f"det_{base}.mp4")

    # Configurar escritor de video (MP4V)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    pbar = tqdm(total=total_frames, desc="Procesando frames")

    # Warmup del modelo (predicción en vacío para cargar en GPU)
    try:
        dummy = np.zeros((1, img_size, img_size, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)
    except Exception:
        pass

    # Iniciar hilo de guardado
    saver_q = queue.Queue()
    metadata_csv = os.path.join(UNIQUE_OUTPUT_DIR, 'video_saved_metadata.csv')
    saver_thread, stop_token = start_saver_thread(saver_q, metadata_csv)

    # Estado del Tracker
    # Lista de diccionarios. Cada track mantiene su estado: id, bbox, best_score, ocr_text, etc.
    tracks = []
    next_track_id = 1

    # Ventana de visualización
    if display:
        try:
            cv2.namedWindow("Detección", cv2.WINDOW_NORMAL)
            max_w = min(1280, width); max_h = min(720, height)
            cv2.resizeWindow("Detección", max_w, max_h)
        except Exception:
            pass

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Preparar imagen para inferencia
        bgr = frame.copy() # OpenCV usa BGR
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_pad, scale, top, left = resize_pad(rgb, img_size)
        inp = (img_pad.astype(np.float32) / 255.0)[None, ...]

        # Predicción
        pred_tensor = model.predict(inp, verbose=0)
        boxes_norm, scores = process_predictions(pred_tensor, img_size=img_size)

        out_frame = bgr.copy() # Frame sobre el que dibujaremos para el video
        detections = []

        # Filtrado inicial y desnormalización de coordenadas
        for box_norm, score in zip(boxes_norm, scores):
            x1_p = int(box_norm[0] * img_size); y1_p = int(box_norm[1] * img_size)
            x2_p = int(box_norm[2] * img_size); y2_p = int(box_norm[3] * img_size)
            
            # Mapeo inverso a tamaño original
            x1 = int(max(0, (x1_p - left) / scale)); y1 = int(max(0, (y1_p - top) / scale))
            x2 = int(min(width, (x2_p - left) / scale)); y2 = int(min(height, (y2_p - top) / scale))
            
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            area = w * h
            aspect = float(w) / float(h) if h > 0 else 0
            
            # Filtrar por área mínima y relación de aspecto (placas suelen ser rectangulares)
            if area < min_area or not (aspect_ratio_min <= aspect <= aspect_ratio_max):
                continue
            
            detections.append({'bbox': [x1, y1, x2, y2], 'score': float(score)})

        # MATCHING: Asociar detecciones a tracks existentes
        matches, unmatched_tracks_idx, unmatched_dets_idx = match_detections_to_tracks(
            detections, tracks, iou_thr=iou_thresh, dist_thr=0.35
        )

        # --- 1. ACTUALIZAR TRACKS EXISTENTES ---
        for (t_idx, d_idx) in matches:
            if t_idx >= len(tracks) or d_idx >= len(detections):
                continue
            
            tr = tracks[t_idx]
            det = detections[d_idx]
            
            # Suavizado exponencial (EMA) para estabilizar la caja
            prev_bbox = tr['bbox']
            det_bbox = det['bbox']
            alpha = dampening_factor
            
            smoothed = [
                int(alpha * prev_bbox[0] + (1 - alpha) * det_bbox[0]),
                int(alpha * prev_bbox[1] + (1 - alpha) * det_bbox[1]),
                int(alpha * prev_bbox[2] + (1 - alpha) * det_bbox[2]),
                int(alpha * prev_bbox[3] + (1 - alpha) * det_bbox[3]),
            ]
            
            tr['bbox'] = smoothed
            tr['missed'] = 0
            tr['seen'] = tr.get('seen', 0) + 1
            tr['last_frame'] = frame_idx
            
            # Calcular calidad para decidir si es el "mejor frame" (para OCR)
            w_s = max(1, smoothed[2] - smoothed[0])
            h_s = max(1, smoothed[3] - smoothed[1])
            area_s = w_s * h_s
            quality = det['score'] * area_s
            
            # Si encontramos una vista mejor de la placa, actualizamos la "mejor toma"
            is_new_best = False
            if quality > tr.get('best_area', 0):
                tr['best_area'] = quality
                tr['best_score'] = det['score']
                tr['best_frame'] = bgr.copy() # Guardamos el frame limpio original
                tr['best_bbox'] = smoothed.copy()
                is_new_best = True

            # Lógica de Confirmación y Guardado
            should_save = False
            
            # A) Si pasa de no confirmado a confirmado
            if not tr.get('confirmed', False):
                if tr.get('seen', 0) >= confirm_frames and tr.get('best_score', 0) >= 0.35 and area_s >= 500:
                    tr['confirmed'] = True
                    should_save = True # Guardar primera vez
            
            # B) Si ya estaba confirmado y la calidad mejoró significativamente, guardamos actualización
            elif tr.get('confirmed', False) and is_new_best:
                # Opcional: solo guardar si la mejora es sustancial para no llenar el disco
                if det['score'] > (tr.get('saved_score', 0) + 0.05): 
                    should_save = True

            # Ejecutar lógica de guardado (OCR + Imagen con cajas)
            if should_save:
                # Recorte para OCR
                best_bbox = tr.get('best_bbox', tr['bbox'])
                bx1, by1, bx2, by2 = [int(v) for v in best_bbox]
                
                # Padding seguro
                pad_x = int((bx2 - bx1) * ocr_padding_ratio)
                pad_y = int((by2 - by1) * ocr_padding_ratio)
                cx1 = max(0, bx1 - pad_x); cy1 = max(0, by1 - pad_y)
                cx2 = min(width, bx2 + pad_x); cy2 = min(height, by2 + pad_y)
                
                crop = tr['best_frame'][cy1:cy2, cx1:cx2].copy()
                
                if crop.size != 0:
                    try:
                        # Preprocesamiento básico OCR
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        gray = cv2.bilateralFilter(gray, 5, 75, 75)
                        gray = cv2.equalizeHist(gray)
                        gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)
                        tr['ocr_text'] = run_ocr(gray)
                    except Exception as e:
                        tr['ocr_text'] = f"ERR"
                
                # Preparar imagen FINAL para guardar (Dibujar sobre ella)
                img_to_save = tr['best_frame'].copy()
                
                # 1. Dibujar Bounding Box
                cv2.rectangle(img_to_save, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                
                # 2. Dibujar Etiqueta OCR
                ocr_txt = tr.get('ocr_text', '')
                if ocr_txt:
                    label_pos = (bx1, max(0, by1 - 10))
                    cv2.putText(img_to_save, ocr_txt, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 3. Enviar a cola de guardado
                clean_text = ocr_txt.replace('[OCR-', '').replace(']', '')
                suffix = clean_text if clean_text and "ERR" not in clean_text else "placa"
                img_name = f"{ts}_ID{tr['id']}_{suffix}.jpg"
                img_path = os.path.join(UNIQUE_OUTPUT_DIR, img_name)
                
                tr['saved_score'] = tr['best_score'] # Actualizar score guardado

                saver_q.put({
                    'type': 'image', 
                    'path': img_path, 
                    'image': img_to_save, # Imagen CON dibujos
                    'track_id': tr['id'], 
                    'score': tr.get('best_score', 0),
                    'ocr_text': ocr_txt, 
                    'timestamp': datetime.now().isoformat(),
                    'video_path': out_video_path, 
                    'bbox': best_bbox
                })

        # --- 2. GESTIONAR TRACKS PERDIDOS ---
        for ti in unmatched_tracks_idx:
            if 0 <= ti < len(tracks):
                tracks[ti]['missed'] = tracks[ti].get('missed', 0) + 1

        # --- 3. CREAR NUEVOS TRACKS ---
        for di in unmatched_dets_idx:
            if di < 0 or di >= len(detections):
                continue
            det = detections[di]
            
            # Crear nuevo objeto track
            tnew = {
                'id': next_track_id,
                'bbox': det['bbox'],
                'best_frame': bgr.copy(),
                'best_bbox': det['bbox'].copy(),
                'best_score': det['score'],
                'best_area': det['score'] * max(1, (det['bbox'][2]-det['bbox'][0]) * (det['bbox'][3]-det['bbox'][1])),
                'missed': 0,
                'seen': 1 if det['score'] >= THRESHOLD else 0,
                'confirmed': False,
                'ocr_text': '',
                'last_frame': frame_idx,
                'saved_score': 0.0
            }
            tracks.append(tnew)
            next_track_id += 1
            
            # Confirmación inmediata si la detección es muy buena (High Confidence)
            w = max(1, tnew['bbox'][2] - tnew['bbox'][0])
            h = max(1, tnew['bbox'][3] - tnew['bbox'][1])
            
            if tnew['best_score'] >= 0.80 and (w * h) >= 1200:
                tnew['confirmed'] = True
                
                # Ejecutar OCR inmediato
                bx1, by1, bx2, by2 = [int(v) for v in tnew['best_bbox']]
                pad_x = int((bx2 - bx1) * ocr_padding_ratio)
                pad_y = int((by2 - by1) * ocr_padding_ratio)
                cx1 = max(0, bx1 - pad_x); cy1 = max(0, by1 - pad_y)
                cx2 = min(width, bx2 + pad_x); cy2 = min(height, by2 + pad_y)
                
                crop = tnew['best_frame'][cy1:cy2, cx1:cx2].copy()
                if crop.size != 0:
                    try:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        gray = cv2.bilateralFilter(gray, 5, 75, 75)
                        gray = cv2.equalizeHist(gray)
                        gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)
                        tnew['ocr_text'] = run_ocr(gray)
                    except:
                        pass
                
                # Guardar inmediato
                img_to_save = tnew['best_frame'].copy()
                cv2.rectangle(img_to_save, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                if tnew['ocr_text']:
                    cv2.putText(img_to_save, tnew['ocr_text'], (bx1, max(0, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                clean_text = tnew['ocr_text'].replace('[OCR-', '').replace(']', '')
                suffix = clean_text if clean_text else "placa"
                img_name = f"{ts}_ID{tnew['id']}_{suffix}.jpg"
                img_path = os.path.join(UNIQUE_OUTPUT_DIR, img_name)
                
                tnew['saved_score'] = tnew['best_score']

                saver_q.put({
                    'type': 'image', 'path': img_path, 'image': img_to_save,
                    'track_id': tnew['id'], 'score': tnew['best_score'],
                    'ocr_text': tnew['ocr_text'], 'timestamp': datetime.now().isoformat(),
                    'video_path': out_video_path, 'bbox': tnew['best_bbox']
                })

        # --- 4. LIMPIEZA Y DIBUJADO EN VIDEO ---
        # Eliminar tracks perdidos hace mucho tiempo
        tracks = [t for t in tracks if t.get('missed', 0) <= max_missed]

        # Dibujar tracks en el frame actual (para el video de salida)
        for tr in tracks:
            # Solo dibujar si está activo recientemente
            if tr['missed'] > 2:
                continue
                
            bx = tr['bbox']
            # Verde si confirmado, Rojo si no
            color = (0, 255, 0) if tr.get('confirmed', False) else (0, 0, 255)
            
            try:
                cv2.rectangle(out_frame, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
            except:
                pass
            
            # Texto superior: ID y Score
            label_top = f"ID:{tr['id']} ({tr.get('best_score', 0)*100:.0f}%)"
            try:
                cv2.putText(out_frame, label_top, (bx[0], max(0, bx[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except:
                pass
            
            # Texto inferior: OCR (persistente del track)
            ocr_disp = tr.get('ocr_text', '')
            if ocr_disp and tr.get('confirmed', False):
                try:
                    # Fondo negro para el texto OCR para mejor legibilidad
                    (tw, th), _ = cv2.getTextSize(ocr_disp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(out_frame, (bx[0], bx[3] + 2), (bx[0] + tw, bx[3] + 18), (0,0,0), -1)
                    cv2.putText(out_frame, ocr_disp, (bx[0], min(height - 4, bx[3] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except:
                    pass

        # Escribir frame en video
        try:
            writer.write(out_frame)
        except Exception as e:
            print("Error al escribir frame:", e)
        
        # Visualización
        if display:
            try:
                cv2.imshow("Detección", out_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass

        pbar.update(1)

    # --- FINALIZACIÓN ---
    cap.release()
    writer.release()
    pbar.close()
    
    try:
        if display:
            cv2.destroyAllWindows()
    except:
        pass

    # Detener hilo de guardado y esperar a que termine tareas pendientes
    saver_q.put(stop_token)
    saver_thread.join(timeout=15)

    print(f"✔ Resultados guardados en: {UNIQUE_OUTPUT_DIR}")
    return out_video_path


# ---------------------------
# CLI (INTERFAZ DE LÍNEA DE COMANDO)
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Inferir un video con detector de placas")
    parser.add_argument("--video", required=True, help="Ruta a archivo de video")
    parser.add_argument("--model", default=MODEL_PATH, help="Ruta al modelo Keras")
    parser.add_argument("--display", action="store_true", help="Mostrar video en pantalla")
    
    # Parámetros ajustados para mejorar seguimiento y evitar tracks duplicados
    parser.add_argument("--confirm_frames", type=int, default=3, help="Frames consecutivos para confirmar un track")
    parser.add_argument("--iou_thresh", type=float, default=0.40, help="Umbral IoU para matching (más alto = más estricto)")
    parser.add_argument("--min_area", type=int, default=800, help="Área mínima para aceptar detección")
    parser.add_argument("--aspect_ratio_min", type=float, default=1.8)
    parser.add_argument("--aspect_ratio_max", type=float, default=8.0)
    parser.add_argument("--max_missed", type=int, default=10, help="Frames perdidos permitidos antes de perder el track")
    parser.add_argument("--dampening_factor", type=float, default=0.75, help="Suavizado de bbox (0.0 a 1.0)")
    parser.add_argument("--ocr_padding_ratio", type=float, default=0.1)

    args = parser.parse_args()

    model = load_model_safe(args.model)
    out = process_video(
        model,
        args.video,
        img_size=IMG_SIZE[0],
        display=args.display,
        max_missed=args.max_missed,
        iou_thresh=args.iou_thresh,
        confirm_frames=args.confirm_frames,
        min_area=args.min_area,
        aspect_ratio_min=args.aspect_ratio_min,
        aspect_ratio_max=args.aspect_ratio_max,
        dampening_factor=args.dampening_factor,
        ocr_padding_ratio=args.ocr_padding_ratio,
    )
    print("Hecho. Salida:", out)


if __name__ == "__main__":
    main()