import os # Importación complementada
import argparse
from datetime import datetime
import csv
import queue
import threading
import time 

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Importación de la librería de OCR
try:
    import easyocr
    # Inicializar el lector de EasyOCR una sola vez al inicio del módulo
    # Utilizar 'es' (español) y 'en' (inglés) para una cobertura más amplia de caracteres.
    READER = easyocr.Reader(['es', 'en'], gpu=False)
except ImportError:
    print("Advertencia: EasyOCR no está instalado. El reconocimiento de caracteres estará deshabilitado.")
    READER = None

# Importaciones de configuración y utilidades (asumo que estas rutas son correctas)
# Asegúrate de que las rutas y constantes estén disponibles
from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR, THRESHOLD, ROOT_MODEL_DIR, LATEST_MODEL_PATH
from src.utils.mpd_utils import resize_pad, nms_numpy
from src.models.efficient_detector_multi_placa import NUM_CLASSES, yolo_ciou_loss


def process_predictions(output_tensor, img_size=IMG_SIZE[0], confidence_threshold=THRESHOLD):
    """
    Decodifica la salida del modelo Yolo-like a bounding boxes y scores.
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
                confidence = float(cell[0])
                
                if confidence < confidence_threshold:
                    continue
                
                # Coordenadas locales
                cx_local = float(cell[1]); cy_local = float(cell[2])
                w_norm = float(cell[3]); h_norm = float(cell[4])
                
                # Normalizar coordenadas: del grid (0..1)
                cx_norm = (j + cx_local) / gw
                cy_norm = (i + cy_local) / gh
                
                # Convertir a (xmin, ymin, xmax, ymax) normalizado (0..1)
                xmin_norm = cx_norm - (w_norm / 2); ymin_norm = cy_norm - (h_norm / 2)
                xmax_norm = cx_norm + (w_norm / 2); ymax_norm = cy_norm + (h_norm / 2)
                
                final_boxes_norm.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
                final_scores.append(confidence)
                
    if len(final_boxes_norm) > 0:
        # Aplicar Non-Maximum Suppression (NMS)
        selected = nms_numpy(final_boxes_norm, final_scores, iou_thresh=0.45, score_thresh=0.0)
        final_boxes = [final_boxes_norm[i] for i in selected]
        final_scores = [final_scores[i] for i in selected]
    else:
        final_boxes = []
        final_scores = []
        
    return final_boxes, final_scores


# --- Funciones de Carga del Modelo (No requieren cambios funcionales) ---
def find_latest_model_in_models_dir():
    # ... (código existente) ...
    if not os.path.isdir(ROOT_MODEL_DIR):
        return None
    subdirs = [d for d in os.listdir(ROOT_MODEL_DIR) if os.path.isdir(os.path.join(ROOT_MODEL_DIR, d))]
    if not subdirs:
        return None
    
    # Ordenar las carpetas por nombre (asumiendo que son timstamps o épocas)
    subdirs.sort(reverse=True)
    
    for latest in subdirs:
        candidate_dir = os.path.join(ROOT_MODEL_DIR, latest)
        
        # Buscar nombres de archivo comunes para el modelo
        for name in ("detector_model.keras", "detector.keras", "detector.h5", "detector"):
            p = os.path.join(candidate_dir, name)
            if os.path.exists(p):
                return p
        
        # Buscar cualquier archivo .keras o .h5 dentro
        for f in os.listdir(candidate_dir):
            if f.endswith('.keras') or f.endswith('.h5'):
                return os.path.join(candidate_dir, f)
                
    return None


def load_model_safe(model_path=None):
    # ... (código existente) ...
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
            # yolo_ciou_loss es necesario para cargar modelos entrenados con esa función de pérdida
            model = tf.keras.models.load_model(p, custom_objects={'yolo_ciou_loss': yolo_ciou_loss}, compile=False)
            print(f"✔ Modelo cargado desde: {p}")
            return model
        except Exception as e:
            last_err = e
            print(f"Intento de carga falló para {p}: {e}")
            
    raise FileNotFoundError(f"❌ No se encontró modelo válido. Intentos: {candidates}. Ultimo error: {last_err}")


# --- Función de OCR Mejorada ---
def run_ocr(cropped_img):
    """Ejecuta EasyOCR en una imagen recortada y devuelve el texto más confiable."""
    if READER is None:
        return "[OCR NO INSTALADO]"
    
    try:
        # Opciones:
        # - allowlist: Limitar a caracteres alfanuméricos de placas (sin Ñ o caracteres regionales complejos por defecto).
        # - detail=0: Devuelve solo el texto reconocido (sin bounding boxes de caracteres).
        # - paragraph=True: Junta los resultados en una sola línea si están cerca.
        results = READER.readtext(cropped_img, 
                                 allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 
                                 detail=0, 
                                 paragraph=True)
        
        if results:
            # Limpieza: Unir el texto, quitar espacios y guiones que EasyOCR pudiera agregar.
            ocr_text = "".join(results).replace(" ", "").replace("-", "")
            return ocr_text.upper() # Asegurar mayúsculas
        else:
            return ""
            
    except Exception as e:
        # Capturar errores del OCR (e.g., fallas de CUDA, problemas de imagen)
        return f"[OCR ERROR: {e}]"

# --- Función Principal de Procesamiento ---
def process_video(model, video_path, out_video_path=None, img_size=IMG_SIZE[0], display=False,
                  # 🔥 AJUSTES PARA MAYOR ESTABILIDAD DEL TRACKING Y OCR 🔥
                  max_missed=10, iou_thresh=0.30, confirm_frames=1, min_area=800,
                  aspect_ratio_min=1.8, aspect_ratio_max=8.0,
                  dampening_factor=0.75, # 💡 NUEVO: Factor de suavizado para el Bounding Box (0.0=det, 1.0=prev)
                  ocr_padding_ratio=0.1 # 💡 NUEVO: Ratio de margen para el recorte de OCR (10%)
                  ):

    # ... (Inicio de process_video, VideoCapture, VideoWriter, Warmup, Saver Thread - SIN CAMBIOS) ...
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ No se pudo abrir el video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Inicialización de VideoWriter ---
    if out_video_path is None:
        base = os.path.basename(video_path)
        ts = datetime.now().strftime('%Y%m%d%H%M%S')
        out_video_path = os.path.join(OUTPUT_FEED_DIR, f"{ts}_det_{base}")
    if not out_video_path.lower().endswith('.mp4'):
        out_video_path = out_video_path + '.mp4'
        
    os.makedirs(OUTPUT_FEED_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Procesando frames")

    # --- WARMUP (Calentamiento del Modelo) ---
    print("▶ Ejecutando Warmup del modelo...")
    dummy_inp = np.zeros((1, img_size, img_size, 3), dtype=np.float32)
    try:
        model.predict(dummy_inp, verbose=0)
        print("✔ Warmup completado.")
    except Exception as e:
        print(f"Advertencia: Fallo en el Warmup: {e}")
    
    # --- Inicialización del Hilo de Guardado (Saver Thread) y CSV ---
    saver_q = queue.Queue()
    stop_token = object()
    metadata_csv = os.path.join(OUTPUT_FEED_DIR, 'video_saved_metadata.csv')
    
    if not os.path.exists(metadata_csv) or os.stat(metadata_csv).st_size == 0:
        with open(metadata_csv, 'w', newline='') as cf:
            writer_csv = csv.writer(cf)
            # 💡 AÑADIR CAMPO 'OCR_TEXT' AL CSV
            writer_csv.writerow(['track_id', 'filepath', 'score', 'ocr_text', 'timestamp', 'video_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])

    def saver_worker(q, csv_path):
        rows = {}
        # Cargar datos existentes si el archivo no está vacío
        if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
            try:
                with open(csv_path, 'r', newline='') as cf:
                    # 💡 CAMBIAR fieldnames para incluir 'ocr_text'
                    reader = csv.DictReader(cf, fieldnames=['track_id', 'filepath', 'score', 'ocr_text', 'timestamp', 'video_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
                    next(reader) # Saltar encabezado
                    for r in reader:
                        tid = r.get('track_id', '')
                        rows[tid] = r
            except Exception as e:
                print(f"Advertencia: Error leyendo CSV existente. Se creará uno nuevo. {e}")
                rows = {} # Reiniciar si hay error
                
        def write_csv():
            try:
                with open(csv_path, 'w', newline='') as cf:
                    # 💡 CAMBIAR fieldnames para incluir 'ocr_text'
                    fieldnames = ['track_id', 'filepath', 'score', 'ocr_text', 'timestamp', 'video_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
                    writer_csv = csv.DictWriter(cf, fieldnames=fieldnames)
                    writer_csv.writeheader()
                    for _, r in rows.items():
                        writer_csv.writerow(r)
            except Exception as e:
                print('Error escribiendo CSV:', e)

        while True:
            job = q.get()
            if job is stop_token:
                q.task_done()
                # Escribir el estado final del CSV al detenerse
                write_csv()
                break
                
            try:
                if job.get('type') == 'image':
                    saved = False
                    try:
                        # Guardar la imagen
                        saved = cv2.imwrite(job['path'], job['image'])
                    except Exception as e:
                        print('cv2.imwrite error:', e)
                        saved = False

                    if saved:
                        # Actualizar/Añadir fila al buffer de filas
                        tid = str(job.get('track_id', ''))
                        rows[tid] = {
                            'track_id': tid,
                            'filepath': job['path'],
                            'score': str(job.get('score', '')),
                            'ocr_text': job.get('ocr_text', ''), # 💡 AGREGAR OCR_TEXT
                            'timestamp': job.get('timestamp', ''),
                            'video_path': job.get('video_path', ''),
                            'bbox_x1': str(job.get('bbox', ['', '', '', ''])[0]),
                            'bbox_y1': str(job.get('bbox', ['', '', '', ''])[1]),
                            'bbox_x2': str(job.get('bbox', ['', '', '', ''])[2]),
                            'bbox_y2': str(job.get('bbox', ['', '', '', ''])[3])
                        }
                        # Escribir el CSV después de cada guardado (o podrías optimizar y escribir menos frecuentemente)
                        write_csv() 
                        
            except Exception as e:
                print('Saver thread error:', e)
            finally:
                q.task_done()

    saver_thread = threading.Thread(target=saver_worker, args=(saver_q, metadata_csv), daemon=True)
    saver_thread.start()

    # --- Inicialización de Tracking ---
    tracks = []
    next_track_id = 1

    def iou(a, b):
        """Calcula Intersection over Union (IoU) para dos cajas [x1, y1, x2, y2]."""
        # ... (código existente) ...
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        boxBArea = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        denom = boxAArea + boxBArea - interArea
        return interArea / denom if denom > 0 else 0.0

    # Inicialización de ventana de visualización
    if display:
        try:
            cv2.namedWindow('Detección', cv2.WINDOW_NORMAL)
            max_w = min(1280, width)
            max_h = min(720, height)
            cv2.resizeWindow('Detección', max_w, max_h)
        except Exception:
            pass

    # =======================================================
    # Bucle Principal de Procesamiento de Video
    # =======================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Preprocesamiento e Inferencia (SIN CAMBIOS)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pad, scale, top, left = resize_pad(rgb, img_size)
        inp = (img_pad.astype(np.float32) / 255.0)[None, ...]
        
        pred_tensor = model.predict(inp, verbose=0) 
        boxes_norm, scores = process_predictions(pred_tensor, img_size=img_size)

        out_frame = frame.copy()
        detections = []
        
        # 2. Desnormalizar y Filtrar Detecciones (SIN CAMBIOS)
        for box_norm, score in zip(boxes_norm, scores):
            # ... (código existente de desnormalización y filtrado) ...
            x1_p = int(box_norm[0] * img_size)
            y1_p = int(box_norm[1] * img_size)
            x2_p = int(box_norm[2] * img_size)
            y2_p = int(box_norm[3] * img_size)
            
            x1_orig = int(max(0, (x1_p - left) / scale))
            y1_orig = int(max(0, (y1_p - top) / scale))
            x2_orig = int(min(width, (x2_p - left) / scale))
            y2_orig = int(min(height, (y2_p - top) / scale))
            
            w = max(1, x2_orig - x1_orig)
            h = max(1, y2_orig - y1_orig)
            area = w * h
            aspect = float(w) / float(h)
            
            valid_shape = True
            if area < min_area:
                valid_shape = False
            if aspect < aspect_ratio_min or aspect > aspect_ratio_max:
                valid_shape = False
                
            if not valid_shape:
                continue
                
            detections.append({'bbox': [x1_orig, y1_orig, x2_orig, y2_orig], 'score': score})

        # 3. Lógica de Tracking (Simple IOU-based)
        matched_track_ids = set()
        new_tracks = []

        for det in detections:
            best_tid = None
            best_iou = 0.0
            
            # Intentar emparejar con tracks existentes
            for t in tracks:
                if t['id'] in matched_track_ids:
                    continue
                # Usar la posición actual del track para el cálculo de IOU
                i = iou(det['bbox'], t['bbox'])
                if i > best_iou:
                    best_iou = i
                    best_tid = t['id']
                    
            if best_tid is not None and best_iou >= iou_thresh:
                # Actualizar track existente
                for t in tracks:
                    if t['id'] == best_tid:
                        # 🔥 Aplicar Suavizado (Damping) al Bounding Box (bbox_previo, no det)
                        prev_bbox = t['bbox']
                        det_bbox = det['bbox']
                        
                        # Nueva posición del track es una media ponderada
                        t['bbox'] = [
                            int(dampening_factor * prev_bbox[i] + (1 - dampening_factor) * det_bbox[i])
                            for i in range(4)
                        ]
                        
                        t['missed'] = 0
                        
                        # Consecutividad y Best Score
                        if det['score'] >= THRESHOLD:
                            t['consec'] = t.get('consec', 0) + 1
                        else:
                            t['consec'] = 0
                            
                        current_best_score = t.get('best_score', 0)
                        
                        # Solo actualizar la mejor detección si mejora
                        if det['score'] > current_best_score:
                            t['best_score'] = det['score']
                            t['best_frame'] = out_frame.copy() 
                        
                        # Confirmar y guardar si alcanza frames consecutivos
                        if not t.get('confirmed', False) and t.get('consec', 0) >= confirm_frames:
                            t['confirmed'] = True
                            
                            # 🔥 RECORTAR Y APLICAR OCR CON PADDING
                            best_bbox = t['bbox'] # Usamos la bbox suavizada para el recorte
                            x1, y1, x2, y2 = [int(x) for x in best_bbox]

                            # 🔥 Aplicar Padding para mejorar el OCR
                            w = x2 - x1
                            h = y2 - y1
                            pad_x = int(w * ocr_padding_ratio)
                            pad_y = int(h * ocr_padding_ratio)
                            
                            x1_p = max(0, x1 - pad_x)
                            y1_p = max(0, y1 - pad_y)
                            x2_p = min(frame.shape[1], x2 + pad_x)
                            y2_p = min(frame.shape[0], y2 + pad_y)
                            
                            # Recortar del mejor frame, pero con las coordenadas paddeadas
                            cropped_plate = t['best_frame'][y1_p:y2_p, x1_p:x2_p] 
                            ocr_text = run_ocr(cropped_plate)
                            t['ocr_text'] = ocr_text 

                            # Guardar imagen y metadatos
                            fname = os.path.basename(video_path)
                            name_suffix = ocr_text if ocr_text and not ocr_text.startswith('[OCR') else f"track{t['id']}"
                            img_name = f"det_{fname}_{name_suffix}.jpg"
                            img_path = os.path.join(OUTPUT_FEED_DIR, img_name)
                            
                            full = t['best_frame'].copy()
                            cv2.rectangle(full, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            saver_q.put({'type': 'image', 'path': img_path, 'image': full, 
                                         'track_id': t['id'], 'score': t['best_score'], 
                                         'ocr_text': ocr_text, 
                                         'timestamp': datetime.now().isoformat(), 
                                         'video_path': out_video_path, 'bbox': best_bbox})
                                         
                        new_tracks.append(t)
                        matched_track_ids.add(t['id'])
                        break
            else:
                # Crear nuevo track
                tnew = {'id': next_track_id, 'bbox': det['bbox'], 'best_score': det['score'], 
                        'missed': 0, 'consec': 0, 'confirmed': False, 'ocr_text': ''}
                next_track_id += 1
                
                # iniciar consec si supera THRESHOLD
                if tnew['best_score'] >= THRESHOLD:
                    tnew['consec'] = 1
                    tnew['best_frame'] = out_frame.copy() # Guardar el frame inicial
                    
                # Confirmar y guardar si alcanza consecutividad en el primer frame (si confirm_frames=1)
                if tnew['consec'] >= confirm_frames:
                    tnew['confirmed'] = True
                    
                    # 🔥 RECORTAR Y APLICAR OCR CON PADDING
                    best_bbox = tnew['bbox']
                    x1, y1, x2, y2 = [int(x) for x in best_bbox]
                    
                    # 🔥 Aplicar Padding para mejorar el OCR
                    w = x2 - x1
                    h = y2 - y1
                    pad_x = int(w * ocr_padding_ratio)
                    pad_y = int(h * ocr_padding_ratio)
                    
                    x1_p = max(0, x1 - pad_x)
                    y1_p = max(0, y1 - pad_y)
                    x2_p = min(frame.shape[1], x2 + pad_x)
                    y2_p = min(frame.shape[0], y2 + pad_y)
                    
                    cropped_plate = tnew['best_frame'][y1_p:y2_p, x1_p:x2_p]
                    ocr_text = run_ocr(cropped_plate)
                    tnew['ocr_text'] = ocr_text 
                    
                    # Guardar imagen y metadatos
                    fname = os.path.basename(video_path)
                    name_suffix = ocr_text if ocr_text and not ocr_text.startswith('[OCR') else f"track{tnew['id']}"
                    img_name = f"det_{fname}_{name_suffix}.jpg"
                    img_path = os.path.join(OUTPUT_FEED_DIR, img_name)
                    
                    full = tnew['best_frame'].copy()
                    cv2.rectangle(full, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    saver_q.put({'type': 'image', 'path': img_path, 'image': full, 
                                 'track_id': tnew['id'], 'score': tnew['best_score'], 
                                 'ocr_text': ocr_text, 
                                 'timestamp': datetime.now().isoformat(), 
                                 'video_path': out_video_path, 'bbox': best_bbox})
                    
                new_tracks.append(tnew)

        # 4. Manejar tracks perdidos (SIN CAMBIOS)
        for t in tracks:
            if t['id'] not in matched_track_ids:
                t['missed'] = t.get('missed', 0) + 1
                if t['missed'] <= max_missed:
                    new_tracks.append(t)
                    
        tracks = [t for t in new_tracks if t.get('missed', 0) <= max_missed]

        # 5. Dibujar Bounding Boxes y escribir frame de salida (SIN CAMBIOS)
        for t in tracks:
            bx = t['bbox']
            color = (0, 255, 0) if t.get('confirmed', False) else (0, 255, 255) 
            if t.get('missed', 0) > 0:
                color = (0, 165, 255) # Naranja
                
            cv2.rectangle(out_frame, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
            
            ocr_text_display = t.get('ocr_text', 'Buscando OCR...')
            label_top = f"ID:{t['id']} M:{t.get('missed', 0)} S:{t.get('best_score', 0)*100:.0f}%"
            label_ocr = f"OCR: {ocr_text_display}"
            
            cv2.putText(out_frame, label_top, (bx[0], max(0, bx[1] - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(out_frame, label_ocr, (bx[0], max(0, bx[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        writer.write(out_frame)
        if display:
            cv2.imshow("Detección", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pbar.update(1)

    # =======================================================
    # Limpieza final (SIN CAMBIOS)
    # =======================================================
    cap.release()
    writer.release()
    pbar.close()
    if display:
        cv2.destroyAllWindows()
        
    saver_q.put(stop_token)
    saver_thread.join(timeout=10) 
    
    print(f"✔ Video de salida guardado en: {out_video_path}")
    return out_video_path


def main():
    parser = argparse.ArgumentParser(description="Inferir un video con detector de placas")
    parser.add_argument("--video", required=True, help="Ruta a archivo de video")
    parser.add_argument("--model", default=MODEL_PATH, help="Ruta al modelo Keras")
    parser.add_argument("--out", default=None, help="Ruta de salida mp4 opcional")
    parser.add_argument("--display", action="store_true", help="Mostrar video en pantalla")
    # 🔥 PARÁMETROS OPTIMIZADOS POR DEFECTO 🔥
    parser.add_argument("--confirm_frames", type=int, default=1, help="Frames consecutivos necesarios para confirmar un track (Default: 1)")
    parser.add_argument("--iou_thresh", type=float, default=0.30, help="Umbral IoU para emparejar detecciones con tracks (Default: 0.30)")
    parser.add_argument("--min_area", type=int, default=800, help="Área mínima en píxeles para aceptar una detección (Default: 800)")
    parser.add_argument("--aspect_ratio_min", type=float, default=1.8, help="Relación de aspecto mínima (w/h) para aceptar detecciones (Default: 1.8)")
    parser.add_argument("--aspect_ratio_max", type=float, default=8.0, help="Relación de aspecto máxima (w/h) para aceptar detecciones (Default: 8.0)")
    parser.add_argument("--max_missed", type=int, default=10, help="Máximo de frames perdidos antes de eliminar un track (Default: 10)")
    
    # 🔥 NUEVOS PARÁMETROS DE TRACKING Y OCR 🔥
    parser.add_argument("--dampening_factor", type=float, default=0.75, help="Factor de suavizado de la posición del track (0.0=no suavizado, 1.0=solo posición anterior). Default: 0.75")
    parser.add_argument("--ocr_padding_ratio", type=float, default=0.1, help="Ratio de padding alrededor de la placa para el OCR (e.g., 0.1 = 10% de margen). Default: 0.1")
    
    args = parser.parse_args()
    
    # 1. Carga del modelo
    model = load_model_safe(args.model)
    
    # 2. Procesamiento
    out = process_video(
        model,
        args.video,
        out_video_path=args.out,
        img_size=IMG_SIZE[0],
        display=args.display,
        max_missed=args.max_missed,
        iou_thresh=args.iou_thresh,
        confirm_frames=args.confirm_frames,
        min_area=args.min_area,
        aspect_ratio_min=args.aspect_ratio_min,
        aspect_ratio_max=args.aspect_ratio_max,
        # 🔥 Pasar los nuevos parámetros
        dampening_factor=args.dampening_factor,
        ocr_padding_ratio=args.ocr_padding_ratio,
    )
    print("Hecho.")


if __name__ == "__main__":
    main()