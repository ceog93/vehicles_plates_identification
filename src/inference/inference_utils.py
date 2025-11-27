# Versión Final Definitiva: Lógica de Inferencia Compartida
import re
import os
import sys
from datetime import datetime
import csv
import queue
import subprocess
import threading, webbrowser

import cv2
import numpy as np
import tensorflow as tf

# OCR - Importación segura
try:
    import easyocr
    READER = easyocr.Reader(['es', 'en'], gpu=False)
    OCR_DEFAULTS = {'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'mag_ratio': 1.0}
except Exception:
    print("Advertencia: EasyOCR no está instalado o falló la inicialización. OCR deshabilitado.")
    READER = None
    OCR_DEFAULTS = {}

# Algoritmo Húngaro para tracking
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_HUNGARIAN = True
except Exception:
    linear_sum_assignment = None
    _HAS_HUNGARIAN = False

# Importaciones del proyecto
from src.config import MODEL_PATH, IMG_SIZE, ROOT_MODEL_DIR, LATEST_MODEL_PATH, THRESHOLD
from src.models.efficient_detector_multi_placa import NUM_CLASES, perdida_ciou
from src.utils.image_bbox_utils import nms_numpy

# ---------------------------
# UTILIDAD DE SISTEMA OPERATIVO
# ---------------------------
def get_plate_from_csv(csv_path, image_filename):
    """Lee el CSV y devuelve la placa para un nombre de archivo de imagen dado."""
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Compara el nombre de archivo base
                csv_filename = os.path.basename(row.get('filepath', ''))
                if csv_filename == image_filename:
                        return row.get('plate_text', '')
    except FileNotFoundError:
        # El archivo puede no existir al principio, es normal.
        pass
    except Exception as e:
        print(f"⚠️ Error leyendo CSV para la galería: {e}")
    return ''

def create_html_gallery(folder_path):
    """
    Crea un archivo index.html en la carpeta de resultados para mostrar una galería
    de las imágenes y videos generados.
    """
    html_path = os.path.join(folder_path, "index.html")
    
    csv_path = os.path.join(folder_path, "metadata.csv")
    # Buscar archivos de imagen y video
    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    videos = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')])

    # Estilo CSS para la galería
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Resultados de Inferencia</title>
        <style>
            body { font-family: sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; }
            h1, h2 { color: #333; text-align: center; }
            .gallery { display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; }
            .gallery-item { cursor: pointer; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); transition: transform 0.2s; }
            .gallery-item:hover { transform: scale(1.05); box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
            .gallery-item h4 { margin: 10px 0 5px 0; color: #555; font-size: 14px; }
            .gallery-item img { width: 300px; height: 200px; object-fit: cover; display: block; }
            .gallery-item video { max-width: 100%; display: block; }
            .video-container { width: 100%; max-width: 800px; margin: 20px auto; }
            /* Estilos del Lightbox */
            .lightbox { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.85); justify-content: center; align-items: center; }
            .lightbox-content { max-width: 90%; max-height: 90%; }
            .lightbox-close { position: absolute; top: 20px; right: 35px; color: #fff; font-size: 40px; font-weight: bold; cursor: pointer; }
            .lightbox-nav { position: absolute; top: 50%; transform: translateY(-50%); background-color: rgba(0,0,0,0.5); color: white; border: none; font-size: 30px; cursor: pointer; padding: 10px 15px; user-select: none; }
            .prev { left: 10px; }
            .next { right: 10px; }
        </style>
    </head>
    <body>
        <h1>Resultados de Inferencia</h1>
    """

    # Sección de Imágenes
    if images:
        html_content += "<h2>Imágenes Guardadas</h2><div class='gallery' id='image-gallery'>"
        for i, image in enumerate(images):
            plate_text = get_plate_from_csv(csv_path, image)
            title = f"Placa: {plate_text}" if plate_text else "Placa: (No detectada)"

            html_content += f"""
            <div class="gallery-item" onclick="openLightbox({i})">
                <img src="{image}" alt="{image}">
                <h4>{title}</h4>
            </div>
            """
        html_content += "</div>"

    # Contenedor del Lightbox
    html_content += """
        <div id="lightbox" class="lightbox">
            <span class="lightbox-close" onclick="closeLightbox()">&times;</span>
            <img class="lightbox-content" id="lightbox-img">
            <button class="lightbox-nav prev" onclick="changeImage(-1)">&#10094;</button>
            <button class="lightbox-nav next" onclick="changeImage(1)">&#10095;</button>
        </div>
    """

    # Script de JavaScript para el Lightbox
    html_content += f"""
    <script>
        const images = {str(images)};
        let currentImageIndex = 0;
        const lightbox = document.getElementById('lightbox');
        const lightboxImg = document.getElementById('lightbox-img');

        function openLightbox(index) {{
            currentImageIndex = index;
            lightboxImg.src = images[currentImageIndex];
            lightbox.style.display = 'flex';
        }}
        function closeLightbox() {{
            lightbox.style.display = 'none';
        }}
        function changeImage(step) {{
            currentImageIndex = (currentImageIndex + step + images.length) % images.length;
            lightboxImg.src = images[currentImageIndex];
        }}
        document.addEventListener('keydown', function(e) {{
            if (lightbox.style.display === 'flex') {{
                if (e.key === 'Escape') closeLightbox();
                if (e.key === 'ArrowLeft') changeImage(-1);
                if (e.key === 'ArrowRight') changeImage(1);
            }}
        }});
    </script>
    </body></html>
    """

    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return html_path
    except Exception as e:
        print(f"⚠️ No se pudo crear la galería HTML: {e}")
        return None

def open_folder(folder_path):
    """
    Crea una galería HTML y muestra la ruta en la consola sin abrirla.
    """
    gallery_path = create_html_gallery(folder_path)
    
    # Ya no se intenta abrir la carpeta/archivo automáticamente.
    # Solo se imprime la ruta para que el usuario la abra manualmente.
    if gallery_path:
        print(f"✔ Galería HTML creada en: {os.path.realpath(gallery_path)}")
    
    print(f"📂 Puedes encontrar todos los resultados en: {os.path.realpath(folder_path)}")

# ---------------------------
# UTILIDAD DE DIBUJADO
# ---------------------------
def draw_labels(image, bbox, track_id, score, ocr_text="", color=(0, 255, 0)):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_img, w_img = image.shape[:2]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label_top = f"ID:{track_id} | {score*100:.0f}%"
    (w_top, h_top), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, max(0, y1 - 25)), (x1 + w_top + 10, max(0, y1)), (0, 0, 0), -1)
    cv2.putText(image, label_top, (x1 + 5, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    plate_text = ocr_text.get('plate', '') if isinstance(ocr_text, dict) else ocr_text
    if plate_text:
        label_bot = f"PLACA: {plate_text}"
        (w_bot, h_bot), _ = cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_bot_start = min(h_img - 1, y2)
        y_bot_end = min(h_img, y2 + 30)
        cv2.rectangle(image, (x1, y_bot_start), (x1 + w_bot + 10, y_bot_end), (0, 0, 0), -1)
        cv2.putText(image, label_bot, (x1 + 5, min(h_img - 5, y2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return image

def draw_bbox_safe(image, bbox, track_id, score, ocr_text="", color=(0, 255, 0)):
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_img, w_img = image.shape[:2]
        x1 = max(0, min(x1, w_img - 1)); y1 = max(0, min(y1, h_img - 1))
        x2 = max(x1 + 1, min(x2, w_img)); y2 = max(y1 + 1, min(y2, h_img))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        try:
            label_top = f"ID:{track_id} | {score*100:.0f}%"
            (w_top, h_top), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, max(0, y1 - 25)), (x1 + w_top + 10, max(0, y1)), (0, 0, 0), -1)
            cv2.putText(image, label_top, (x1 + 5, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except: pass
        plate_text = ocr_text.get('plate', '') if isinstance(ocr_text, dict) else ocr_text
        if plate_text:
            try:
                label_bot = f"PLACA: {plate_text}"
                (w_bot, h_bot), _ = cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                y_bot_start = min(h_img - 1, y2)
                y_bot_end = min(h_img, y2 + 30)
                cv2.rectangle(image, (x1, y_bot_start), (x1 + w_bot + 10, y_bot_end), (0, 0, 0), -1)
                cv2.putText(image, label_bot, (x1 + 5, min(h_img - 5, y2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            except: pass
    except Exception: pass
    return image

# ---------------------------
# LÓGICA DE PROCESAMIENTO
# ---------------------------
def process_predictions(output_tensor, img_size=IMG_SIZE[0], confidence_threshold=THRESHOLD):
    arr = np.asarray(output_tensor)
    if arr.ndim == 4 and arr.shape[0] == 1: arr = arr[0]
    gh, gw, channels = arr.shape[:3]
    per_anchor = 5 + NUM_CLASES
    anchors = int(channels // per_anchor)
    arr = arr.reshape(gh, gw, anchors, per_anchor)
    final_boxes_norm, final_scores = [], []
    for i in range(gh):
        for j in range(gw):
            for a in range(anchors):
                cell = arr[i, j, a]
                conf = float(cell[0])
                if conf < confidence_threshold: continue
                cx_local, cy_local = float(cell[1]), float(cell[2])
                w_norm, h_norm = float(cell[3]), float(cell[4])
                cx_norm = (j + cx_local) / gw; cy_norm = (i + cy_local) / gh
                xmin = cx_norm - (w_norm / 2.0); ymin = cy_norm - (h_norm / 2.0)
                xmax = cx_norm + (w_norm / 2.0); ymax = cy_norm + (h_norm / 2.0)
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
    if not os.path.isdir(ROOT_MODEL_DIR): return None
    subdirs = [d for d in os.listdir(ROOT_MODEL_DIR) if os.path.isdir(os.path.join(ROOT_MODEL_DIR, d))]
    if not subdirs: return None
    subdirs.sort(reverse=True)
    for sd in subdirs:
        candidate_dir = os.path.join(ROOT_MODEL_DIR, sd)
        for name in ("detector_model.keras", "detector.keras", "detector.h5", "detector"):
            p = os.path.join(candidate_dir, name)
            if os.path.exists(p): return p
        for f in os.listdir(candidate_dir):
            if f.endswith(".keras") or f.endswith(".h5"): return os.path.join(candidate_dir, f)
    return None

def load_model_safe(model_path=None):
    candidates = []
    if model_path: candidates.append(model_path)
    candidates.append(MODEL_PATH)
    if LATEST_MODEL_PATH: candidates.append(LATEST_MODEL_PATH)
    auto = find_latest_model_in_models_dir()
    if auto: candidates.append(auto)
    last_err = None
    for p in candidates:
        if not p or not os.path.exists(p): continue
        try:
            model = tf.keras.models.load_model(p, custom_objects={'perdida_ciou': perdida_ciou}, compile=False)
            print(f"✔ Modelo cargado desde: {p}")
            return model
        except Exception as e:
            last_err = e
            print(f"Intento de carga falló para {p}: {e}")
    raise FileNotFoundError(f"No se encontró modelo válido. Intentos: {candidates}. Ultimo error: {last_err}")

def run_ocr(cropped_img):
    if READER is None: return {'plate': "[OCR NO INSTALADO]", 'city': ''}
    try:
        rgb_crop = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB) if len(cropped_img.shape) == 3 else cropped_img
        results = READER.readtext(rgb_crop, **OCR_DEFAULTS)
        if not results: return {'plate': "", 'city': ""}
        results.sort(key=lambda r: r[0][0][1])
        plate_text, city_text = "", ""
        if len(results) > 0:
            raw_plate = results[0][1]
            plate_text = re.sub(r'\s+', '', raw_plate).replace("-", "").upper()

            # --- NUEVO: LIMPIEZA POR LONGITUD (para eliminar bordes) ---
            # Caso 1: Se detectan 7 caracteres (un borde)
            if len(plate_text) == 7:
                # Generar dos candidatos: quitando el primero y quitando el último
                candidate1 = plate_text[1:]  # Ej: "1ABC123" -> "ABC123"
                candidate2 = plate_text[:-1] # Ej: "ABC1237" -> "ABC123"

                # Función para puntuar qué tan "buena" es una placa (3 letras, 3 números)
                def score_plate(p):
                    if len(p) != 6: return 0
                    score = 0
                    for i in range(3):
                        if 'A' <= p[i] <= 'Z': score += 1
                    for i in range(3, 6):
                        if '0' <= p[i] <= '9': score += 1
                    return score

                # Elegir el mejor candidato
                if score_plate(candidate1) > score_plate(candidate2):
                    plate_text = candidate1
                else:
                    plate_text = candidate2
            # Caso 2: Se detectan 8 caracteres (ambos bordes)
            elif len(plate_text) == 8:
                plate_text = plate_text[1:-1] # Ej: "1ABC1237" -> "ABC123"

            if len(plate_text) == 6:
                corrected_plate = ""
                for i in range(3):
                    char = plate_text[i]
                    if char == '0': corrected_plate += 'O'
                    elif char == '1' or char == '4': corrected_plate += 'I'
                    else: corrected_plate += char
                for i in range(3, 6):
                    char = plate_text[i]
                    if i <= 4:
                        if char == 'O' or char == 'Q': corrected_plate += '0'
                        elif char == 'I' or char == 'L': corrected_plate += '1'
                        elif char == 'S': corrected_plate += '5'
                        else: corrected_plate += char
                    else: corrected_plate += char
                plate_text = corrected_plate
        if len(results) > 1:
            raw_city = results[1][1]
            city_text = re.sub(r'[^A-Z\s]', '', raw_city.upper()).strip()
        return {'plate': plate_text, 'city': city_text}
    except Exception: return {'plate': f"[OCR ERROR]", 'city': ''}

# ---------------------------
# LÓGICA DE TRACKING
# ---------------------------
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
    history = track.get('detection_history', [])
    if not history: return track.get('best_frame'), track.get('best_bbox'), track.get('best_score'), track.get('ocr_text', {'plate': '', 'city': ''})
    best_det = history[0]
    col_plate_re = re.compile(r'^[A-Z]{3}[0-9]{3}$')
    col_matches = [d for d in history if isinstance(d.get('ocr_text'), dict) and col_plate_re.match(d.get('ocr_text', {}).get('plate', ''))]
    if col_matches:
        best_det = max(col_matches, key=lambda x: x['score'])
        return (best_det['frame'], best_det['bbox'], best_det['score'], best_det.get('ocr_text', {'plate': '', 'city': ''}))
    ocr_success = [d for d in history if isinstance(d.get('ocr_text'), dict) and d.get('ocr_text', {}).get('plate') and 'ERR' not in d.get('ocr_text', {}).get('plate', '')]
    if ocr_success:
        best_det = max(ocr_success, key=lambda x: x['score'])
        return (best_det['frame'], best_det['bbox'], best_det['score'], best_det.get('ocr_text', {'plate': '', 'city': ''}))
    best_det = max(history, key=lambda x: x['score'])
    return (best_det['frame'], best_det['bbox'], best_det['score'], best_det.get('ocr_text', {'plate': '', 'city': ''}))

def match_detections_to_tracks(detections, tracks, iou_thr=0.30, dist_thr=0.35):
    if len(tracks) == 0: return [], [], list(range(len(detections)))
    if len(detections) == 0: return [], list(range(len(tracks))), []
    T, D = len(tracks), len(detections)
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
    matches, matched_t, matched_d = [], set(), set()
    if _HAS_HUNGARIAN and linear_sum_assignment is not None:
        try:
            t_idx, d_idx = linear_sum_assignment(cost)
            for ti, di in zip(t_idx, d_idx):
                if _iou_numpy(tracks[ti]['bbox'], detections[di]['bbox']) >= iou_thr:
                    matches.append((ti, di)); matched_t.add(ti); matched_d.add(di)
        except Exception: pass
    if len(matches) == 0:
        flat = sorted([(cost[ti, di], ti, di) for ti in range(T) for di in range(D)], key=lambda x: x[0])
        used_t, used_d = set(), set()
        for c, ti, di in flat:
            if ti in used_t or di in used_d: continue
            if _iou_numpy(tracks[ti]['bbox'], detections[di]['bbox']) >= iou_thr:
                matches.append((ti, di)); used_t.add(ti); used_d.add(di)
        matched_t.update(used_t); matched_d.update(used_d)
    unmatched_t = [i for i in range(T) if i not in matched_t]
    unmatched_d = [j for j in range(D) if j not in matched_d]
    return matches, unmatched_t, unmatched_d

# ---------------------------
# HILOS ASÍNCRONOS
# ---------------------------
def start_saver_thread(q, csv_path):
    stop_token = object()
    def worker():
        rows = {}
        if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
            try:
                with open(csv_path, 'r', newline='') as cf:
                    for rec in csv.DictReader(cf):
                        # Usar el filepath como clave única para evitar sobrescrituras
                        if rec.get('filepath'): rows[rec['filepath']] = rec
            except Exception: pass
            
        fieldnames = ['track_id', 'filepath', 'score', 'plate_text', 'city_text', 'timestamp', 'video_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
        def write_csv():
            try:
                with open(csv_path, 'w', newline='') as cf:
                    writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows.values())
            except Exception as e: print("Error CSV:", e)
        while True:
            job = q.get()
            if job is stop_token:
                write_csv(); q.task_done(); break
            try:
                if job.get('type') == 'image':
                    path = job['path']
                    if cv2.imwrite(path, job['image']):
                        tid = str(job.get('track_id', ''))
                        bbox = job.get('bbox', [0,0,0,0])
                        ocr_data = job.get('ocr_text', {'plate': '', 'city': ''})
                        # Usar el path como clave única en el diccionario
                        rows[path] = {
                            'track_id': tid, 'filepath': path, 'score': f"{job.get('score', 0):.4f}",
                            'plate_text': ocr_data.get('plate', ''), 'city_text': ocr_data.get('city', ''),
                            'timestamp': job.get('timestamp', ''), 'video_path': job.get('video_path', ''),
                            'bbox_x1': str(bbox[0]), 'bbox_y1': str(bbox[1]),
                            'bbox_x2': str(bbox[2]), 'bbox_y2': str(bbox[3]),
                        }
                        write_csv()
            except Exception: pass
            finally: q.task_done()
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t, stop_token

def start_ocr_thread(q, results_dict):
    stop_token = object()
    def ocr_worker():
        while True:
            job = q.get()
            if job is stop_token: q.task_done(); break
            track_id, crop_img = job.get('track_id'), job.get('image')
            if track_id is None or crop_img is None or crop_img.size == 0:
                q.task_done(); continue
            try:
                ocr_text = run_ocr(crop_img)
                if ocr_text and ocr_text.get('plate'):
                    results_dict[track_id] = ocr_text
            finally: q.task_done()
    t = threading.Thread(target=ocr_worker, daemon=True)
    t.start()
    return t, stop_token