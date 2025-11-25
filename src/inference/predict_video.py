# src/inference/predict_video.py
# Versión final corregida — (c) Tu proyecto
import os

# ---------------------------
# IMPORTANT: set TF env BEFORE importing tensorflow
# ---------------------------
# Reduce XLA / cuDNN autotune activity that can be unstable on some GPUs (e.g., GTX 1650)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# Optional: reduce GPU memory growth behavior (safer)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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

# OCR
try:
    import easyocr
    READER = easyocr.Reader(['es', 'en'], gpu=False)
except Exception:
    print("Advertencia: EasyOCR no está instalado o falló la inicialización. OCR deshabilitado.")
    READER = None

# Hungarian (linear_sum_assignment)
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_HUNGARIAN = True
except Exception:
    linear_sum_assignment = None
    _HAS_HUNGARIAN = False

# Config & utilidades (asegúrate de que src.config y src.utils.mpd_utils existan)
from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR, THRESHOLD, ROOT_MODEL_DIR, LATEST_MODEL_PATH
from src.utils.mpd_utils import resize_pad, nms_numpy
from src.models.efficient_detector_multi_placa import NUM_CLASSES, yolo_ciou_loss

# ---------------------------
# Helpers / Decoding YOLO-like output
# ---------------------------
def process_predictions(output_tensor, img_size=IMG_SIZE[0], confidence_threshold=THRESHOLD):
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
                cx_local, cy_local, w_norm, h_norm = float(cell[1]), float(cell[2]), float(cell[3]), float(cell[4])
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


# ---------------------------
# Model loading utilities
# ---------------------------
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
        if not p:
            continue
        if not os.path.exists(p):
            continue
        try:
            model = tf.keras.models.load_model(p, custom_objects={'yolo_ciou_loss': yolo_ciou_loss}, compile=False)
            print(f"✔ Modelo cargado desde: {p}")
            return model
        except Exception as e:
            last_err = e
            print(f"Intento de carga falló para {p}: {e}")
    raise FileNotFoundError(f"No se encontró modelo válido. Intentos: {candidates}. Ultimo error: {last_err}")


# ---------------------------
# OCR helper (preprocessing)
# ---------------------------
def run_ocr(cropped_img):
    if READER is None:
        return "[OCR NO INSTALADO]"
    try:
        # cropped_img must be BGR (cv2), EasyOCR accepts RGB or grayscale; we will convert to RGB inside
        if len(cropped_img.shape) == 3 and cropped_img.shape[2] == 3:
            rgb_crop = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_crop = cropped_img
        results = READER.readtext(rgb_crop, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', detail=0, paragraph=True)
        if results:
            text = "".join(results).replace(" ", "").replace("-", "")
            return text.upper()
        return ""
    except Exception as e:
        return f"[OCR ERROR: {e}]"


# ---------------------------
# Matcher IoU + Distancia (robusto)
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


def match_detections_to_tracks(detections, tracks, iou_thr=0.30, dist_thr=0.35):
    """
    detections: list of {'bbox':[x1,y1,x2,y2], 'score':float}
    tracks: list of {'bbox':..., 'id':...}
    returns: matches [(ti, di)], unmatched_tracks_idx, unmatched_det_idx
    """
    if len(tracks) == 0:
        return [], [], list(range(len(detections)))
    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    T = len(tracks); D = len(detections)
    cost = np.ones((T, D), dtype=np.float32)

    for ti, tr in enumerate(tracks):
        box_t = tr['bbox']; ct = _center(box_t)
        # normalizing reference (bbox diagonal)
        ref = max(1.0, np.linalg.norm(np.array([box_t[2] - box_t[0], box_t[3] - box_t[1]])))
        for di, det in enumerate(detections):
            box_d = det['bbox']; cd = _center(box_d)
            iou_v = _iou_numpy(box_t, box_d)
            dist = np.linalg.norm(ct - cd)
            dist_norm = 1.0 - (dist / (ref + 1e-6))
            dist_norm = np.clip(dist_norm, 0.0, 1.0)
            # combined score (higher = better)
            score = 0.65 * iou_v + 0.35 * dist_norm
            cost[ti, di] = 1.0 - score

    cost_matrix = cost.copy()

    matches = []
    matched_t = set()
    matched_d = set()

    # Try Hungarian
    if _HAS_HUNGARIAN and linear_sum_assignment is not None:
        try:
            t_idx, d_idx = linear_sum_assignment(cost_matrix)
            for ti, di in zip(t_idx, d_idx):
                # check IoU threshold (and small dist) before accepting
                if _iou_numpy(tracks[ti]['bbox'], detections[di]['bbox']) >= iou_thr:
                    matches.append((ti, di))
                    matched_t.add(ti); matched_d.add(di)
        except Exception as e:
            print("⚠ Hungarian falló:", e)
            # fall through to greedy fallback

    # Greedy fallback for remaining
    if len(matches) == 0:
        flat = []
        for ti in range(T):
            for di in range(D):
                flat.append((cost_matrix[ti, di], ti, di))
        flat.sort(key=lambda x: x[0])
        used_t = set(); used_d = set()
        for c, ti, di in flat:
            if ti in used_t or di in used_d: continue
            iou_v = _iou_numpy(tracks[ti]['bbox'], detections[di]['bbox'])
            # accept only if IoU or distance-derived confidence is acceptable
            if iou_v >= iou_thr:
                matches.append((ti, di))
                used_t.add(ti); used_d.add(di)
        matched_t.update(used_t); matched_d.update(used_d)

    unmatched_t = [i for i in range(T) if i not in matched_t]
    unmatched_d = [j for j in range(D) if j not in matched_d]

    return matches, unmatched_t, unmatched_d


# ---------------------------
# Saver thread robust (CSV safe)
# ---------------------------
def start_saver_thread(q, csv_path):
    """
    q: queue.Queue receives dict jobs
    csv_path: path to metadata csv
    """
    stop_token = object()

    def worker():
        rows = {}  # key = track_id -> dict
        # read existing CSV if any (safe)
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
                # flush and exit
                try:
                    write_csv()
                except Exception as e:
                    print("Error final escribiendo CSV:", e)
                q.task_done()
                break

            try:
                if job.get('type') == 'image':
                    saved = False
                    try:
                        saved = cv2.imwrite(job['path'], job['image'])
                    except Exception as e:
                        print("cv2.imwrite error:", e)
                        saved = False
                    if saved:
                        tid = str(job.get('track_id', ''))
                        rows[tid] = {
                            'track_id': tid,
                            'filepath': job.get('path', ''),
                            'score': str(job.get('score', '')),
                            'ocr_text': job.get('ocr_text', ''),
                            'timestamp': job.get('timestamp', ''),
                            'video_path': job.get('video_path', ''),
                            'bbox_x1': str(job.get('bbox', ['', '', '', ''])[0]),
                            'bbox_y1': str(job.get('bbox', ['', '', '', ''])[1]),
                            'bbox_x2': str(job.get('bbox', ['', '', '', ''])[2]),
                            'bbox_y2': str(job.get('bbox', ['', '', '', ''])[3]),
                        }
                        # write after each save to minimize data loss
                        write_csv()
            except Exception:
                print("Saver thread exception:", traceback.format_exc())
            finally:
                q.task_done()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t, stop_token


# ---------------------------
# MAIN PROCESS_VIDEO
# ---------------------------
def process_video(model, video_path, out_video_path=None, img_size=IMG_SIZE[0], display=False,
                  max_missed=5, iou_thresh=0.45, confirm_frames=3, min_area=800,
                  aspect_ratio_min=1.8, aspect_ratio_max=8.0, dampening_factor=0.85,
                  ocr_padding_ratio=0.1):
    # open capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    base = os.path.basename(video_path)
    UNIQUE_OUTPUT_DIR = os.path.join(OUTPUT_FEED_DIR, ts)
    os.makedirs(UNIQUE_OUTPUT_DIR, exist_ok=True)

    out_video_path = os.path.join(UNIQUE_OUTPUT_DIR, f"det_{base}.mp4")

    # writer expects BGR frames with same size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    pbar = tqdm(total=total_frames, desc="Procesando frames")

    # warmup
    try:
        dummy = np.zeros((1, img_size, img_size, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)
    except Exception:
        pass

    # saver thread
    saver_q = queue.Queue()
    metadata_csv = os.path.join(UNIQUE_OUTPUT_DIR, 'video_saved_metadata.csv')
    saver_thread, stop_token = start_saver_thread(saver_q, metadata_csv)

    # tracker state
    tracks = []  # each dict: id,bbox,missed,seen,best_frame,best_score,best_area,saved,last_frame
    next_track_id = 1

    # display window single
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

        # Ensure BGR frame (cv2 gives BGR)
        bgr = frame.copy()
        # inference expects RGB for resize_pad (your helper likely expects RGB)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_pad, scale, top, left = resize_pad(rgb, img_size)
        inp = (img_pad.astype(np.float32) / 255.0)[None, ...]
        # predict
        pred_tensor = model.predict(inp, verbose=0)
        boxes_norm, scores = process_predictions(pred_tensor, img_size=img_size)

        out_frame = bgr.copy()
        detections = []

        # denormalize and filter
        for box_norm, score in zip(boxes_norm, scores):
            x1_p = int(box_norm[0] * img_size); y1_p = int(box_norm[1] * img_size)
            x2_p = int(box_norm[2] * img_size); y2_p = int(box_norm[3] * img_size)
            x1 = int(max(0, (x1_p - left) / scale)); y1 = int(max(0, (y1_p - top) / scale))
            x2 = int(min(width, (x2_p - left) / scale)); y2 = int(min(height, (y2_p - top) / scale))
            w = max(1, x2 - x1); h = max(1, y2 - y1)
            area = w * h
            aspect = float(w) / float(h) if h > 0 else 0
            if area < min_area or not (aspect_ratio_min <= aspect <= aspect_ratio_max):
                continue
            detections.append({'bbox': [x1, y1, x2, y2], 'score': float(score)})

        # matching
        matches, unmatched_tracks_idx, unmatched_dets_idx = match_detections_to_tracks(detections, tracks, iou_thr=iou_thresh, dist_thr=0.35)

        # update matched tracks
        used_track_ids = set()
        for (t_idx, d_idx) in matches:
            if t_idx >= len(tracks) or d_idx >= len(detections):
                continue
            tr = tracks[t_idx]
            det = detections[d_idx]
            prev_bbox = tr['bbox']; det_bbox = det['bbox']
            # EMA smoothing
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
            # update best_frame using score * area to prefer clear larger crops
            w_s = max(1, smoothed[2] - smoothed[0]); h_s = max(1, smoothed[3] - smoothed[1])
            area_s = w_s * h_s
            quality = det['score'] * area_s
            if quality > tr.get('best_area', 0):
                tr['best_area'] = quality
                tr['best_score'] = det['score']
                tr['best_frame'] = out_frame.copy()
                tr['best_bbox'] = smoothed.copy()

            # confirm logic
            if not tr.get('confirmed', False):
                if tr.get('seen', 0) >= confirm_frames and tr.get('best_score', 0) >= 0.35 and area_s >= 500:
                    tr['confirmed'] = True
                    # perform OCR on best frame (safe crop)
                    best_bbox = tr.get('best_bbox', tr['bbox'])
                    x1, y1, x2, y2 = [int(v) for v in best_bbox]
                    pad_x = int((x2 - x1) * ocr_padding_ratio); pad_y = int((y2 - y1) * ocr_padding_ratio)
                    x1p = max(0, x1 - pad_x); y1p = max(0, y1 - pad_y)
                    x2p = min(width, x2 + pad_x); y2p = min(height, y2 + pad_y)
                    crop = tr.get('best_frame', out_frame)[y1p:y2p, x1p:x2p].copy()
                    if crop.size != 0:
                        try:
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            gray = cv2.bilateralFilter(gray, 5, 75, 75)
                            gray = cv2.equalizeHist(gray)
                            gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)
                            ocr_text = run_ocr(gray)
                        except Exception as e:
                            ocr_text = f"[OCR ERROR: {e}]"
                    else:
                        ocr_text = ""
                    tr['ocr_text'] = ocr_text
                    # save image via saver thread
                    clean_text = ocr_text.replace('[OCR-', '').replace(']', '')
                    name_suffix = clean_text if clean_text and not clean_text.startswith('[OCR') else "none"
                    img_name = f"{ts}_track{tr['id']}_{name_suffix}.jpg"
                    img_path = os.path.join(UNIQUE_OUTPUT_DIR, img_name)
                    full = tr.get('best_frame', out_frame).copy()
                    try:
                        cv2.rectangle(full, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    except Exception:
                        pass
                    saver_q.put({'type': 'image', 'path': img_path, 'image': full,
                                 'track_id': tr['id'], 'score': tr.get('best_score', 0),
                                 'ocr_text': tr.get('ocr_text', ''), 'timestamp': datetime.now().isoformat(),
                                 'video_path': out_video_path, 'bbox': best_bbox})
            used_track_ids.add(tr['id'])

        # increment missed for unmatched tracks
        for ti in unmatched_tracks_idx:
            if 0 <= ti < len(tracks):
                tracks[ti]['missed'] = tracks[ti].get('missed', 0) + 1

        # create new tracks for unmatched detections
        for di in unmatched_dets_idx:
            if di < 0 or di >= len(detections):
                continue
            det = detections[di]
            tnew = {
                'id': next_track_id,
                'bbox': det['bbox'],
                'best_frame': out_frame.copy(),
                'best_bbox': det['bbox'].copy(),
                'best_score': det['score'],
                'best_area': det['score'] * max(1, (det['bbox'][2]-det['bbox'][0]) * (det['bbox'][3]-det['bbox'][1])),
                'missed': 0,
                'seen': 1 if det['score'] >= THRESHOLD else 0,
                'confirmed': False,
                'ocr_text': '',
                'last_frame': frame_idx
            }
            tracks.append(tnew)
            next_track_id += 1
            # immediate confirm for very confident and large boxes
            w = max(1, tnew['bbox'][2] - tnew['bbox'][0]); h = max(1, tnew['bbox'][3] - tnew['bbox'][1])
            if tnew['best_score'] >= 0.75 and (w * h) >= 1200:
                tnew['confirmed'] = True
                # do OCR immediately
                x1, y1, x2, y2 = [int(v) for v in tnew['best_bbox']]
                pad_x = int((x2 - x1) * ocr_padding_ratio); pad_y = int((y2 - y1) * ocr_padding_ratio)
                x1p = max(0, x1 - pad_x); y1p = max(0, y1 - pad_y)
                x2p = min(width, x2 + pad_x); y2p = min(height, y2 + pad_y)
                crop = tnew['best_frame'][y1p:y2p, x1p:x2p].copy()
                if crop.size != 0:
                    try:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        gray = cv2.bilateralFilter(gray, 5, 75, 75)
                        gray = cv2.equalizeHist(gray)
                        gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)
                        ocr_text = run_ocr(gray)
                    except Exception as e:
                        ocr_text = f"[OCR ERROR: {e}]"
                else:
                    ocr_text = ""
                tnew['ocr_text'] = ocr_text
                name_suffix = ocr_text.replace('[OCR-', '').replace(']', '') if ocr_text else "none"
                img_name = f"{ts}_track{tnew['id']}_{name_suffix}.jpg"
                img_path = os.path.join(UNIQUE_OUTPUT_DIR, img_name)
                full = tnew['best_frame'].copy()
                saver_q.put({'type': 'image', 'path': img_path, 'image': full,
                             'track_id': tnew['id'], 'score': tnew['best_score'],
                             'ocr_text': tnew['ocr_text'], 'timestamp': datetime.now().isoformat(),
                             'video_path': out_video_path, 'bbox': tnew['best_bbox']})

        # prune old tracks
        tracks = [t for t in tracks if t.get('missed', 0) <= max_missed]

        # draw tracks on frame (single window)
        for tr in tracks:
            bx = tr['bbox']
            color = (0, 255, 0) if tr.get('confirmed', False) else (0, 0, 255)
            try:
                cv2.rectangle(out_frame, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
            except Exception:
                pass
            label = f"ID:{tr['id']} S:{tr.get('best_score', 0)*100:.0f}% C:{tr.get('seen', 0)}"
            try:
                cv2.putText(out_frame, label, (bx[0], max(0, bx[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                ocr_disp = tr.get('ocr_text', '')
                if ocr_disp:
                    cv2.putText(out_frame, ocr_disp, (bx[0], min(height - 4, bx[3] + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception:
                pass

        # write and show
        try:
            writer.write(out_frame)
        except Exception as e:
            print("Error al escribir frame:", e)
        if display:
            try:
                cv2.imshow("Detección", out_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                # If imshow fails (headless), ignore
                pass

        pbar.update(1)

    # cleanup
    cap.release()
    writer.release()
    pbar.close()
    try:
        if display:
            cv2.destroyAllWindows()
    except Exception:
        pass

    # stop saver thread and wait until it flushes
    saver_q.put(stop_token)
    saver_thread.join(timeout=15)

    print(f"✔ Resultados guardados en: {UNIQUE_OUTPUT_DIR}")
    return out_video_path


# ---------------------------
# CLI main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Inferir un video con detector de placas")
    parser.add_argument("--video", required=True, help="Ruta a archivo de video")
    parser.add_argument("--model", default=MODEL_PATH, help="Ruta al modelo Keras")
    parser.add_argument("--display", action="store_true", help="Mostrar video en pantalla")
    parser.add_argument("--confirm_frames", type=int, default=1, help="Frames consecutivos para confirmar un track")
    parser.add_argument("--iou_thresh", type=float, default=0.30, help="Umbral IoU para matching")
    parser.add_argument("--min_area", type=int, default=800, help="Área mínima para aceptar detección")
    parser.add_argument("--aspect_ratio_min", type=float, default=1.8)
    parser.add_argument("--aspect_ratio_max", type=float, default=8.0)
    parser.add_argument("--max_missed", type=int, default=10)
    parser.add_argument("--dampening_factor", type=float, default=0.75)
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
