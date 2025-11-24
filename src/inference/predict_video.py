"""Inferencia de video - versión final limpia.
"""

import os
import argparse
from datetime import datetime
import csv
import queue
import threading

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR, THRESHOLD, ROOT_MODEL_DIR, LATEST_MODEL_PATH
from src.utils.mpd_utils import resize_pad, nms_numpy
from src.models.efficient_detector_multi_placa import NUM_CLASSES, yolo_ciou_loss


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
                confidence = float(cell[0])
                if confidence < confidence_threshold:
                    continue
                cx_local = float(cell[1]); cy_local = float(cell[2])
                w_norm = float(cell[3]); h_norm = float(cell[4])
                # Normalizar coordenadas: usar `gw` (grid width) para X y `gh` (grid height) para Y
                cx_norm = (j + cx_local) / gw
                cy_norm = (i + cy_local) / gh
                xmin_norm = cx_norm - (w_norm / 2); ymin_norm = cy_norm - (h_norm / 2)
                xmax_norm = cx_norm + (w_norm / 2); ymax_norm = cy_norm + (h_norm / 2)
                final_boxes_norm.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
                final_scores.append(confidence)
    if len(final_boxes_norm) > 0:
        selected = nms_numpy(final_boxes_norm, final_scores, iou_thresh=0.45, score_thresh=0.0)
        final_boxes = [final_boxes_norm[i] for i in selected]
        final_scores = [final_scores[i] for i in selected]
    else:
        final_boxes = []
        final_scores = []
    return final_boxes, final_scores


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


def process_video(model, video_path, out_video_path=None, img_size=IMG_SIZE[0], display=False,
                  max_missed=5, iou_thresh=0.45, confirm_frames=3, min_area=1500,
                  aspect_ratio_min=2.0, aspect_ratio_max=6.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    saver_q = queue.Queue()
    stop_token = object()
    metadata_csv = os.path.join(OUTPUT_FEED_DIR, 'video_saved_metadata.csv')
    if not os.path.exists(metadata_csv):
        with open(metadata_csv, 'w', newline='') as cf:
            writer_csv = csv.writer(cf)
            writer_csv.writerow(['track_id', 'filepath', 'score', 'timestamp', 'video_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])

    def saver_worker(q, csv_path):
        rows = {}
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', newline='') as cf:
                    reader = csv.DictReader(cf)
                    for r in reader:
                        tid = r.get('track_id', '')
                        rows[tid] = r
            except Exception:
                rows = {}

        while True:
            job = q.get()
            if job is stop_token:
                q.task_done()
                break
            try:
                if job.get('type') == 'image':
                    saved = False
                    try:
                        saved = cv2.imwrite(job['path'], job['image'])
                    except Exception as e:
                        print('cv2.imwrite error:', e)
                        saved = False

                    if saved:
                        tid = str(job.get('track_id', ''))
                        rows[tid] = {
                            'track_id': tid,
                            'filepath': job['path'],
                            'score': str(job.get('score', '')),
                            'timestamp': job.get('timestamp', ''),
                            'video_path': job.get('video_path', ''),
                            'bbox_x1': str(job.get('bbox', ['', '', '', ''])[0]),
                            'bbox_y1': str(job.get('bbox', ['', '', '', ''])[1]),
                            'bbox_x2': str(job.get('bbox', ['', '', '', ''])[2]),
                            'bbox_y2': str(job.get('bbox', ['', '', '', ''])[3])
                        }
                        try:
                            with open(csv_path, 'w', newline='') as cf:
                                fieldnames = ['track_id', 'filepath', 'score', 'timestamp', 'video_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
                                writer_csv = csv.DictWriter(cf, fieldnames=fieldnames)
                                writer_csv.writeheader()
                                for _, r in rows.items():
                                    writer_csv.writerow(r)
                        except Exception as e:
                            print('Error escribiendo CSV:', e)
            except Exception as e:
                print('Saver thread error:', e)
            finally:
                q.task_done()

    saver_thread = threading.Thread(target=saver_worker, args=(saver_q, metadata_csv), daemon=True)
    saver_thread.start()

    tracks = []
    next_track_id = 1

    def iou(a, b):
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

    if display:
        try:
            cv2.namedWindow('Detección', cv2.WINDOW_NORMAL)
            max_w = min(1280, width)
            max_h = min(720, height)
            cv2.resizeWindow('Detección', max_w, max_h)
        except Exception:
            pass

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pad, scale, top, left = resize_pad(rgb, img_size)
        inp = (img_pad.astype(np.float32) / 255.0)[None, ...]
        pred_tensor = model.predict(inp)
        boxes_norm, scores = process_predictions(pred_tensor, img_size=img_size)

        out_frame = frame.copy()
        detections = []
        for box_norm, score in zip(boxes_norm, scores):
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
            # Filtrar detecciones por tamaño y aspecto para reducir falsos positivos (faros/llantas)
            valid_shape = True
            if area < min_area:
                valid_shape = False
            if aspect < aspect_ratio_min or aspect > aspect_ratio_max:
                valid_shape = False
            if not valid_shape:
                continue
            detections.append({'bbox': [x1_orig, y1_orig, x2_orig, y2_orig], 'score': score})

        matched_track_ids = set()
        new_tracks = []

        for det in detections:
            best_tid = None
            best_iou = 0.0
            for t in tracks:
                if t['id'] in matched_track_ids:
                    continue
                i = iou(det['bbox'], t['bbox'])
                if i > best_iou:
                    best_iou = i
                    best_tid = t['id']
            if best_tid is not None and best_iou >= iou_thresh:
                for t in tracks:
                    if t['id'] == best_tid:
                        t['bbox'] = det['bbox']
                        t['missed'] = 0
                        # Actualizar consec/confirmación para estabilidad temporal
                        if det['score'] >= THRESHOLD:
                            t['consec'] = t.get('consec', 0) + 1
                        else:
                            t['consec'] = 0
                        if det['score'] > t.get('best_score', 0):
                            t['best_score'] = det['score']
                        # Confirmar track solo después de N frames consecutivos
                        if not t.get('confirmed', False) and t.get('consec', 0) >= confirm_frames:
                            t['confirmed'] = True
                            # guardar imagen inicial al confirmarse
                            fname = os.path.basename(video_path)
                            img_name = f"det_{fname}_track{t['id']}.jpg"
                            img_path = os.path.join(OUTPUT_FEED_DIR, img_name)
                            full = out_frame.copy()
                            bx = det['bbox']
                            cv2.rectangle(full, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
                            saver_q.put({'type': 'image', 'path': img_path, 'image': full, 'track_id': t['id'], 'score': t['best_score'], 'timestamp': datetime.now().isoformat(), 'video_path': out_video_path, 'bbox': bx})
                        # Si ya confirmado y mejora score, sobrescribir imagen
                        elif t.get('confirmed', False) and det['score'] > t.get('best_score', 0):
                            t['best_score'] = det['score']
                            fname = os.path.basename(video_path)
                            img_name = f"det_{fname}_track{t['id']}.jpg"
                            img_path = os.path.join(OUTPUT_FEED_DIR, img_name)
                            full = out_frame.copy()
                            bx = det['bbox']
                            cv2.rectangle(full, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
                            saver_q.put({'type': 'image', 'path': img_path, 'image': full, 'track_id': t['id'], 'score': t['best_score'], 'timestamp': datetime.now().isoformat(), 'video_path': out_video_path, 'bbox': bx})
                        new_tracks.append(t)
                        matched_track_ids.add(t['id'])
                        break
            else:
                tnew = {'id': next_track_id, 'bbox': det['bbox'], 'best_score': det['score'], 'missed': 0, 'consec': 0, 'confirmed': False}
                next_track_id += 1
                # iniciar consec si supera THRESHOLD
                if tnew['best_score'] >= THRESHOLD:
                    tnew['consec'] = 1
                # confirmar y guardar solo si alcanza consecutividad
                if tnew['consec'] >= confirm_frames:
                    tnew['confirmed'] = True
                    fname = os.path.basename(video_path)
                    img_name = f"det_{fname}_track{tnew['id']}.jpg"
                    img_path = os.path.join(OUTPUT_FEED_DIR, img_name)
                    full = out_frame.copy()
                    bx = det['bbox']
                    cv2.rectangle(full, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
                    saver_q.put({'type': 'image', 'path': img_path, 'image': full, 'track_id': tnew['id'], 'score': tnew['best_score'], 'timestamp': datetime.now().isoformat(), 'video_path': out_video_path, 'bbox': bx})
                new_tracks.append(tnew)

        for t in tracks:
            if t['id'] not in matched_track_ids:
                t['missed'] = t.get('missed', 0) + 1
                if t['missed'] <= max_missed:
                    new_tracks.append(t)
        tracks = [t for t in new_tracks if t.get('missed', 0) <= max_missed]

        for t in tracks:
            bx = t['bbox']
            cv2.rectangle(out_frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
            label = f"Placa: {t.get('best_score', 0)*100:.0f}%"
            cv2.putText(out_frame, label, (bx[0], max(0, bx[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
    saver_q.put(stop_token)
    saver_thread.join(timeout=5)
    print(f"Video de salida guardado en: {out_video_path}")
    return out_video_path


def main():
    parser = argparse.ArgumentParser(description="Inferir un video con detector de placas")
    parser.add_argument("--video", required=True, help="Ruta a archivo de video")
    parser.add_argument("--model", default=MODEL_PATH, help="Ruta al modelo Keras")
    parser.add_argument("--out", default=None, help="Ruta de salida mp4 opcional")
    parser.add_argument("--display", action="store_true", help="Mostrar video en pantalla")
    parser.add_argument("--confirm_frames", type=int, default=3, help="Frames consecutivos necesarios para confirmar un track")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="Umbral IoU para emparejar detecciones con tracks")
    parser.add_argument("--min_area", type=int, default=1500, help="Área mínima en píxeles para aceptar una detección")
    parser.add_argument("--aspect_ratio_min", type=float, default=2.0, help="Relación de aspecto mínima (w/h) para aceptar detecciones")
    parser.add_argument("--aspect_ratio_max", type=float, default=6.0, help="Relación de aspecto máxima (w/h) para aceptar detecciones")
    parser.add_argument("--max_missed", type=int, default=5, help="Máximo de frames perdidos antes de eliminar un track")
    args = parser.parse_args()
    model = load_model_safe(args.model)
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
    )
    print("Hecho.")


if __name__ == "__main__":
    main()
