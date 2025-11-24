# src/inference/predict_webcam.py
"""
Inferencia en vivo desde webcam (o índice de cámara):
- Preprocesa cada frame con resize_pad
- Predice bbox y lo mapea al frame original
- Muestra en pantalla (y opcionalmente guarda frames en OUTPUT_FEED_DIR)
"""

import os
import argparse
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
import threading
import queue
import csv
import time

from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR, ROOT_MODEL_DIR, LATEST_MODEL_PATH
from src.utils.mpd_utils import resize_pad, nms_numpy
from src.models.efficient_detector_multi_placa import NUM_ANCHORS, NUM_CLASSES, yolo_ciou_loss
import numpy as np

CSV_LABELS_PATH = '/mnt/data/_processed_data_labels.csv'
os.makedirs(OUTPUT_FEED_DIR, exist_ok=True)

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


def list_cameras(max_index: int = 8, warmup_frames: int = 2):
    """Intenta abrir índices de cámara y devuelve lista de índices disponibles.

    Se prueba desde 0 hasta max_index-1, intentando leer un frame para
    confirmar que la cámara funciona.
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        ok = False
        for _ in range(warmup_frames):
            ret, _ = cap.read()
            if ret:
                ok = True
                break
        cap.release()
        if ok:
            available.append(i)
    return available

def run_webcam(model, cam_index=0, img_size=IMG_SIZE[0], save_output=False, save_dir=OUTPUT_FEED_DIR,
               save_video=False, save_high_conf=False, high_conf_thresh=0.9):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir la cámara index {cam_index}")

    # Preparar escritor de video si se solicitó
    writer = None
    out_video_path = None
    # Preparar cola e hilo de guardado asíncrono si se solicita guardado
    saver_q = None
    saver_thread = None
    metadata_csv = None
    stop_token = object()
    def saver_worker(q, csv_path):
        # worker que procesa jobs de guardado: {'type':'image'|'video', ...}
        # asegura que el CSV de metadatos existe y escribe filas
        while True:
            job = q.get()
            if job is stop_token:
                q.task_done()
                break
            try:
                if job['type'] == 'image':
                    # write image to disk
                    cv2.imwrite(job['path'], job['image'])
                    # append metadata
                    with open(csv_path, 'a', newline='') as cf:
                        writer = csv.writer(cf)
                        writer.writerow([job.get('track_id', ''), job['path'], job.get('is_full', False), job.get('score', ''), job.get('timestamp', ''), *job.get('bbox', ['','','',''])])
                elif job['type'] == 'video':
                    # nothing to do besides record path
                    with open(csv_path, 'a', newline='') as cf:
                        writer = csv.writer(cf)
                        writer.writerow(['video', job['path'], '', job.get('timestamp', ''), '', '', '', ''])
            except Exception as e:
                print('Saver thread error:', e)
            finally:
                q.task_done()
    if save_video:
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_video_path = os.path.join(save_dir, f"webcam_{ts}.mp4")
        # intentar recuperar FPS de la cámara
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 20.0
        # tamaño del frame (se actualizará al primer frame si es necesario)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

    # iniciar saver thread si guardado activo
    if save_video or save_high_conf:
        saver_q = queue.Queue()
        metadata_csv = os.path.join(save_dir, 'webcam_saved_metadata.csv')
        # si no existe, crear cabecera
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(metadata_csv):
            with open(metadata_csv, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(['track_id','filepath','is_full','score','timestamp','bbox_x1','bbox_y1','bbox_x2','bbox_y2'])
        saver_thread = threading.Thread(target=saver_worker, args=(saver_q, metadata_csv), daemon=True)
        saver_thread.start()

    # tracking structures: se inicializan antes del bucle
    frame_idx = 0
    tracks = []
    track_id_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if img_size is None:
            try:
                mshape = model.input_shape
                if mshape and len(mshape) >= 3 and mshape[1] is not None:
                    target_size = int(mshape[1])
                else:
                    target_size = IMG_SIZE[0]
            except Exception:
                target_size = IMG_SIZE[0]
        else:
            target_size = img_size
        img_pad, scale, top, left = resize_pad(rgb, target_size)
        inp = (img_pad.astype(np.float32) / 255.0)[None, ...]

        pred = model.predict(inp)
        arr = np.asarray(pred)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]

        out_frame = frame.copy()

        # Si la salida es una única caja de 4 valores, mantener compatibilidad
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 4):
            box = np.asarray(arr).reshape(-1)
            x1_p = int(box[0] * target_size)
            y1_p = int(box[1] * target_size)
            x2_p = int(box[2] * target_size)
            y2_p = int(box[3] * target_size)

            h, w = frame.shape[:2]
            x1_orig = int(max(0, (x1_p - left) / scale))
            y1_orig = int(max(0, (y1_p - top) / scale))
            x2_orig = int(min(w, (x2_p - left) / scale))
            y2_orig = int(min(h, (y2_p - top) / scale))

            cv2.rectangle(out_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (0,255,0), 2)
            label = "Placa"
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            rect_pt1 = (x1_orig, max(0, y1_orig - ts[1] - 8))
            rect_pt2 = (x1_orig + ts[0] + 8, max(0, y1_orig))
            cv2.rectangle(out_frame, rect_pt1, rect_pt2, (0,255,0), -1)
            cv2.putText(out_frame, label, (x1_orig + 4, max(0, y1_orig - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        else:
            # Decodificar salida YOLO-like
            gh, gw, channels = arr.shape[:3]
            per_anchor = 5 + NUM_CLASSES
            anchors = int(channels // per_anchor)
            arr = arr.reshape(gh, gw, anchors, per_anchor)

            boxes = []
            scores = []
            for i in range(gh):
                for j in range(gw):
                    for a in range(anchors):
                        cell = arr[i, j, a]
                        conf = float(cell[0])
                        if conf < 0.2:
                            continue
                        cx_local = float(cell[1])
                        cy_local = float(cell[2])
                        w_norm = float(cell[3])
                        h_norm = float(cell[4])

                        cx = (j + cx_local) / gh
                        cy = (i + cy_local) / gh
                        xmin = cx - w_norm / 2
                        ymin = cy - h_norm / 2
                        xmax = cx + w_norm / 2
                        ymax = cy + h_norm / 2

                        x1_p = int(xmin * target_size)
                        y1_p = int(ymin * target_size)
                        x2_p = int(xmax * target_size)
                        y2_p = int(ymax * target_size)

                        h, w = frame.shape[:2]
                        x1_orig = int(max(0, (x1_p - left) / scale))
                        y1_orig = int(max(0, (y1_p - top) / scale))
                        x2_orig = int(min(w, (x2_p - left) / scale))
                        y2_orig = int(min(h, (y2_p - top) / scale))

                        boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
                        scores.append(conf)

            if len(boxes) > 0:
                selected = nms_numpy(boxes, scores, iou_thresh=0.45, score_thresh=0.2)
                for idx in selected:
                    b = boxes[int(idx)]
                    s = scores[int(idx)]
                    cv2.rectangle(out_frame, (b[0], b[1]), (b[2], b[3]), (0,255,0), 2)
                    # Mostrar confianza en porcentaje (1 decimal) y fondo para legibilidad
                    label = f"Placa: {s*100:.1f}%"
                    ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    rect_pt1 = (b[0], max(0, b[1] - ts[1] - 8))
                    rect_pt2 = (b[0] + ts[0] + 8, max(0, b[1]))
                    cv2.rectangle(out_frame, rect_pt1, rect_pt2, (0,255,0), -1)
                    cv2.putText(out_frame, label, (b[0] + 4, max(0, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.imshow("Webcam - Detección", out_frame)
        # Escribir frame al video si corresponde
        if save_video:
            if writer is None:
                # crear writer con el tamaño del frame
                h_f, w_f = out_frame.shape[:2]
                writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w_f, h_f))
            writer.write(out_frame)

        # Tracking y guardado selectivo por placa (guardar mejor frame por track)
        if save_high_conf:
            # Construir lista de detecciones (solo si boxes/scores existen)
            detections = []
            if 'boxes' in locals() and len(boxes) > 0:
                # selected es la lista de índices retornada por NMS
                try:
                    sel = selected
                except NameError:
                    sel = range(len(boxes))
                for idx in sel:
                    b = boxes[int(idx)]
                    s = float(scores[int(idx)])
                    if s >= high_conf_thresh:
                        detections.append((b, s))

            # función IoU para comparar bboxes
            def iou(boxA, boxB):
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interW = max(0, xB - xA)
                interH = max(0, yB - yA)
                inter = interW * interH
                areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
                areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
                union = areaA + areaB - inter + 1e-6
                return inter / union

            # Actualizar tracks con detecciones del frame
            for bbox, s in detections:
                matched = False
                for t in tracks:
                    if iou(bbox, t['bbox']) >= 0.4:
                        matched = True
                        t['last_seen'] = frame_idx
                        t['bbox'] = bbox
                        if s > t['best_score']:
                            t['best_score'] = s
                            t['best_frame'] = out_frame.copy()
                            t['best_bbox'] = bbox
                            # guardar inmediatamente el mejor frame y el crop (sobrescribe previo)
                            try:
                                x1, y1, x2, y2 = [int(v) for v in bbox]
                                h_img, w_img = out_frame.shape[:2]
                                x1c = max(0, min(x1, w_img-1))
                                x2c = max(0, min(x2, w_img))
                                y1c = max(0, min(y1, h_img-1))
                                y2c = max(0, min(y2, h_img))
                                crop = out_frame[y1c:y2c, x1c:x2c].copy()
                                fname_crop = os.path.join(save_dir, f"best_plate_{t['id']}.jpg")
                                fname_full = os.path.join(save_dir, f"best_plate_full_{t['id']}.jpg")
                                # push save jobs to queue (overwrite by same filename)
                                if saver_q is not None:
                                    saver_q.put({'type':'image','path':fname_crop,'image':crop,'track_id':t['id'],'is_full':False,'score':t['best_score'],'timestamp':datetime.now().isoformat(),'bbox':[x1c,y1c,x2c,y2c]})
                                    full = out_frame.copy()
                                    cv2.rectangle(full, (x1c, y1c), (x2c, y2c), (0,255,0), 2)
                                    saver_q.put({'type':'image','path':fname_full,'image':full,'track_id':t['id'],'is_full':True,'score':t['best_score'],'timestamp':datetime.now().isoformat(),'bbox':[x1c,y1c,x2c,y2c]})
                                else:
                                    # fallback sync write
                                    cv2.imwrite(fname_crop, crop)
                                    full = out_frame.copy()
                                    cv2.rectangle(full, (x1c, y1c), (x2c, y2c), (0,255,0), 2)
                                    cv2.imwrite(fname_full, full)
                                print(f"Updated best frame for track {t['id']} (score={t['best_score']:.3f}): {fname_crop}")
                            except Exception as e:
                                print('Error saving best frame for track', t['id'], e)
                        break
                if not matched:
                    new_t = {
                        'id': track_id_counter,
                        'bbox': bbox,
                        'best_score': s,
                        'best_frame': out_frame.copy(),
                        'best_bbox': bbox,
                        'last_seen': frame_idx
                    }
                    tracks.append(new_t)
                    # guardar inmediatamente el primer best frame para este track
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        h_img, w_img = out_frame.shape[:2]
                        x1c = max(0, min(x1, w_img-1))
                        x2c = max(0, min(x2, w_img))
                        y1c = max(0, min(y1, h_img-1))
                        y2c = max(0, min(y2, h_img))
                        crop = out_frame[y1c:y2c, x1c:x2c].copy()
                        fname_crop = os.path.join(save_dir, f"best_plate_{new_t['id']}.jpg")
                        fname_full = os.path.join(save_dir, f"best_plate_full_{new_t['id']}.jpg")
                        if saver_q is not None:
                            saver_q.put({'type':'image','path':fname_crop,'image':crop,'track_id':new_t['id'],'is_full':False,'score':s,'timestamp':datetime.now().isoformat(),'bbox':[x1c,y1c,x2c,y2c]})
                            full = out_frame.copy()
                            cv2.rectangle(full, (x1c, y1c), (x2c, y2c), (0,255,0), 2)
                            saver_q.put({'type':'image','path':fname_full,'image':full,'track_id':new_t['id'],'is_full':True,'score':s,'timestamp':datetime.now().isoformat(),'bbox':[x1c,y1c,x2c,y2c]})
                        else:
                            cv2.imwrite(fname_crop, crop)
                            full = out_frame.copy()
                            cv2.rectangle(full, (x1c, y1c), (x2c, y2c), (0,255,0), 2)
                            cv2.imwrite(fname_full, full)
                        print(f"Saved initial best frame for track {new_t['id']} (score={s:.3f}): {fname_crop}")
                    except Exception as e:
                        print('Error saving best frame for new track', new_t['id'], e)
                    track_id_counter += 1
        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if save_output and key == ord('s'):
            fname = os.path.join(save_dir, f"webcam_{int(tf.timestamp())}.jpg")
            cv2.imwrite(fname, out_frame)
            print("Frame guardado:", fname)
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Video guardado en: {out_video_path}")
    # Al finalizar, guardar mejores frames por track (si corresponde)
    if save_high_conf and len(tracks) > 0:
        os.makedirs(save_dir, exist_ok=True)
        for t in tracks:
            try:
                score = t.get('best_score', 0.0)
                bf = t.get('best_frame', None)
                bb = t.get('best_bbox', None)
                if bf is None or bb is None:
                    continue
                # guardar crop de la placa y la imagen completa con bbox
                x1, y1, x2, y2 = [int(v) for v in bb]
                h_img, w_img = bf.shape[:2]
                x1c = max(0, min(x1, w_img-1))
                x2c = max(0, min(x2, w_img))
                y1c = max(0, min(y1, h_img-1))
                y2c = max(0, min(y2, h_img))
                crop = bf[y1c:y2c, x1c:x2c].copy()
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                fname_crop = os.path.join(save_dir, f"best_plate_{t['id']}_{int(score*1000)}_{ts}.jpg")
                cv2.imwrite(fname_crop, crop)
                # guardar full frame con bbox dibujada
                full = bf.copy()
                cv2.rectangle(full, (x1c, y1c), (x2c, y2c), (0,255,0), 2)
                fname_full = os.path.join(save_dir, f"best_plate_full_{t['id']}_{int(score*1000)}_{ts}.jpg")
                cv2.imwrite(fname_full, full)
                print(f"Saved best frame for track {t['id']} (score={score:.3f}): {fname_crop}")
            except Exception as e:
                print('Error saving best frame for track', t.get('id'), e)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Inferencia en webcam con MPD detector")
    parser.add_argument("--cam", type=int, default=0, help="Índice de la cámara (default 0)")
    parser.add_argument("--model", default=MODEL_PATH, help="Ruta al modelo Keras")
    parser.add_argument("--save", action="store_true", help="Permitir guardar frames con 's'")
    args = parser.parse_args()

    model = load_model_safe(args.model)
    run_webcam(model, cam_index=args.cam, save_output=args.save)

if __name__ == "__main__":
    main()
