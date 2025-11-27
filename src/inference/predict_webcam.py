# src/inference/predict_webcam.py
"""
Inferencia en vivo desde webcam (o índice de cámara):
- Utiliza la misma lógica de tracking y guardado que predict_video.py
- Procesa OCR y guardado en hilos para no afectar el rendimiento.
"""

import os
import argparse
import cv2
import numpy as np
from datetime import datetime
import queue
from tqdm import tqdm

# Importar lógica compartida
from src.inference.inference_utils import (
    load_model_safe, process_predictions, match_detections_to_tracks,
    start_saver_thread, start_ocr_thread, draw_bbox_safe, draw_labels, _center, open_folder
)
from src.utils.image_bbox_utils import resize_pad
from src.config import IMG_SIZE, OUTPUT_FEED_DIR, THRESHOLD

def list_cameras(max_index: int = 8, warmup_frames: int = 2):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

def run_webcam(cam_index=0, model=None, img_size=IMG_SIZE[0], display=True,
               max_missed=10, iou_thresh=0.30, confirm_frames=3, min_area=400,
               aspect_ratio_min=1.8, aspect_ratio_max=8.0, dampening_factor=0.85):
    
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir la cámara index {cam_index}")

    # Preparar escritor de video si se solicitó
    writer = None
    out_video_path = None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 20

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    UNIQUE_OUTPUT_DIR = os.path.join(OUTPUT_FEED_DIR, f"webcam_{ts}")
    os.makedirs(UNIQUE_OUTPUT_DIR, exist_ok=True)

    out_video_path = os.path.join(UNIQUE_OUTPUT_DIR, f"webcam_record_{ts}.mp4")
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Cargar el modelo si no se proporcionó (para flexibilidad)
    if model is None:
        model = load_model_safe()

    # Iniciar hilos de guardado y OCR
    saver_q = queue.Queue()
    saver_thread, stop_token = start_saver_thread(saver_q, os.path.join(UNIQUE_OUTPUT_DIR, 'metadata.csv'))
    ocr_q = queue.Queue()
    ocr_results = {}
    ocr_thread, ocr_stop_token = start_ocr_thread(ocr_q, ocr_results)

    # Estructuras de tracking
    frame_idx = 0
    tracks = []
    next_track_id = 1

    if display:
        cv2.namedWindow("Webcam - Detección", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Webcam - Detección", min(1280, width), min(720, height))

    pbar = tqdm(desc="Procesando Webcam")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        bgr = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pad, scale, top, left = resize_pad(rgb, img_size)
        inp = (img_pad.astype(np.float32) / 255.0)[None, ...]

        pred_tensor = model.predict(inp, verbose=0)
        boxes_norm, scores = process_predictions(pred_tensor, img_size=img_size)

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
            detections, tracks, iou_thr=iou_thresh
        )
        OCR_CANDIDATE_SCORE = 0.65

        # Actualizar tracks emparejados
        for (t_idx, d_idx) in matches:
            tr = tracks[t_idx]; det = detections[d_idx]
            alpha = float(dampening_factor)
            tb = tr['bbox']; db = det['bbox']
            current_velocity = np.array(_center(db)) - np.array(_center(tb))
            tr['bbox'] = [int(alpha * tb[i] + (1 - alpha) * db[i]) for i in range(4)]
            old_velocity = tr.get('velocity', np.array([0.0, 0.0]))
            tr['velocity'] = 0.7 * current_velocity + 0.3 * old_velocity
            tr.update({
                'last_detection_bbox': det['bbox'].copy(), 'current_score': det['score'],
                'missed': 0, 'seen': tr.get('seen', 0) + 1, 'age': tr.get('age', 0) + 1,
                'last_frame': frame_idx, 'last_seen_conf': det['score']
            })
            if det['score'] >= OCR_CANDIDATE_SCORE and tr['id'] not in ocr_results:
                bx1, by1, bx2, by2 = [int(v) for v in det['bbox']]
                crop = bgr[by1:by2, bx1:bx2].copy()
                if crop.size > 0: ocr_q.put({'track_id': tr['id'], 'image': crop})
            if tr['id'] in ocr_results: tr['ocr_text'] = ocr_results[tr['id']]

            should_save = False
            if tr.get('confirmed', False):
                current_score = det['score']
                ocr_dict = tr.get('ocr_text', {})
                plate_text = ocr_dict.get('plate', '')
                ocr_is_valid = plate_text and 'ERR' not in plate_text and '[OCR' not in plate_text
                if current_score >= 0.90 and ocr_is_valid and current_score > tr.get('saved_score', 0):
                    should_save = True
                    tr.update({'best_frame': bgr.copy(), 'best_bbox': det['bbox'].copy(), 'best_score': current_score})
            
            if should_save:
                best_bbox = tr.get('best_bbox', tr['bbox'])
                img_to_save = tr['best_frame'].copy()
                img_to_save = draw_labels(img_to_save, best_bbox, tr['id'], tr['best_score'], tr['ocr_text'])
                plate_text = tr['ocr_text'].get('plate', '')
                suffix = plate_text if plate_text and "ERR" not in plate_text else "placa"
                img_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                img_name = f"{img_ts}_{tr['id']}_{suffix}.jpg"
                tr['saved_score'] = tr['best_score']
                saver_q.put({
                    'type': 'image', 'path': os.path.join(UNIQUE_OUTPUT_DIR, img_name), 'image': img_to_save,
                    'track_id': tr['id'], 'score': tr.get('best_score', 0), 'ocr_text': tr['ocr_text'],
                    'timestamp': datetime.now().isoformat(), 'video_path': out_video_path, 'bbox': best_bbox
                })

        # Actualizar tracks no emparejados
        for ti in unmatched_tracks_idx:
            tracks[ti]['missed'] += 1
            tracks[ti]['age'] += 1
            velocity = tracks[ti].get('velocity', np.array([0.0, 0.0]))
            for i in range(4): tracks[ti]['bbox'][i] += int(velocity[i % 2])

        # Crear nuevos tracks
        for di in unmatched_dets_idx:
            det = detections[di]
            w = det['bbox'][2] - det['bbox'][0]; h = det['bbox'][3] - det['bbox'][1]
            if det['score'] < 0.25 or (w*h) < (min_area * 0.5): continue
            tnew = {
                'id': next_track_id, 'bbox': det['bbox'].copy(), 'last_detection_bbox': det['bbox'].copy(),
                'best_frame': bgr.copy(), 'best_bbox': det['bbox'].copy(), 'best_score': det['score'],
                'missed': 0, 'seen': 1, 'confirmed': False, 'ocr_text': {'plate': '', 'city': ''},
                'last_frame': frame_idx, 'saved_score': 0.0, 'age': 1, 'last_seen_conf': det['score']
            }
            if tnew['best_score'] >= OCR_CANDIDATE_SCORE:
                bx1, by1, bx2, by2 = [int(v) for v in det['bbox']]
                crop = bgr[by1:by2, bx1:bx2].copy()
                if crop.size > 0: ocr_q.put({'track_id': tnew['id'], 'image': crop})
            tnew['ocr_text'] = ocr_results.get(tnew['id'], {'plate': '', 'city': ''})
            tnew['current_score'] = det['score']
            tracks.append(tnew); next_track_id += 1
            if (tnew['best_score'] >= 0.50) or (tnew['best_score'] >= 0.35 and (w*h) >= 800):
                tnew['confirmed'] = True

        # Dibujar en el frame de salida
        out_frame = frame.copy()
        for tr in tracks:
            if tr['id'] in ocr_results: tr['ocr_text'] = ocr_results[tr['id']]
            display_bbox = tr['bbox']
            display_score = tr.get('current_score', tr.get('best_score', 0))
            color = (0, 255, 0) if tr.get('confirmed', False) else (0, 165, 255)
            ocr_disp = tr.get('ocr_text', {'plate': '', 'city': ''})
            out_frame = draw_bbox_safe(out_frame, display_bbox, tr['id'], display_score, ocr_disp, color)

        # Eliminar tracks expirados
        tracks = [t for t in tracks if t.get('missed', 0) <= (max_missed * 2 if t.get('confirmed') else max_missed)]

        if display:
            cv2.imshow("Webcam - Detección", out_frame)
        
        if writer:
            writer.write(out_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        pbar.update(1)

    cap.release()
    if writer: writer.release()
    pbar.close()
    if display: cv2.destroyAllWindows()
    saver_q.put(stop_token); saver_thread.join(timeout=5)
    saver_q.join() # Esperar a que la cola de guardado se vacíe
    ocr_q.put(ocr_stop_token); ocr_thread.join(timeout=5)
    print(f"✔ Proceso de webcam finalizado. Resultados en: {UNIQUE_OUTPUT_DIR}")

    # Abrir la carpeta de resultados
    open_folder(UNIQUE_OUTPUT_DIR)

def main():
    parser = argparse.ArgumentParser(description="Inferencia en webcam con tracking avanzado.")
    parser.add_argument("--cam", type=int, default=0, help="Índice de la cámara (default 0).")
    parser.add_argument("--model", default=None, help="Ruta al modelo Keras. Si no se especifica, busca el último.")
    parser.add_argument("--no-display", action="store_true", help="No mostrar la ventana de visualización.")
    args = parser.parse_args()

    print("Cámaras disponibles:", list_cameras())
    model = load_model_safe(args.model)
    run_webcam(model, cam_index=args.cam, display=not args.no_display)

if __name__ == "__main__":
    main()
