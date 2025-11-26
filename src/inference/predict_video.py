# src/inference/predict_video.py
# Versión Final Definitiva: Etiquetas Completas y Lógica de Tracking
import os
import argparse
from datetime import datetime
import queue
import threading

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------
# CONFIGURACIÓN DEL ENTORNO TF
# ---------------------------
import tensorflow as tf
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# OCR - Importación segura
try:
    from src.inference.inference_utils import READER
except ImportError:
    READER = None

# Importaciones del proyecto
from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR, THRESHOLD
from src.utils.mpd_utils import resize_pad
from src.inference.inference_utils import (
    load_model_safe, process_predictions, match_detections_to_tracks,
    start_saver_thread, start_ocr_thread, draw_bbox_safe, draw_labels, _center, open_folder
)

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

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
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

        bgr = frame.copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_pad, scale, top, left = resize_pad(rgb, img_size)
        inp = (img_pad.astype(np.float32) / 255.0)[None, ...]

        pred_tensor = model.predict(inp, verbose=0)
        boxes_norm, scores = process_predictions(pred_tensor, img_size=img_size)

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
            detections, tracks, iou_thr=iou_thresh
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
            ocr_dict = tr.get('ocr_text', {}) if isinstance(tr.get('ocr_text'), dict) else {}
            current_ocr_fail = ('ERR' in ocr_dict.get('plate', '')) or not ocr_dict.get('plate')
            
            if (quality > tr.get('best_area', 0) * 1.05) or \
               (det['score'] > tr.get('best_score', 0) + 0.10) or \
               (current_ocr_fail and det['score'] > 0.70): # Intentar mejorar la captura si el OCR falló
                
                tr['best_area'] = quality
                tr['best_score'] = det['score']
                tr['best_frame'] = bgr.copy()
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
            if tr.get('confirmed', False):
                # 1. Asegurarse de que el texto del OCR esté actualizado ANTES de tomar la decisión.
                if tr['id'] in ocr_results:
                    tr['ocr_text'] = ocr_results[tr['id']]
                
                # 2. Evaluar las condiciones con la información más reciente.
                current_score = det['score']
                ocr_dict = tr.get('ocr_text', {})
                plate_text = ocr_dict.get('plate', '')
                ocr_is_valid = plate_text and 'ERR' not in plate_text and '[OCR' not in plate_text

                if current_score >= 0.90 and ocr_is_valid:
                    if current_score > tr.get('saved_score', 0):
                        should_save = True
                        # Actualizar la mejor captura con la detección actual
                        tr['best_frame'] = bgr.copy()
                        tr['best_bbox'] = det['bbox'].copy()
                        tr['best_score'] = current_score

            if should_save:
                best_bbox = tr.get('best_bbox', tr['bbox'])

                img_to_save = tr['best_frame'].copy()
                img_to_save = draw_labels(img_to_save, best_bbox, tr['id'], tr['best_score'], tr['ocr_text'])

                ocr_dict = tr.get('ocr_text', {})
                plate_text = ocr_dict.get('plate', '')
                suffix = plate_text if plate_text and "ERR" not in plate_text else "placa"
                img_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # Timestamp en tiempo real para la imagen
                img_name = f"{img_ts}_{tr['id']}_{suffix}.jpg"

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
                'ocr_text': {'plate': '', 'city': ''}, # Inicializar como diccionario
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
            initial_ocr = ocr_results.get(tnew['id'], {'plate': '', 'city': ''})
            tnew['ocr_text'] = initial_ocr
            
            tnew['detection_history'] = [
                {'bbox': det['bbox'].copy(), 'score': det['score'], 'frame': bgr.copy(), 'frame_idx': frame_idx, 'ocr_text': initial_ocr.copy()}
            ]
            tnew['current_score'] = det['score']
            tracks.append(tnew); next_track_id += 1

            # Confirmar un track nuevo si la detección inicial es de calidad razonable
            if (tnew['best_score'] >= 0.50) or (tnew['best_score'] >= 0.35 and (w*h) >= 800):
                tnew['confirmed'] = True

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
            ocr_disp = tr.get('ocr_text', {'plate': '', 'city': ''})
            
            # DIBUJAR SIEMPRE si el track está activo
            out_frame = draw_bbox_safe(out_frame, display_bbox, tr['id'], display_score, ocr_disp, color)
            drawn_count += 1
        
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
    saver_q.join() # Esperar a que la cola de guardado se vacíe
    ocr_q.put(ocr_stop_token); ocr_thread.join(timeout=15)
    # Esperar a que la cola de OCR se vacíe antes del guardado final
    ocr_q.join()

    print(f"✔ Resultados guardados en: {UNIQUE_OUTPUT_DIR}")
    return UNIQUE_OUTPUT_DIR

def main():
    parser = argparse.ArgumentParser(description="Inferir video con etiquetas completas (ID, Confianza, Placa)")
    parser.add_argument("--video", required=False, default='input_feed', help="Video entrada o 'input_feed' para procesar la carpeta 03_production/input_feed/")
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