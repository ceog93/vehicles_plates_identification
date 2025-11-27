# src/inference/predict_image.py
"""
Inferencia sobre UNA imagen:
- Usa resize_pad para mantener aspecto (letterbox)
- Ejecuta model.predict sobre la imagen preprocesada
- Convierte la bbox normalizada a coordenadas del frame original
- Dibuja y guarda la imagen con la detección
"""

import os
import sys
from datetime import datetime
import argparse
import cv2
import numpy as np
import tensorflow as tf

from src.config import IMG_SIZE, OUTPUT_FEED_DIR
from src.utils.image_bbox_utils import resize_pad
from src.inference.inference_utils import (
    load_model_safe, process_predictions, run_ocr, draw_labels
)

# Path al CSV subido (por si quieres calcular anchors o logs)
CSV_LABELS_PATH = '/mnt/data/_processed_data_labels.csv'

os.makedirs(OUTPUT_FEED_DIR, exist_ok=True)

def load_model_safe(model_path):
    return load_model_safe(model_path)

def infer_image(model, img_path, out_dir=None, img_size=None, saver_q=None, ocr_q=None, ocr_results=None):
    # Leer imagen original
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Preprocesamiento: letterbox
    # Determinar tamaño esperado por el modelo (si no se pasó explicitamente)
    if img_size is None:
        try:
            mshape = model.input_shape
            # input_shape puede ser (None, H, W, C)
            if mshape and len(mshape) >= 3 and mshape[1] is not None:
                img_size = int(mshape[1])
            else:
                img_size = IMG_SIZE[0]
        except Exception:
            img_size = IMG_SIZE[0]

    img_pad, scale, top, left = resize_pad(img_rgb, img_size)
    inp = img_pad.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, 0)  # batch=1

    # Predict y post-procesar
    pred_tensor = model.predict(inp, verbose=0)
    boxes_norm, scores = process_predictions(pred_tensor, img_size=img_size)

    out_bgr = img_bgr.copy() # Usar una copia de la imagen original para dibujar
    
    # Inicializar ocr_data para evitar NameError si no hay detecciones
    # Usaremos un diccionario para mapear track_id a ocr_data
    detections_data = {}

    track_id_counter = 1 # Usar un ID simple para cada detección en la imagen
    for box_norm, score in zip(boxes_norm, scores):
        # Mapear de coordenadas normalizadas a píxeles de la imagen original
        x1_p = int(box_norm[0] * img_size); y1_p = int(box_norm[1] * img_size)
        x2_p = int(box_norm[2] * img_size); y2_p = int(box_norm[3] * img_size)

        x1_orig = int(max(0, (x1_p - left) / scale)); y1_orig = int(max(0, (y1_p - top) / scale))
        x2_orig = int(min(orig_w, (x2_p - left) / scale)); y2_orig = int(min(orig_h, (y2_p - top) / scale))
        
        bbox_orig = [x1_orig, y1_orig, x2_orig, y2_orig]

        # 1. Enviar todos los trabajos de OCR a la cola
        if ocr_q is not None and ocr_results is not None:
            crop = img_bgr[y1_orig:y2_orig, x1_orig:x2_orig].copy()
            if crop.size > 0:
                ocr_q.put({'track_id': track_id_counter, 'image': crop})
        
        # Guardar la información de la detección sin el OCR por ahora
        detections_data[track_id_counter] = {'bbox': bbox_orig, 'score': score}
        track_id_counter += 1

    # 2. Esperar a que el hilo de OCR termine todos los trabajos para esta imagen
    if ocr_q is not None:
        ocr_q.join()

    # 3. Ahora que el OCR ha terminado, actualizar los datos y dibujar las etiquetas
    for tid, data in detections_data.items():
        # Actualizar la entrada de la detección con su resultado de OCR
        ocr_data = ocr_results.get(tid, {'plate': '', 'city': ''}) if ocr_results is not None else {'plate': '', 'city': ''}
        data['ocr'] = ocr_data

        # Dibujar las etiquetas usando la función compartida
        out_bgr = draw_labels(out_bgr, data['bbox'], tid, data['score'], data['ocr'])

    # Solo guardar la imagen si se encontró al menos una detección
    if track_id_counter > 1:
        if out_dir is None:
            ts_folder = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_dir = os.path.join(OUTPUT_FEED_DIR, f"img_{ts_folder}")
            os.makedirs(out_dir, exist_ok=True)

        # Construir el nombre del archivo con el formato solicitado
        ts_file = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        first_det_data = detections_data.get(1, {})
        plate_text = first_det_data.get('ocr', {}).get('plate', '')
        suffix = plate_text if plate_text and "ERR" not in plate_text else "placa"
        out_path = os.path.join(out_dir, f"{ts_file}_ID_1_{suffix}.jpg")
        
        # Si se está usando el sistema de colas, enviar al saver_thread
        if saver_q is not None:
            # FIX: Definir la variable que faltaba obteniendo los datos de la primera detección
            first_det_data = detections_data.get(1, {})
            # Enviamos la primera detección como representativa para el CSV
            saver_q.put({'type': 'image', 'path': out_path, 'image': out_bgr, 'track_id': 1, 
                         'score': first_det_data.get('score', 0), 'ocr_text': first_det_data.get('ocr', {}),
                         'timestamp': datetime.now().isoformat(), 'bbox': first_det_data.get('bbox', [])})
            print(f"Enviado a la cola de guardado: {out_path}")
        else: # Fallback a guardado síncrono
            cv2.imwrite(out_path, out_bgr)
            print(f"Resultado guardado en: {out_path}")

    return out_path

def main():
    parser = argparse.ArgumentParser(description="Inferir una imagen con detector de placas")
    parser.add_argument("--image", required=True, help="Ruta a la imagen de entrada.")
    parser.add_argument("--model", default=None, help="Ruta al modelo Keras. Si no, busca el último.")
    parser.add_argument("--out", default=None, help="Ruta de salida opcional")
    args = parser.parse_args()

    model = load_model_safe(args.model)
    out = infer_image(model, args.image, out_dir=args.out)
    print("Hecho.")

if __name__ == "__main__":
    main()
