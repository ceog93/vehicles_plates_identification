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
from src.utils.mpd_utils import resize_pad
from src.inference.inference_utils import (
    load_model_safe, process_predictions, run_ocr, draw_labels
)

# Path al CSV subido (por si quieres calcular anchors o logs)
CSV_LABELS_PATH = '/mnt/data/_processed_data_labels.csv'

os.makedirs(OUTPUT_FEED_DIR, exist_ok=True)

def load_model_safe(model_path):
    return load_model_safe(model_path)

def infer_image(model, img_path, out_dir=None, img_size=None):
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
    
    track_id_counter = 1 # Usar un ID simple para cada detección en la imagen
    for box_norm, score in zip(boxes_norm, scores):
        # Mapear de coordenadas normalizadas a píxeles de la imagen original
        x1_p = int(box_norm[0] * img_size); y1_p = int(box_norm[1] * img_size)
        x2_p = int(box_norm[2] * img_size); y2_p = int(box_norm[3] * img_size)

        x1_orig = int(max(0, (x1_p - left) / scale)); y1_orig = int(max(0, (y1_p - top) / scale))
        x2_orig = int(min(orig_w, (x2_p - left) / scale)); y2_orig = int(min(orig_h, (y2_p - top) / scale))
        
        bbox_orig = [x1_orig, y1_orig, x2_orig, y2_orig]

        # Recortar la región de la placa para OCR
        crop = img_bgr[y1_orig:y2_orig, x1_orig:x2_orig].copy()
        ocr_data = {'plate': '', 'city': ''}
        if crop.size > 0:
            ocr_data = run_ocr(crop)
        
        # Dibujar las etiquetas usando la función compartida
        out_bgr = draw_labels(out_bgr, bbox_orig, track_id_counter, score, ocr_data)
        track_id_counter += 1

    # Si no se especifica un directorio de salida, se crea uno nuevo para esta imagen.
    if out_dir is None:
        ts_folder = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(OUTPUT_FEED_DIR, f"img_{ts_folder}")
        os.makedirs(out_dir, exist_ok=True)

    # Construir el nombre del archivo con el formato estandarizado
    # Nota: para una sola imagen, solo hay una detección, así que tomamos el último OCR.
    ts_file = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ocr_dict = ocr_data if isinstance(ocr_data, dict) else {}
    plate_text = ocr_dict.get('plate', '')
    suffix = plate_text if plate_text and "ERR" not in plate_text else "placa"
    # Usamos el último ID de detección para el nombre del archivo
    out_path = os.path.join(out_dir, f"{ts_file}_ID_{track_id_counter-1}_{suffix}.jpg")
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
