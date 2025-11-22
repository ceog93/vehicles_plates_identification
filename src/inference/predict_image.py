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
import argparse
import cv2
import numpy as np
import tensorflow as tf

from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR
from src.utils.mpd_utils import resize_pad, denormalize_box

# Path al CSV subido (por si quieres calcular anchors o logs)
CSV_LABELS_PATH = '/mnt/data/_processed_data_labels.csv'

os.makedirs(OUTPUT_FEED_DIR, exist_ok=True)

def load_model_safe(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Modelo cargado desde: {model_path}")
        return model
    except Exception as e:
        print("Error cargando el modelo con load_model():", e)
        raise

def infer_image(model, img_path, out_path=None, img_size=IMG_SIZE[0]):
    # Leer imagen original
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Preprocesamiento: letterbox
    img_pad, scale, top, left = resize_pad(img_rgb, img_size)
    inp = img_pad.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, 0)  # batch=1

    # Predict
    pred = model.predict(inp)
    # asumimos salida (1,4) normalizada: xmin,ymin,xmax,ymax
    if isinstance(pred, list) or isinstance(pred, tuple):
        # si por alguna razón model devuelve lista, tomar la primera
        pred = pred[0]
    pred = np.asarray(pred).reshape(-1)
    # Convertir bbox normalizada respecto a padded image (size x size)
    x1_p = int(pred[0] * img_size)
    y1_p = int(pred[1] * img_size)
    x2_p = int(pred[2] * img_size)
    y2_p = int(pred[3] * img_size)

    # Mapear de padded imagen a original: quitar padding y dividir por scale
    x1_orig = int(max(0, (x1_p - left) / scale))
    y1_orig = int(max(0, (y1_p - top) / scale))
    x2_orig = int(min(orig_w, (x2_p - left) / scale))
    y2_orig = int(min(orig_h, (y2_p - top) / scale))

    # Dibujar en la copia BGR original
    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(out_bgr, (x1_orig, y1_orig), (x2_orig, y2_orig), (0,255,0), 2)
    cv2.putText(out_bgr, "placa", (x1_orig, max(0, y1_orig-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Guardar resultado
    if out_path is None:
        base = os.path.basename(img_path)
        out_path = os.path.join(OUTPUT_FEED_DIR, f"det_{base}")
    cv2.imwrite(out_path, out_bgr)
    print(f"Resultado guardado en: {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Inferir una imagen con detector de placas")
    parser.add_argument("--image", required=True, help="Ruta a la imagen de entrada")
    parser.add_argument("--model", default=MODEL_PATH, help="Ruta al modelo Keras")
    parser.add_argument("--out", default=None, help="Ruta de salida opcional")
    args = parser.parse_args()

    model = load_model_safe(args.model)
    out = infer_image(model, args.image, out_path=args.out)
    print("Hecho.")

if __name__ == "__main__":
    main()
