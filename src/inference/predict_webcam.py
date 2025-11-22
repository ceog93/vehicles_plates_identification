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
import tensorflow as tf

from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR
from src.utils.mpd_utils import resize_pad

CSV_LABELS_PATH = '/mnt/data/_processed_data_labels.csv'
os.makedirs(OUTPUT_FEED_DIR, exist_ok=True)

def load_model_safe(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        print("Error al cargar el modelo:", e)
        raise

def run_webcam(model, cam_index=0, img_size=IMG_SIZE[0], save_output=False, save_dir=OUTPUT_FEED_DIR):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir la cámara index {cam_index}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pad, scale, top, left = resize_pad(rgb, img_size)
        inp = (img_pad.astype(np.float32) / 255.0)[None, ...]

        pred = model.predict(inp)
        if isinstance(pred, list) or isinstance(pred, tuple):
            pred = pred[0]
        pred = np.asarray(pred).reshape(-1)

        # Convertir pred a coordenadas en padded image
        x1_p = int(pred[0] * img_size)
        y1_p = int(pred[1] * img_size)
        x2_p = int(pred[2] * img_size)
        y2_p = int(pred[3] * img_size)

        # Mapear a frame original
        h, w = frame.shape[:2]
        x1_orig = int(max(0, (x1_p - left) / scale))
        y1_orig = int(max(0, (y1_p - top) / scale))
        x2_orig = int(min(w, (x2_p - left) / scale))
        y2_orig = int(min(h, (y2_p - top) / scale))

        out_frame = frame.copy()
        cv2.rectangle(out_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (0,255,0), 2)
        cv2.putText(out_frame, "placa", (x1_orig, max(0, y1_orig-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Webcam - Detección", out_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if save_output and key == ord('s'):
            fname = os.path.join(save_dir, f"webcam_{int(tf.timestamp())}.jpg")
            cv2.imwrite(fname, out_frame)
            print("Frame guardado:", fname)

    cap.release()
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
