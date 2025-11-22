# src/inference/predict_video.py
"""
Inferencia sobre un video:
- Lee frames, aplica letterbox (resize_pad)
- Predice bbox por frame (modelo de regresión 4 valores)
- Mapea coords a frame original, dibuja y guarda output video
- Guarda video resultante en OUTPUT_FEED_DIR
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR
from src.utils.mpd_utils import resize_pad

CSV_LABELS_PATH = '/mnt/data/_processed_data_labels.csv'
os.makedirs(OUTPUT_FEED_DIR, exist_ok=True)

def load_model_safe(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Modelo cargado desde: {model_path}")
        return model
    except Exception as e:
        print("Error cargando el modelo:", e)
        raise

def process_video(model, video_path, out_video_path=None, img_size=IMG_SIZE[0], display=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if out_video_path is None:
        base = os.path.basename(video_path)
        out_video_path = os.path.join(OUTPUT_FEED_DIR, f"det_{base}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Procesando frames")

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

        # Convertir pred a coordenadas sobre padded img
        x1_p = int(pred[0] * img_size)
        y1_p = int(pred[1] * img_size)
        x2_p = int(pred[2] * img_size)
        y2_p = int(pred[3] * img_size)

        # Mapear de padded -> original
        x1_orig = int(max(0, (x1_p - left) / scale))
        y1_orig = int(max(0, (y1_p - top) / scale))
        x2_orig = int(min(width, (x2_p - left) / scale))
        y2_orig = int(min(height, (y2_p - top) / scale))

        # Dibujar
        out_frame = frame.copy()
        cv2.rectangle(out_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (0,255,0), 2)
        cv2.putText(out_frame, "placa", (x1_orig, max(0, y1_orig-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

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

    print(f"Video de salida guardado en: {out_video_path}")
    return out_video_path

def main():
    parser = argparse.ArgumentParser(description="Inferir un video con detector de placas")
    parser.add_argument("--video", required=True, help="Ruta a archivo de video")
    parser.add_argument("--model", default=MODEL_PATH, help="Ruta al modelo Keras")
    parser.add_argument("--out", default=None, help="Ruta de salida mp4 opcional")
    parser.add_argument("--display", action="store_true", help="Mostrar video en pantalla")
    args = parser.parse_args()

    model = load_model_safe(args.model)
    out = process_video(model, args.video, out_video_path=args.out, display=args.display)
    print("Hecho.")

if __name__ == "__main__":
    main()
