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
from src.utils.mpd_utils import resize_pad, nms_numpy
from src.models.efficient_detector_multi_placa import NUM_ANCHORS, NUM_CLASSES
import numpy as np

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
        arr = np.asarray(pred)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]

        out_frame = frame.copy()

        # Si la salida es una única caja de 4 valores, mantener compatibilidad
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 4):
            box = np.asarray(arr).reshape(-1)
            x1_p = int(box[0] * img_size)
            y1_p = int(box[1] * img_size)
            x2_p = int(box[2] * img_size)
            y2_p = int(box[3] * img_size)

            h, w = frame.shape[:2]
            x1_orig = int(max(0, (x1_p - left) / scale))
            y1_orig = int(max(0, (y1_p - top) / scale))
            x2_orig = int(min(w, (x2_p - left) / scale))
            y2_orig = int(min(h, (y2_p - top) / scale))

            cv2.rectangle(out_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (0,255,0), 2)
            cv2.putText(out_frame, "placa", (x1_orig, max(0, y1_orig-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
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

                        x1_p = int(xmin * img_size)
                        y1_p = int(ymin * img_size)
                        x2_p = int(xmax * img_size)
                        y2_p = int(ymax * img_size)

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
                    cv2.putText(out_frame, f"placa {s:.2f}", (b[0], max(0, b[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

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
