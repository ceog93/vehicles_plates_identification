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

from src.config import MODEL_PATH, IMG_SIZE, OUTPUT_FEED_DIR, TRAIN_DATA_DIR
from src.utils.mpd_utils import resize_pad, denormalize_box, nms_numpy
from src.models.efficient_detector_multi_placa import GRID_SIZE, NUM_ANCHORS, NUM_CLASSES

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

    # Si la salida es una única caja (por ejemplo modelos simples que devuelven [xmin,ymin,xmax,ymax])
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Caso A: salida directa de 4 valores por imagen
    if isinstance(pred, np.ndarray) and pred.ndim == 2 and pred.shape[1] == 4:
        box = np.asarray(pred).reshape(-1)
        x1_p = int(box[0] * img_size)
        y1_p = int(box[1] * img_size)
        x2_p = int(box[2] * img_size)
        y2_p = int(box[3] * img_size)

        # Mapear de padded imagen a original: quitar padding y dividir por scale
        x1_orig = int(max(0, (x1_p - left) / scale))
        y1_orig = int(max(0, (y1_p - top) / scale))
        x2_orig = int(min(orig_w, (x2_p - left) / scale))
        y2_orig = int(min(orig_h, (y2_p - top) / scale))

        cv2.rectangle(out_bgr, (x1_orig, y1_orig), (x2_orig, y2_orig), (0,255,0), 2)
        cv2.putText(out_bgr, "placa", (x1_orig, max(0, y1_orig-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    else:
        # Caso B: salida tipo YOLO (grid_h, grid_w, anchors*(5+num_classes))
        arr = np.asarray(pred)
        # Asegurar que tenemos (G,G,channels)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]

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
                    w = w_norm
                    h = h_norm

                    xmin = cx - w / 2
                    ymin = cy - h / 2
                    xmax = cx + w / 2
                    ymax = cy + h / 2

                    x1_p = int(xmin * img_size)
                    y1_p = int(ymin * img_size)
                    x2_p = int(xmax * img_size)
                    y2_p = int(ymax * img_size)

                    x1_orig = int(max(0, (x1_p - left) / scale))
                    y1_orig = int(max(0, (y1_p - top) / scale))
                    x2_orig = int(min(orig_w, (x2_p - left) / scale))
                    y2_orig = int(min(orig_h, (y2_p - top) / scale))

                    boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
                    scores.append(conf)

        # Aplicar NMS
        if len(boxes) > 0:
            selected = nms_numpy(boxes, scores, iou_thresh=0.45, score_thresh=0.2)
            for idx in selected:
                b = boxes[int(idx)]
                s = scores[int(idx)]
                cv2.rectangle(out_bgr, (b[0], b[1]), (b[2], b[3]), (0,255,0), 2)
                cv2.putText(out_bgr, f"placa {s:.2f}", (b[0], max(0, b[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

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
