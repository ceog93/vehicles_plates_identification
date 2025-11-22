# archivo: infer_video_mpd.py
# Inferencia en video/cámara usando MPD-Net.
# Inferencia se hace a INFER_SIZE (416) para balance precisión/latencia.

import cv2
import numpy as np
from math import exp
from src.models.mpd_net import construir_mpd_net
from src.utils.mpd_utils import resize_pad, nms_numpy # type: ignore
import pandas as pd
from sklearn.cluster import KMeans

# CONFIG
INPUT_SIZE_TRAIN = 640
INFER_SIZE = 416
CSV_LABELS = '/mnt/data/_processed_data_labels.csv'
MODEL_PATH = './checkpoints/mpd_model_inference.h5'
VIDEO_SOURCE = 0  # 0 = cámara; o ruta a archivo video.mp4

# recalcular anchors y escalarlos a tamaño de inferencia
def calcular_anclas_flat(csv_path, input_size=640, n_anchors=9):
    df = pd.read_csv(csv_path)
    wh = df[['w','h']].values * input_size
    kmeans = KMeans(n_clusters=n_anchors, random_state=0).fit(wh)
    centers = kmeans.cluster_centers_
    centers = centers[np.argsort(centers[:,0])]
    anchors = [(float(c[0]), float(c[1])) for c in centers]
    return anchors

anchors_flat = calcular_anclas_flat(CSV_LABELS, input_size=INPUT_SIZE_TRAIN, n_anchors=9)
anchors = [anchors_flat[0:3], anchors_flat[3:6], anchors_flat[6:9]]
ratio = INFER_SIZE / INPUT_SIZE_TRAIN
anchors_inf = [[(w * ratio, h * ratio) for (w, h) in group] for group in anchors]

# construir modelo (acepta input variable) y cargar pesos
model = construir_mpd_net(input_size_train=INPUT_SIZE_TRAIN, anchors_por_escala=3)
model.load_weights(MODEL_PATH)
print('Modelo cargado en', MODEL_PATH)

# abrir fuente de video
cap = cv2.VideoCapture(VIDEO_SOURCE)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # convertir a formato RGB y resize+pad a INFER_SIZE
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized, scale, top, left = resize_pad(img_rgb, INFER_SIZE)
    input_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, 0)  # batch 1

    # obtener predicciones (lista de 3 tensores)
    preds = model(input_tensor, training=False)

    boxes_all = []
    scores_all = []
    # decodificar cada escala
    for si, p in enumerate(preds):
        p = p[0].numpy()  # shape: S,S,A*5
        S = p.shape[0]
        A = p.shape[2] // 5
        p = p.reshape((S, S, A, 5))
        for i in range(S):
            for j in range(S):
                for a in range(A):
                    tx, ty, tw, th, tobj = p[i, j, a]
                    prob = 1.0 / (1.0 + np.exp(-tobj))
                    if prob < 0.25:
                        continue
                    # center offsets: sigmoid-like for stability
                    cx = (j + 1.0 / (1.0 + np.exp(-tx))) * (INFER_SIZE / S)
                    cy = (i + 1.0 / (1.0 + np.exp(-ty))) * (INFER_SIZE / S)
                    aw, ah = anchors_inf[si][a]
                    w_px = aw * np.exp(tw)
                    h_px = ah * np.exp(th)
                    x1 = cx - w_px / 2.0
                    y1 = cy - h_px / 2.0
                    x2 = cx + w_px / 2.0
                    y2 = cy + h_px / 2.0
                    boxes_all.append([x1, y1, x2, y2])
                    scores_all.append(prob)

    if len(boxes_all) == 0:
        disp = cv2.cvtColor(img_resized.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow('MPD Detect', disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes_all = np.array(boxes_all)
    scores_all = np.array(scores_all)
    keep = nms_numpy(boxes_all, scores_all, iou_thresh=0.45, score_thresh=0.25)
    disp = cv2.cvtColor(img_resized.astype(np.uint8), cv2.COLOR_RGB2BGR)
    for idx in keep:
        x1, y1, x2, y2 = map(int, boxes_all[idx])
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(disp, f'placa {scores_all[idx]:.2f}', (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('MPD Detect', disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
