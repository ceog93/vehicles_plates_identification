# archivo: train_mpd.py
# Script de entrenamiento MPD-Net (lista para copiar/pegar)
# Usa el CSV que subiste: /mnt/data/_processed_data_labels.csv para calcular anchors.

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.models.mpd_net import construir_mpd_net
from src.dataset.mpd_dataset import MPDDataset
from src.losses.mpd_loss import PerdidaMPD

# CONFIGURACIÓN
INPUT_SIZE_TRAIN = 640
BATCH_SIZE = 8
EPOCHS = 80
CSV_LABELS = '/mnt/data/_processed_data_labels.csv'  # ruta del CSV subido
IMAGES_DIR = './data/images'
LABELS_DIR = './data/labels'
CKPT_DIR = './checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

# 1) calcular anchors con k-means sobre w,h del CSV (normalizados)
def calcular_anclas(csv_path, input_size=640, n_anchors=9):
    df = pd.read_csv(csv_path)
    if not all(col in df.columns for col in ['w','h']):
        raise ValueError('CSV debe contener columnas w,h normalizadas')
    wh = df[['w','h']].values * input_size
    kmeans = KMeans(n_clusters=n_anchors, random_state=0).fit(wh)
    centers = kmeans.cluster_centers_
    centers = centers[np.argsort(centers[:,0])]
    anchors = [(float(round(c[0])), float(round(c[1]))) for c in centers]
    return anchors

anchors_flat = calcular_anclas(CSV_LABELS, input_size=INPUT_SIZE_TRAIN, n_anchors=9)
anchors = [anchors_flat[0:3], anchors_flat[3:6], anchors_flat[6:9]]
print('Anchors (pixeles):', anchors)

# 2) construir modelo y pérdida en scope (por si usas estrategia distribuida)
strategy = tf.distribute.get_strategy()
with strategy.scope():
    model = construir_mpd_net(input_size_train=INPUT_SIZE_TRAIN, anchors_por_escala=3)
    loss_fn = PerdidaMPD(anchors=anchors)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# 3) dataset
dataset = MPDDataset(IMAGES_DIR, LABELS_DIR, input_size=INPUT_SIZE_TRAIN, batch_size=BATCH_SIZE, anchors=anchors, augment=True)

# helper de entrenamiento
@tf.function
def train_step(images, y_trues):
    with tf.GradientTape() as tape:
        preds = model(images, training=True)
        loss = loss_fn(y_trues, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# 4) bucle de entrenamiento
for epoch in range(EPOCHS):
    total_loss = 0.0
    steps = 0
    for imgs, y_trues in dataset:
        loss_v = train_step(imgs, y_trues)
        total_loss += float(loss_v)
        steps += 1
        if steps % 10 == 0:
            print(f'Epoch {epoch+1} step {steps} loss {loss_v:.4f}')
    avg = total_loss / max(1, steps)
    print(f'Epoch {epoch+1} avg_loss {avg:.4f}')
    model.save_weights(os.path.join(CKPT_DIR, f'ep_{epoch+1}.ckpt'))

# guardar modelo para inferencia (formato h5)
model.save(os.path.join(CKPT_DIR, 'mpd_model_inference.h5'))
print('Entrenamiento finalizado. Modelo guardado en', os.path.join(CKPT_DIR, 'mpd_model_inference.h5'))
