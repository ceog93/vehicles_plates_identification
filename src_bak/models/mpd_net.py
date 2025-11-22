# archivo: src/dataset/mpd_dataset.py
# Loader y builder de targets para MPD-Net
# - Lee etiquetas estilo YOLO: class cx cy w h (normalizados)
# - Aplica augmentaciones básicas
# - Construye y_true por escala con offsets y codificación log-space para w,h

import os
import random
import numpy as np
import tensorflow as tf
import cv2
from ..utils.mpd_utils import resize_pad # type: ignore

class MPDDataset:
    """
    Iterador simple para entrenar MPD-Net.
    - images_dir: carpeta con imágenes
    - labels_dir: carpeta con .txt (por imagen) con líneas: class cx cy w h (normalizado)
    - input_size: tamaño de entrenamiento (ej. 640)
    - batch_size: tamaño de batch
    - anchors: lista por escala [[(w,h),...], ...] en pixeles (relativos a input_size)
    """
    def __init__(self, images_dir, labels_dir, input_size=640, batch_size=8, anchors=None, augment=True, shuffle=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.anchors = anchors
        self.augment = augment
        self.shuffle = shuffle
        self.files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])

    def leer_labels(self, base_name):
        path = os.path.join(self.labels_dir, base_name + '.txt')
        boxes = []
        if not os.path.exists(path):
            return np.zeros((0,5), dtype=np.float32)
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # asumimos clase 0 = placa
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append([cx, cy, w, h, 1.0])
        return np.array(boxes, dtype=np.float32)

    def _augment_basic(self, img, boxes):
        # jitter básico de color y blur opcional
        if random.random() < 0.5:
            factor = 1.0 + (random.random() - 0.5) * 0.4
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        if random.random() < 0.15:
            k = random.choice([3,5])
            img = cv2.GaussianBlur(img, (k,k), 0)
        return img, boxes

    def construir_y_true(self, batch_boxes):
        """
        Convierte un batch (lista de arrays Nx5) a tensores por escala:
        salida: [y_fina, y_media, y_gruesa] con shapes (B, S, S, A, 5)
        donde 5 = tx,ty,tw,th,obj
        """
        B = len(batch_boxes)
        # correspondencia de strides para input 640: 8,16,32 -> grids 80,40,20
        strides = [8, 16, 32]
        y_list = []
        for si, anchors in enumerate(self.anchors):
            stride = strides[si]
            grid = self.input_size // stride
            A = len(anchors)
            y = np.zeros((B, grid, grid, A, 5), dtype=np.float32)
            for b_idx in range(B):
                boxes = batch_boxes[b_idx]
                if boxes.shape[0] == 0:
                    continue
                for box in boxes:
                    cx, cy, w, h, obj = box
                    gx = cx * grid
                    gy = cy * grid
                    gi = int(min(grid - 1, max(0, int(np.floor(gx)))))
                    gj = int(min(grid - 1, max(0, int(np.floor(gy)))))
                    gw = w * self.input_size
                    gh = h * self.input_size
                    # seleccionar anchor que mejor se ajuste por IoU (w,h)
                    best = 0
                    best_iou = -1
                    for a_idx, (aw, ah) in enumerate(anchors):
                        inter_w = min(aw, gw)
                        inter_h = min(ah, gh)
                        inter = inter_w * inter_h
                        union = aw * ah + gw * gh - inter
                        iou = inter / (union + 1e-6)
                        if iou > best_iou:
                            best_iou = iou
                            best = a_idx
                    # asignar targets: offsets dentro de la celda y codificación log-scale w,h
                    y[b_idx, gj, gi, best, 0] = gx - gi
                    y[b_idx, gj, gi, best, 1] = gy - gj
                    aw, ah = anchors[best]
                    y[b_idx, gj, gi, best, 2] = np.log(gw / (aw + 1e-6) + 1e-6)
                    y[b_idx, gj, gi, best, 3] = np.log(gh / (ah + 1e-6) + 1e-6)
                    y[b_idx, gj, gi, best, 4] = 1.0
            y_list.append(y)
        # convertir a tensores tf.float32 para consumo por pérdida
        return [tf.convert_to_tensor(arr, dtype=tf.float32) for arr in y_list]

    def __iter__(self):
        files = list(self.files)
        if self.shuffle:
            random.shuffle(files)
        for i in range(0, len(files), self.batch_size):
            batch_files = files[i:i + self.batch_size]
            imgs = []
            batch_boxes = []
            for fname in batch_files:
                img = cv2.imread(os.path.join(self.images_dir, fname))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                base = os.path.splitext(fname)[0]
                boxes = self.leer_labels(base)
                img_resized, _, _, _ = resize_pad(img, self.input_size)
                if self.augment:
                    img_resized, boxes = self._augment_basic(img_resized, boxes)
                imgs.append(img_resized.astype(np.float32) / 255.0)
                batch_boxes.append(boxes)
            y_trues = self.construir_y_true(batch_boxes)
            yield np.stack(imgs, axis=0), y_trues
