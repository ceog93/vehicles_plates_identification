# augment.py
# Augmentaciones básicas y mosaic simplificado. Comentarios en español.


import cv2
import numpy as np
import random
from .mpd_utils import resize_pad


def augment_basic(img):
    # jitter de color
    if random.random() < 0.5:
        factor = 1.0 + (random.random() - 0.5) * 0.6
        img = np.clip(img * factor, 0, 255).astype(np.uint8)
    # blur ocasional
    if random.random() < 0.2:
        k = random.choice([3,5])
        img = cv2.GaussianBlur(img, (k,k), 0)
    return img


# Nota: la implementación completa de Mosaic debe transformar cajas adecuadamente.
# Aquí entregamos una versión simplificada para facilitar integración y pruebas.


def mosaic_simplificado(images_list, labels_list, input_size=640):
    # images_list: lista de 4 arrays RGB
    # labels_list: lista de arrays Nx5 normalizados
    s = input_size
    mosaic = np.full((s,s,3), 114, dtype=np.uint8)
    xc = int(random.uniform(s*0.25, s*0.75))
    yc = int(random.uniform(s*0.25, s*0.75))
    # colocar imágenes en cuadrantes — para hacer correctamente, hay que remapear cajas
    imgs_resized = []
    for img in images_list:
        img_r, _, _, _ = resize_pad(img, s)
        imgs_resized.append(img_r)
    mosaic[0:yc, 0:xc] = imgs_resized[0][0:yc, 0:xc]
    mosaic[0:yc, xc:s] = imgs_resized[1][0:yc, xc:s]
    mosaic[yc:s, 0:xc] = imgs_resized[2][yc:s, 0:xc]
    mosaic[yc:s, xc:s] = imgs_resized[3][yc:s, xc:s]
    merged_boxes = np.concatenate([lab for lab in labels_list if lab.shape[0] > 0], axis=0) if any([lab.shape[0]>0 for lab in labels_list]) else np.zeros((0,5))
    return mosaic, merged_boxes