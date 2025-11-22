# mpd_dataset.py
# Data loader que produce batches (imgs, [y_fina,y_media,y_gruesa])
# Labels formato YOLO: class cx cy w h (normalizados)


import os
import numpy as np
import cv2
import random
from ..utils.mpd_utils import resize_pad # type: ignore


class MPDDataset:
    def __init__(self, images_dir, labels_dir, input_size=640, batch_size=8, anchors=None, augment=True, shuffle=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.input_size = input_size
        self.batch_size = batch_size