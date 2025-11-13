# src/train.py
# Entrenamiento desde cero de un detector de placas vehiculares
# CNN + regresión de bounding boxes (dataset Roboflow CSV)

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

# ============================
# CONFIGURACIÓN
# ============================
IMG_DIR = "dataset/images"           # Carpeta de imágenes
CSV_PATH = "dataset/_annotations.csv"  # Archivo CSV de anotaciones
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4

# ============================
# FUNCIÓN: CARGAR DATASET
# ============================

def load_dataset(img_dir, csv_path):
    df = pd.read_csv(csv_path)
    X, y = [], []

    # Si el CSV tiene columnas tipo: filename, xmin, ymin, xmax, ymax, class
    # ajusta según tus encabezados exactos
    for filename, group in df.groupby("filename"):
        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            continue

        # Leer y redimensionar imagen
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, IMG_SIZE)
        img_resized = img_resized.astype(np.float32) / 255.0

        # Tomamos la primera caja (una por imagen, simplificación)
        row = group.iloc[0]
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

        # Normalizar coordenadas al rango [0,1]
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        box_width = (xmax - xmin) / w
        box_height = (ymax - ymin) / h

        X.append(img_resized)
        y.append([x_center, y_center, box_width, box_height])

    return np.array(X), np.array(y)

# ============================
# FUNCIÓN: CONSTRUIR MODELO
# ============================

def build_detector():
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)  # x, y, w, h normalizados
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse', metrics=['mae'])
    return model

# ============================
# MAIN
# ============================

if __name__ == "__main__":
    print("Cargando dataset desde CSV...")
    X, y = load_dataset(IMG_DIR, CSV_PATH)
    print(f"Total de imágenes cargadas: {len(X)}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"Entrenamiento: {len(X_train)}, Validación: {len(X_val)}")

    print("Compilando modelo...")
    model = build_detector()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Guardar modelo
    os.makedirs("models", exist_ok=True)
    model.save("models/detector.h5")
    print("✅ Modelo guardado en models/detector.h5")

    # Graficar pérdida
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title("Pérdida del detector")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()
