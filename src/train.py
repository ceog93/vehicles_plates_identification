# src/train.py
# Entrenamiento desde cero de un detector de placas vehiculares
# CNN + regresión de bounding boxes (dataset Roboflow CSV)

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

# ============================
# CONFIGURACIÓN
# ============================
IMG_DIR = "dataset/images"           # Carpeta de imágenes
CSV_PATH = "dataset/_annotations.csv"  # Archivo CSV de anotaciones
IMG_SIZE = (224, 224) # Tamaño de entrada de la imagen
BATCH_SIZE = 16 # Tamaño de batch (Lote de datos a entrenar por ciclo) (ajustar según memoria GPU en GB)
EPOCHS = 100 # Número de epochs de entrenamiento (ajustar según dataset) (ciclos completo de entrenamiento)
LR = 1e-4 # Tasa de aprendizaje 1e-4=1×10−4=0.0001

# ============================
# FUNCIÓN: CARGAR DATASET
# ============================

def load_dataset(img_dir, csv_path):
    '''
    Cargar dataset desde CSV y preprocesar imágenes y cajas delimitadoras
    Retorna:
        X: array numpy de imágenes preprocesadas
        y: array numpy de cajas delimitadoras normalizadas
    '''
    df = pd.read_csv(csv_path) # dataframe con anotaciones
    X, y = [], [] # imágenes y cajas

    # Iterar por cada imagen única y sus anotaciones

    for filename, group in df.groupby("filename"):
        img_path = os.path.join(img_dir, filename) # ruta completa de la imagen

        # Verificar si la imagen existe
        if not os.path.exists(img_path):
            continue

        # Leer y redimensionar imagen
        img = cv2.imread(img_path) # leer imagen

        # Verificar si la imagen se cargó correctamente
        if img is None:
            continue

        # Obtener dimensiones originales de la imagen
        h, w = img.shape[:2] # dimensiones originales

        # Preprocesamiento de la imagen
        img_resized = cv2.resize(img, IMG_SIZE) # redimensionar imagen 
        img_resized = img_resized.astype(np.float32) / 255.0 # normalizar [0,1]

        # Tomamos la primera caja (una por imagen, simplificación)
        row = group.iloc[0] # primera anotación
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"] # coordenadas absolutas de la caja

        # Normalizar datos de la caja
        # preprocesamiento de datos con Normalización Mín-Máx

        x_center = ((xmin + xmax) / 2) / w # centro de la caja en el eje x normalizado
        y_center = ((ymin + ymax) / 2) / h # centro de la caja en el eje y normalizado
        box_width = (xmax - xmin) / w # ancho de la caja en el eje x normalizado
        box_height = (ymax - ymin) / h # alto de la caja en el eje y normalizado

        X.append(img_resized) # agregar imagen preprocesada
        y.append([x_center, y_center, box_width, box_height]) # agregar caja normalizada

    return np.array(X), np.array(y) # convertir a arrays numpy y retornar 

# ============================
# FUNCIÓN: CONSTRUIR MODELO
# ============================

def build_detector():
    '''
    Construir modelo CNN para regresión de bounding boxes
    Retorna:
        model: modelo compilado de Keras
    '''
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)) # entrada de imagen 
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs) # primera capa convolucional
    x = layers.MaxPooling2D(2)(x) # primera capa de pooling
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x) # segunda capa convolucional
    x = layers.MaxPooling2D(2)(x) # segunda capa de pooling
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x) # tercera capa convolucional
    x = layers.MaxPooling2D(2)(x) # tercera capa de pooling
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x) # cuarta capa convolucional
    x = layers.GlobalAveragePooling2D()(x) # capa de pooling global
    x = layers.Dense(256, activation='relu')(x) # capa densa
    x = layers.Dropout(0.3)(x) # capa de dropout para regularización
    outputs = layers.Dense(4, activation='sigmoid')(x)  # x, y, w, h normalizados
    model = models.Model(inputs, outputs) # construir modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse', metrics=['mae']) # compilar modelo con Adam y MSE
    return model # retornar modelo compilado

# ============================
# FUNCIÓN: GRAFICAR HISTORIAL DE ENTRENAMIENTO
# ============================

def plot_training_history(history):
    '''
    Graficar métricas de entrenamiento y validación
    history: objeto History retornado por model.fit()
    '''
    # Graficar pérdida
    # La métrica de pérdida utilizada es el Error Cuadrático Medio (MSE).
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title("Pérdida del detector")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig("models/logs/plots/training_loss.png") # guardar figura
    plt.show()

# ============================
# FUNCIÓN: ENTRENAR MODELO
# ============================

def train_model():
    '''
    Función principal para entrenar el modelo de detección de placas vehiculares
    IMG_DIR: carpeta de imágenes
    CSV_PATH: archivo CSV de anotaciones
    X: array numpy de imágenes preprocesadas
    y: array numpy de cajas delimitadoras normalizadas
    test_size: proporción del dataset para validación
    EPOCHS: número de epochs de entrenamiento
    BATCH_SIZE: tamaño de batch para entrenamiento

    '''
    print("=========================================")
    print("      ENTRENAMIENTO DEL MODELO")
    print("=========================================")
    print(f"        1/6 Iniciando entrenamiento del modelo de detección de placas vehiculares...")
    print(f"        2/6 Cargando dataset desde CSV...")

    X, y = load_dataset(IMG_DIR, CSV_PATH)

    print(f"            2.1 Total de imágenes cargadas: {len(X)}")
    print(f"        3/6 Dividiendo dataset en entrenamiento (75%) y validación (25%)...")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    print(f"            3.1 Datos de Entrenamiento: {len(X_train)}")
    print(f"            3.2 Datos de Validación: {len(X_val)}")

    print(f"        4/6 Compilando modelo CNN para regresión de bounding boxes...\n")

    model = build_detector()

    print(f"        5/6 Iniciando proceso de entrenamiento...\n")

    # Crea el callback para registrar métricas del entrenamiento en CSV
    os.makedirs("models", exist_ok=True) # crear carpeta si no existe
    os.makedirs("models/logs", exist_ok=True) # crear carpeta si no existe
    csv_logger = CSVLogger('logs/training_log.csv', append=True) # registrar métricas en CSV

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[csv_logger]
    )

    print(f"\n          5.1 ✅ Entrenamiento completado.")

    # Guardar modelo
    model.save("models/detector.h5")
    print(f"\n        6/6 ✅ Modelo guardado en models/detector.h5")

    # Graficar historial de entrenamiento
    print(f"\n        Graficando historial de entrenamiento...")
    plot_training_history(history)

    input(" Presione enter para continuar...")

    return history, model

