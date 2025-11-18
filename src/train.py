# src/train_corrected.py
# Entrenamiento de un detector de placas vehiculares con Transfer Learning (MobileNetV2)
# Regresión de Bounding Boxes (dataset Roboflow CSV)

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2 # Importación para Transfer Learning
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

# ============================
# CONFIGURACIÓN
# ============================
IMG_DIR = "dataset/images" # Carpeta de imágenes
CSV_PATH = "dataset/_annotations.csv" # Archivo CSV de anotaciones
IMG_SIZE = (224, 224) # Tamaño de entrada de la imagen (Estándar para MobileNetV2)
BATCH_SIZE = 32 # Aumentado para mayor eficiencia (ajustar según memoria GPU)
EPOCHS = 100 # Número de epochs máximo (Early Stopping lo detendrá antes)
LR = 1e-4 # Tasa de aprendizaje inicial

# Directorios de salida
LOGS_DIR = "models/logs"
PLOTS_DIR = os.path.join(LOGS_DIR, "plots")
MODEL_PATH = "models/detector.h5"

# Semilla para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================
# FUNCIÓN: CARGAR DATASET (Corregida)
# ============================

def load_dataset(img_dir, csv_path):
    '''
    Cargar dataset desde CSV y preprocesar imágenes y cajas delimitadoras.
    Retorna:
        X: array numpy de imágenes preprocesadas (normalizadas [0,1])
        y: array numpy de cajas delimitadoras normalizadas (xmin, ymin, xmax, ymax)
        
    NOTA: Esta función de carga de datos es simple y ASUME UNA SOLA PLACA POR IMAGEN.
    Si el dataset tiene múltiples placas, se necesitaría un enfoque diferente (e.g., YOLO).
    '''
    df = pd.read_csv(csv_path) 
    X, y = [], [] 

    print("     Iniciando carga de imágenes...")
    loaded_count = 0
    
    # Usamos una lista de nombres únicos para iterar
    unique_filenames = df['filename'].unique()

    for filename in unique_filenames:
        img_path = os.path.join(img_dir, filename) 
        group = df[df['filename'] == filename] # Filtramos las anotaciones para esta imagen

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path) 
        # Convertir de BGR a RGB para Keras/Matplotlib
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2] # Dimensiones originales

        # Preprocesamiento de la imagen (Redimensionar y normalizar [0,1])
        img_resized = cv2.resize(img, IMG_SIZE)
        img_resized = img_resized.astype(np.float32) / 255.0 

        # Tomamos la primera caja (asumiendo una sola placa, como el código original)
        row = group.iloc[0] 
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

        # Normalizar a coordenadas xmin, ymin, xmax, ymax en rango [0, 1]
        # Se prefiere esta representación para la salida de la CNN sobre (centro, w, h)
        xmin_norm = xmin / w
        ymin_norm = ymin / h
        xmax_norm = xmax / w
        ymax_norm = ymax / h

        X.append(img_resized) 
        y.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm]) # xmin, ymin, xmax, ymax normalizados
        loaded_count += 1
        
    print(f"Carga finalizada. {loaded_count} imágenes procesadas.")
    return np.array(X), np.array(y)

# ============================
# FUNCIÓN: CONSTRUIR MODELO (Mejorada con Transfer Learning)
# ============================

def build_detector():
    '''
    Construir modelo usando MobileNetV2 pre-entrenado (Transfer Learning) 
    para la regresión de bounding boxes.
    '''
    # 1. Cargar el modelo base pre-entrenado
    base_model = MobileNetV2(
        weights='imagenet', # Usar pesos de ImageNet
        include_top=False,  # Excluir la capa de clasificación final
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Congelar las capas del modelo base (no se entrenarán)
    base_model.trainable = False

    # 2. Construir el modelo completo
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False) # Usar el modelo base en modo inferencia

    # Agregar capas de regresión personalizadas
    x = layers.GlobalAveragePooling2D()(x) # Reduce la dimensionalidad espacial
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x) 
    
    # Salida de 4 coordenadas (xmin, ymin, xmax, ymax) normalizadas [0, 1]
    outputs = layers.Dense(4, activation='sigmoid', name='bounding_box_output')(x) 
    
    model = models.Model(inputs, outputs) 
    
    # Usamos Huber Loss (Smooth L1) que es más robusta para regresión de cajas que MSE puro
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR), 
        loss=tf.keras.losses.Huber(), # Mejora sobre 'mse'
        metrics=['mae', 'mse']
    ) 
    return model

# ============================
# FUNCIÓN: GRAFICAR HISTORIAL DE ENTRENAMIENTO (Corregida)
# ============================

def plot_training_history(history):
    '''
    Graficar métricas de entrenamiento y validación
    history: objeto History retornado por model.fit()
    '''
    os.makedirs(PLOTS_DIR, exist_ok=True) # Crear carpeta 'plots'

    # Graficar pérdida (Huber Loss)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida Validación')
    plt.title("Pérdida del detector (Huber Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, "training_loss.png")) # Ruta corregida
    plt.show()
    
# ============================
# FUNCIÓN: ENTRENAR MODELO
# ============================

def train_model():
    '''
    Función principal para entrenar el modelo de detección de placas vehiculares
    '''
    print("=========================================")
    print("         ENTRENAMIENTO DEL MODELO")
    print("=========================================")
    
    # 1. Configuración de directorios de salida
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) # Asegurar models/ existe

    print(f"         1/6 Iniciando entrenamiento del modelo de detección de placas vehiculares...")
    print(f"         2/6 Cargando dataset desde CSV...")

    X, y = load_dataset(IMG_DIR, CSV_PATH)

    print(f"             2.1 Dimensiones de X: {X.shape}")
    print(f"         3/6 Dividiendo dataset en entrenamiento (75%) y validación (25%)...")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=SEED) # Dividir dataset 25/75

    print(f"             3.1 Datos de Entrenamiento: {len(X_train)}")
    print(f"             3.2 Datos de Validación: {len(X_val)}")

    print(f"         4/6 Compilando modelo con MobileNetV2 (Transfer Learning)...\n")

    model = build_detector()
    # model.summary() # Descomentar para ver la arquitectura

    print(f"         5/6 Iniciando proceso de entrenamiento con Callbacks...\n")

    CSV_LOG_PATH = os.path.join(LOGS_DIR, "training_log.csv")
    
    # Callbacks para un entrenamiento robusto
    callbacks = [
        # Guarda el historial de métricas
        CSVLogger(CSV_LOG_PATH, append=True),
        # Detiene el entrenamiento si la pérdida de validación no mejora tras 10 epochs
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        # Reduce la tasa de aprendizaje si la pérdida de validación no mejora tras 5 epochs
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, # El número real de epochs será manejado por EarlyStopping
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )

    print(f"\n          5.1 ✅ Entrenamiento completado.")

    # Guardar el modelo (el que tiene los mejores pesos según EarlyStopping)
    model.save(MODEL_PATH)
    print(f"\n        6/6 ✅ Modelo guardado en {MODEL_PATH}")

    # Graficar historial de entrenamiento
    print(f"\n        Graficando historial de entrenamiento en {os.path.join(PLOTS_DIR, 'training_loss.png')}...")
    plot_training_history(history)

    input("\nPresione enter para continuar...")

    return history, model
