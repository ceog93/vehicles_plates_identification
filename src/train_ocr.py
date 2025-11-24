
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.models.ocr_model import build_ocr_model
import matplotlib.pyplot as plt

# CONFIGURACIÓN
DATA_DIR = "data_ocr"
IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 20
MODEL_SAVE_PATH = "saved_models/ocr_model.h5"

def train_ocr():
    # 1. Preparar Generador de Datos (Carga imágenes de las carpetas)
    # Usamos validation_split para separar automáticamente 20% para validar
    datagen = ImageDataGenerator(
        rescale=1./255,          # Normalizar pixeles de 0-255 a 0-1
        validation_split=0.2,    # 20% para validación
        rotation_range=10,       # Data Augmentation leve
        zoom_range=0.1
    )

    print("Cargando datos de entrenamiento...")
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale', # Importante: 1 canal
        class_mode='sparse',    # Etiquetas numéricas
        subset='training',
        shuffle=True
    )

    print("Cargando datos de validación...")
    val_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='sparse',
        subset='validation'
    )

    # Imprimir el mapeo de clases (A=10, B=11...) para saber qué es qué
    print("Mapeo de Clases:", train_generator.class_indices)

    # 2. Construir Modelo
    model = build_ocr_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=36)
    model.summary()

    # 3. Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss')
    ]

    # 4. Entrenar
    print("Iniciando entrenamiento OCR...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 5. Graficar resultados
    plt.plot(history.history['accuracy'], label='Accuracy Train')
    plt.plot(history.history['val_accuracy'], label='Accuracy Val')
    plt.title('Precisión del OCR')
    plt.legend()
    plt.show()

    print(f"Modelo OCR guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_ocr()