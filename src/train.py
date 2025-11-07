# src/train.py
'''
entrenamiendo del modelo
'''

# src/train.py

import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from config import DATA_DIR, MODEL_PATH, IMG_SIZE, BATCH_SIZE, EPOCHS
from utils import create_data_generators
import os

def train_model():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    train_gen, val_gen = create_data_generators(train_dir, val_dir, IMG_SIZE, BATCH_SIZE)

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False  # Transfer learning

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=preds)

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    print("[INFO] Entrenando modelo...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    model.save(MODEL_PATH)
    print(f"[INFO] Modelo guardado en {MODEL_PATH}")

    return model, history

if __name__ == "__main__":
    train_model()
