# src/config.py

'''
Configuraciones globales
'''

# src/config.py

import os
# === CONFIGURACIÓN GLOBAL ===
# Rutas
# Directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Carpeta de datos
DATA_DIR = os.path.join(BASE_DIR, "data")
# Carpeta donde se guardará el modelo entrenado
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "models", "detector_model.h5")

# Parámetros de entrenamiento
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

# Umbral de predicción
THRESHOLD = 0.7

# Crear carpetas si no existen
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
