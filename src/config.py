# src/config.py

'''
Configuraciones globales
'''

# src/config.py

import os
# IMPORTA datetime PARA MARCAR LA FECHA Y HORA EN LOS MODELOS
from datetime import datetime
# === CONFIGURACIÓN GLOBAL ===
# Rutas
# Directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ============================= DATOS =============================
# Carpeta de datos
DATA_DIR = os.path.join(BASE_DIR, "01_data")
os.makedirs(DATA_DIR, exist_ok=True) # Asegura que la carpeta de datos exista
RAW_DATASET_DIR = os.path.join(DATA_DIR, "raw_dataset")
os.makedirs(RAW_DATASET_DIR, exist_ok=True) # Asegura que la carpeta de dataset crudo exista
PROCESED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
os.makedirs(PROCESED_DATA_DIR, exist_ok=True) # Asegura que la carpeta de datos procesados exista
TEST_DATA_DIR = os.path.join(PROCESED_DATA_DIR, "test")
os.makedirs(TEST_DATA_DIR, exist_ok=True) # Asegura que la carpeta de datos de prueba exista
TRAIN_DATA_DIR = os.path.join(PROCESED_DATA_DIR, "train")
os.makedirs(TRAIN_DATA_DIR, exist_ok=True) # Asegura que la carpeta de datos de entrenamiento exista
VALIDATION_DATA_DIR = os.path.join(PROCESED_DATA_DIR, "validation")
os.makedirs(VALIDATION_DATA_DIR, exist_ok=True) # Asegura que la carpeta de datos de validación exista
LABELS_DIR = os.path.join(DATA_DIR, "labels")
os.makedirs(LABELS_DIR, exist_ok=True) # Asegura que la carpeta de etiquetas exista
RAW_DATA_LABELS_CSV = os.path.join(LABELS_DIR, "_raw_data_labels.csv") # Ruta del CSV con etiquetas del dataset crudo
PROCESSED_DATA_LABELS_CSV = os.path.join(LABELS_DIR, "_processed_data_labels.csv") # Ruta del CSV con etiquetas del dataset procesado

## ============================= MODELOS =============================
# --- Configuración para GUARDAR el modelo actual ---
# Carpeta RAIZ donde se guardan todos los modelos entrenados
ROOT_MODEL_DIR = os.path.join(BASE_DIR, "02_models")
# hora y fecha actual YYYYMMDD_HHMMSS para versionado
current_time = datetime.now().strftime("%Y%m%d_%H%MM%S")
model_name = (f"model_{current_time}")
# Carpeta específica para el modelo actual: 02_models/<FECHA_ACTUAL>|
CURRENT_MODEL_DIR = os.path.join(ROOT_MODEL_DIR, model_name)
# Ruta completa para guardar el modelo: 02_models/<FECHA_ACTUAL>/detector
MODEL_PATH = os.path.join(CURRENT_MODEL_DIR, "detector_model.keras")
#os.makedirs(CURRENT_MODEL_DIR, exist_ok=True) # Asegura que la carpeta del modelo actual exista
LOGS_DIR = os.path.join(CURRENT_MODEL_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True) # Asegura que la carpeta de logs exista
PLOTS_DIR = os.path.join(LOGS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True) # Asegura que la carpeta de plots exista
TRAINING_LOG_CSV = os.path.join(LOGS_DIR, "training_log.csv")
TRAINING_LOSS_PLOT_PATH = os.path.join(PLOTS_DIR, "training_loss.png")

# --- Configuración para CARGAR el último modelo entrenado ---
LATEST_MODEL_PATH = None
if os.path.exists(ROOT_MODEL_DIR):
    # Buscamos todas las subcarpetas (versiones) dentro de 02_models
    subdirs = [d for d in os.listdir(ROOT_MODEL_DIR) 
               if os.path.isdir(os.path.join(ROOT_MODEL_DIR, d))]
    if subdirs:
        # La subcarpeta más reciente (el string más grande por orden alfabético)
        latest_subdir = max(subdirs) 
        # Ruta completa del detector en la versión más reciente
        LATEST_MODEL_PATH = os.path.join(ROOT_MODEL_DIR, latest_subdir, "detector")

# ============================= PRODUCCIÓN =============================
# Carpeta para producción (videos para inferencia)
PRODUCTION_DIR = os.path.join(BASE_DIR, "03_production")
os.makedirs(PRODUCTION_DIR, exist_ok=True) # Asegura que la carpeta de producción exista
INPUT_FEED_DIR = os.path.join(PRODUCTION_DIR, "input_feed")
os.makedirs(INPUT_FEED_DIR, exist_ok=True) # Asegura que la carpeta de entrada exista
OUTPUT_FEED_DIR =  os.path.join(PRODUCTION_DIR, "output_results")
os.makedirs(OUTPUT_FEED_DIR, exist_ok=True) # Asegura que la carpeta de salida exista
TEST_VIDEO_PATH = os.path.join(INPUT_FEED_DIR, "video_prueba.mp4")

# ============================= DATOS SINTÉTICOS =============================
LOGO_DIR = os.path.join(BASE_DIR, "images")
LOGO_PATH = os.path.join(LOGO_DIR, "logo_placa.png")

# ============================= ENTRENAMIENTO =============================
# Parámetros de entrenamiento
IMG_SIZE = (320, 320) # Tamaño de las imágenes de entrada para el modelo
BATCH_SIZE = 16 # Tamaño del lote para el entrenamiento (segun capacidad de memoria de la GPU o CPU)
EPOCHS = 100 # Número de épocas (iteraciones completas sobre el conjunto de datos) para el entrenamiento
LEARNING_RATE = 1e-4 # 0.001 # Tasa de aprendizaje para el optimizador
THRESHOLD = 0.7 # Umbral de predicción
# ============================= OTROS =============================
# Semilla aleatoria para reproducibilidad
RANDOM_SEED = 42
# Número de trabajadores para la carga de datos
NUM_WORKERS = 4 
# ============================= FIN DEL ARCHIVO =============================


