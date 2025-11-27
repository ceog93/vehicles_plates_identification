# src/train.py
# Entrenamiento de un detector de placas vehiculares (CNN desde cero)
# Regresión de Bounding Boxes (dataset Roboflow CSV)

import os
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
import cv2
import matplotlib.pyplot as plt
#from src.models.efficient_detector import build_efficient_detector as build_detector # Usa EfficientNetB0 como modelo base
#from src.models.mobile_net_detector import build_mobile_net_detector as build_detector_mobilenet # Alternativa: Usa MobileNetV2 como modelo base
# from src.utils.preprocess_and_augmentation import preprocess_and_save_data_modular # Se comenta ya que se usa dataset pre-procesado
from src.models.efficient_detector_multi_placa import construir_detector_multiplaca_desde_cero as build_detector 
from src.utils.batch_data_loader import ImageBatchLoader # Batch data generator for memory efficiency


# ============================
# CONFIGURACIÓN (Importadas de src.config)
# ============================
from src.config import RAW_DATASET_DIR as IMG_DIR, TRAIN_DATA_DIR, VALIDATION_DATA_DIR, TEST_DATA_DIR, CURRENT_MODEL_DIR
from src.config import RAW_DATA_LABELS_CSV as CSV_PATH, PROCESSED_DATA_LABELS_CSV
from src.config import IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, THRESHOLD
from src.config import LOGS_DIR, PLOTS_DIR, MODEL_PATH, TRAINING_LOSS_PLOT_PATH
from src.config import TRAINING_LOG_CSV

'''
# Se comenta ya que se usa generador de datos para evitar error de memoria
# ============================
# FUNCIÓN: load_processed_split (Auxiliar para cargar datos guardados)
# ============================

def load_processed_split(data_dir, df_split):
    X_split, y_split = [], [] 
    print(f"            Iniciando carga de {len(df_split)} imágenes desde {data_dir}...")
    
    for idx, row in df_split.iterrows():
        filename = row['filename']
        img_path = os.path.join(data_dir, filename) 
        
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path) 
        if img is None:
            continue
        # Convertir a RGB y Normalizar
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, IMG_SIZE)
        img_normalized = img_resized.astype(np.float32) / 255.0 

        # Recuperar y Normalizar Bounding Box
        w, h = row['width'], row['height']
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

        xmin_norm = xmin / w
        ymin_norm = ymin / h
        xmax_norm = xmax / w
        ymax_norm = ymax / h

        X_split.append(img_normalized) 
        y_split.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])

    print(f"             Carga finalizada. {len(X_split)} imágenes procesadas.")
    return np.array(X_split), np.array(y_split)
'''


# ============================
# FUNCIÓN: GRAFICAR HISTORIAL DE ENTRENAMIENTO (Sin cambios)
# ============================

def plot_training_history(history):
    ''' Graficar métricas de entrenamiento y validación '''
    plt.figure(figsize=(10, 6)) # Tamaño de la figura
    plt.plot(history.history['loss'], label='Pérdida Entrenamiento') # Eje Y: Pérdida de entrenamiento
    plt.plot(history.history['val_loss'], label='Pérdida Validación') # Eje X: Pérdida de validación
    plt.title("Pérdida del detector (Huber Loss)") # Título del gráfico
    plt.xlabel("Epoch") # Etiqueta del eje X para épocas o iteraciones
    plt.ylabel("Huber Loss") # Etiqueta del eje Y para "Huber Loss" función de pérdida utilizada en regresión robusta
    plt.legend() # Mostrar leyenda
    plt.grid(True) # Mostrar cuadrícula
    plt.savefig(TRAINING_LOSS_PLOT_PATH) # Guardar la figura en la ruta especificada
    # Se comenta plt.show() para evitar que el script se bloquee en entornos sin GUI (como WSL).
    # El gráfico ya se ha guardado en un archivo, que es el comportamiento deseado.
    plt.close()
    print(f"               ✅ Gráfico de pérdida guardado en {TRAINING_LOSS_PLOT_PATH}")

# ============================
# FUNCIÓN: ENTRENAR MODELO (Actualizada para usar las variables)
# ============================

def train_model(epochs=EPOCHS, batch_size=BATCH_SIZE, img_size=IMG_SIZE,learning_rate=LEARNING_RATE):
    ''' Función principal para entrenar el modelo de detección de placas vehiculares '''
    print("=========================================")
    print("             ENTRENAMIENTO DEL MODELO")
    print("=========================================")
    
    # 1. Configuración de directorios de salida
    # Crear las carpetas necesarias para guardar checkpoints, logs y plots
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) # Carpeta del modelo actual
    os.makedirs(LOGS_DIR, exist_ok=True) # Carpeta de logs
    os.makedirs(PLOTS_DIR, exist_ok=True) # Carpeta de plots

    '''
    # --------------------------  PREPROCESAMIENTO Y GUARDADO DE DATOS --------------------------
    # Se comenta ya que los datos ya fueron procesados y guardados previamente
    
    print(f"             1/7 Iniciando proceso de preprocesamiento, augmentation modular y guardado...")
    df_processed = preprocess_and_save_data_modular() # Puede recibir parámetros si es necesario
    '''
    print(f"             Epochs: {epochs}\n")
    print(f"             Batch Size: {batch_size}\n")
    print(f"             Learning Rate: {learning_rate}\n")
    # --------------------------  LECTURA DE DATOS PROCESADOS --------------------------
    print(f"             1/6 Cargando datos procesados desde CSV existente...")
    
    df_processed = pd.read_csv(PROCESSED_DATA_LABELS_CSV) # Cargar directamente el CSV procesado existente
    
    # --------------------------  Se separan los datos de entrenamiento y validación --------------------------
    
    print(f"             2/6 Separando datos de entrenamiento y validación...")
    
    df_train = df_processed[df_processed['split'] == 'train'] # Seleccionar filas de entrenamiento en el csv
    df_val = df_processed[df_processed['split'] == 'valid'] # Seleccionar filas de validación en el csv
    
    # --------------------------  CARGA DE DATOS PROCESADOS [Arrays] --------------------------
    '''
    # Cargar datos procesados en memoria
    # Se comenta ya que al cargar dataset de más de 5k imágenes colapsa la memoria RAM al generar los arrays
    X_train, y_train = load_processed_split(TRAIN_DATA_DIR, df_train)
    X_val, y_val = load_processed_split(VALIDATION_DATA_DIR, df_val)
    
    print(f"                2.1 Datos de Entrenamiento: {len(X_train)}")
    print(f"                2.2 Datos de Validación: {len(X_val)}")
    '''
    # --------------------------  USO DE GENERADORES DE DATOS  --------------------------
       
    # Crea los generadores de datos (¡Solución al error de memoria!)
    '''
    Funciona como un DataLoader que carga imágenes por lotes desde el disco durante el entrenamiento
    en lugar de cargar todo el dataset en memoria RAM. Esto es crucial para datasets grandes.
    '''
    train_generator = ImageBatchLoader(df_train, TRAIN_DATA_DIR, batch_size, image_shape=img_size, shuffle=True)
    validation_generator = ImageBatchLoader(df_val, VALIDATION_DATA_DIR, batch_size, image_shape=img_size, shuffle=False)
    
    # 2. Definir Callbacks
       
    print(f"             3/6 Entrenando y compilando modelo,...\n")
    model = build_detector(img_size=IMG_SIZE, learning_rate=learning_rate) # Construir el modelo de detección
    
    print(f"             4/6 Iniciando proceso de entrenamiento con Callbacks...\n")
    
    # Lista de Callbacks
    '''
    Lista de callbacks para el entrenamiento:
    - CSVLogger: Guarda el historial de entrenamiento en un archivo CSV.
    - EarlyStopping: Detiene el entrenamiento si la pérdida de validación no mejora después de 7 épocas.
    - ReduceLROnPlateau: Reduce la tasa de aprendizaje si la pérdida de validación no mejora después de 5 épocas.
    - ModelCheckpoint: Guarda el mejor modelo basado en la pérdida de validación.
    '''
    callbacks = [
        CSVLogger(TRAINING_LOG_CSV, append=True), # Log de entrenamiento en CSV
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1), # Early Stopping (Detener si no mejora)
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1), # Reducción de tasa de aprendizaje
        ModelCheckpoint(MODEL_PATH,monitor='val_loss',save_best_only=True,verbose=1) # Guardar el mejor modelo basado en val_loss
    ]
    '''
    Iniciar el entrenamiento del modelo
    Usamos los generadores de datos en lugar de arrays completos para evitar errores de memoria
    Nota: Al usar generadores, no se especifica el parámetro batch_size en model.fit()
    porque el generador ya maneja los lotes internamente.
    por ese motivo, se comenta la línea batch_size=BATCH_SIZE
    Además, los datos de entrada y validación son los generadores creados anteriormente.
    '''
    history = model.fit(
        #X_train, y_train, # Datos de Entrenamiento x: Imágenes, y: Bounding Boxes
        #validation_data=(X_val, y_val), # Datos de Validación X_val: Imágenes, y_val: Bounding Boxes
        train_generator,
        validation_data=validation_generator,
        epochs=epochs, # Número de epochs (Ciclos completos de entrenamiento)
        # batch_size=BATCH_SIZE, # Tamaño de batch (Número de muestras por actualización de gradiente) # Usado en el generador
        verbose=1, # Mostrar progreso en consola
        callbacks=callbacks, # Llamados para control del entrenamiento
    )
    
    print(f"\n               4.1 ✅ Entrenamiento completado.")

    #model.save(MODEL_PATH) # Se comenta ya que se guarda con ModelCheckpoint
    print(f"\n             5/6 ✅ Modelo guardado en {MODEL_PATH}")
    

    print(f"\n             Graficando historial de entrenamiento en {os.path.join(PLOTS_DIR, 'training_loss.png')}...")
    plot_training_history(history) # Graficar y guardar el historial de entrenamiento
    
    print(f"               6/6 Proceso de entrenamiento finalizado.")

    input("\nPresione enter para continuar...")

    return history, model

# test quick forward pass
if __name__ == "__main__":
    # construir modelo
    m = build_detector(img_size=IMG_SIZE, learning_rate=LEARNING_RATE)
    import numpy as np
    x = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
    out = m.predict(x)
    print("Modelo cargado — salida shape:", out.shape)  # debe ser (1,4)
