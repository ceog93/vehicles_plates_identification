'''
Módulo que define un generador de datos para cargar imágenes en lotes desde el disco.
Utiliza la clase Sequence de Keras para manejar grandes conjuntos de datos que no caben en memoria.
'''
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2 # Usaremos OpenCV para cargar y redimensionar imágenes
# Importar parámetros del modelo para mantener consistencia entre salida y etiquetas
from src.models.efficient_detector_multi_placa import GRID_SIZE, NUM_ANCHORS as BBOX_ANCHORS, NUM_CLASSES, BBOX_DIM

# OUTPUT_DIM debe coincidir con la capa final del modelo: anchors * (bbox_dim + num_classes)
OUTPUT_DIM = int(BBOX_ANCHORS * (BBOX_DIM + NUM_CLASSES))

class ImageSequence(tf.keras.utils.Sequence):
    """Generador de datos para Keras que carga imágenes por lotes."""
    
    def __init__(self, df: pd.DataFrame, data_dir: str, batch_size: int, 
                 image_shape: tuple = (640, 640), n_channels: int = 3, 
                 shuffle: bool = True, **kwargs): # ACEPTAR **kwargs
        
        # LLAMAR AL CONSTRUCTOR PADRE
        super().__init__(**kwargs)
        
        # Almacenamos el DataFrame y otros parámetros
        self.df = df
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.n_channels = n_channels
        self.shuffle = shuffle
        
        # La columna 'file_name' debe contener el nombre del archivo de imagen
        # La columna 'label' debe contener la etiqueta (ej. el caracter o la clase)
        # Ajusta estos nombres de columna si son diferentes en tu df_train
        self.file_names = df['filename'].values 
        self.bbox_labels = df[['xmin', 'ymin', 'xmax', 'ymax']].values # Etiquetas Y
        
        # Necesario para la normalización del BBox
        self.original_widths = df['width'].values
        self.original_heights = df['height'].values
        
        self.on_epoch_end()

    def __len__(self):
        """Devuelve el número de batches por época."""
        # np.floor asegura que devolvemos un número entero de batches
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        """Genera un batch de datos (imágenes y etiquetas)."""
        
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        indexes_in_batch = self.indexes[start:end]

        # Obtener los datos necesarios del lote
        # ⚠️ IMPORTANTE: PASAR LOS ÍNDICES EN LUGAR DE LOS ARRAYS REPETIDOS
        # Esto permite que __data_generation acceda a width y height con el mismo índice
        X, y = self.__data_generation(indexes_in_batch) 
        return X, y

    def on_epoch_end(self):
        """Se llama al final de cada época, para reordenar los datos si es necesario."""
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes_in_batch):
        """Carga y preprocesa el lote de imágenes y etiquetas desde el disco (Generador Multiplaca)."""
        
        target_width, target_height = self.image_shape
        
        # Inicializar arrays vacíos para el lote
        X = np.empty((self.batch_size, target_height, target_width, self.n_channels), dtype=np.float32)
        
        # ⚠️ CAMBIO CRÍTICO: El output y ahora es (BATCH, GRID_SIZE, GRID_SIZE, OUTPUT_DIM)
        y = np.zeros((self.batch_size, GRID_SIZE, GRID_SIZE, OUTPUT_DIM), dtype=np.float32) 
        
        # Cargar y preprocesar cada imagen
        for i, idx in enumerate(indexes_in_batch): 
            
            # --- 1. Obtener Datos por Índice ---
            file_name = self.file_names[idx]
            bbox = self.bbox_labels[idx]
            original_w = self.original_widths[idx]
            original_h = self.original_heights[idx]

            # ... (Carga de imagen y procesamiento de X idéntico a tu código original) ...
            # Se asume que X[i,] es llenado correctamente con la imagen normalizada
            # --- 2. Procesamiento de IMAGEN (X) ---
            # (tu código de carga y resize va aquí, omitido por brevedad)
            
            img_path = os.path.join(self.data_dir, file_name)
            img = cv2.imread(img_path)
            if img is None:
                # Si la imagen falta o está corrupta, rellenar la posición con ceros
                # y dejar la etiqueta en cero (sin objeto). Esto evita desajustes
                # en los batches y permite continuar el entrenamiento sin fallos.
                X[i,] = np.zeros((target_height, target_width, self.n_channels), dtype=np.float32)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.image_shape)
                X[i,] = img.astype('float32') / 255.0

            # --- 3. Procesamiento de ETIQUETA (Y) - Transformación YOLO/SSD ---
            
            # Normalización a [0, 1]
            xmin_norm = bbox[0] / original_w
            ymin_norm = bbox[1] / original_h
            xmax_norm = bbox[2] / original_w
            ymax_norm = bbox[3] / original_h
            
            # Convertir a formato (cx, cy, w, h)
            cx_norm = (xmin_norm + xmax_norm) / 2
            cy_norm = (ymin_norm + ymax_norm) / 2
            w_norm = xmax_norm - xmin_norm
            h_norm = ymax_norm - ymin_norm
            
            # Determinar la CELDA de la cuadrícula (i, j) donde cae el centro (cx_norm, cy_norm)
            cell_x = int(cx_norm * GRID_SIZE)
            cell_y = int(cy_norm * GRID_SIZE)
            
            # ⚠️ Verificación de límites
            if cell_x >= GRID_SIZE or cell_y >= GRID_SIZE: continue

            # Normalizar el centro (cx, cy) respecto a la celda (coordenadas locales)
            cx_local = (cx_norm * GRID_SIZE) - cell_x
            cy_local = (cy_norm * GRID_SIZE) - cell_y
            
            # Construir el vector de Ground Truth para la celda (cell_y, cell_x)
            # Asignar a la primera ancla (BBOX_ANCHORS=3, usamos la primera por simplicidad)
            anchor_index = 0 
            
            # Índice de inicio para la primera ancla (5 * 0)
            start_idx = 5 * anchor_index 
            
            # [Confianza (1), cx_local, cy_local, w_norm, h_norm]
            y[i, cell_y, cell_x, start_idx : start_idx + 5] = [1.0, cx_local, cy_local, w_norm, h_norm]
            
            # Clasificación (opcional, si NUM_CLASSES > 1, se colocaría en los últimos índices)
            if NUM_CLASSES > 0:
                y[i, cell_y, cell_x, -1] = 1.0 # Clase 1: Placa

        return X, y