"""
src/utils/batch_data_loader.py

Cargador de datos por lotes para el entrenamiento con Keras. Carga imágenes en lotes desde el disco
durante el entrenamiento en lugar de cargar todo el conjunto de datos en la RAM. Este módulo
utiliza Keras Sequence para manejar grandes conjuntos de datos que no caben en la memoria.

La clase ImageBatchLoader genera lotes de imágenes y etiquetas de cajas delimitadoras
en el formato requerido por el modelo detector de múltiples placas (predicciones basadas en grid).
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
# Importar parámetros del modelo para mantener la consistencia entre la salida y las etiquetas
from src.models.efficient_detector_multi_placa import TAM_GRID, NUM_ANCLAS as BBOX_ANCHORS, NUM_CLASES, DIM_CAJA

# OUTPUT_DIM must match the final layer of the model: anchors * (dim_caja + num_clases)
OUTPUT_DIM = int(BBOX_ANCHORS * (DIM_CAJA + NUM_CLASES))


class ImageBatchLoader(tf.keras.utils.Sequence):
    """
    Generador de datos de Keras que carga imágenes en lotes.
    
    Carga etiquetas de imágenes y cajas delimitadoras desde el disco en lotes durante el entrenamiento,
    procesándolas sobre la marcha para una mayor eficiencia de memoria.
    """
    
    def __init__(self, df: pd.DataFrame, data_dir: str, batch_size: int, 
                 image_shape: tuple = (640, 640), n_channels: int = 3, 
                 shuffle: bool = True, **kwargs):
        """
        Inicializa el cargador de lotes.
        
        Args:
            df: DataFrame con las columnas 'filename', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height'
            data_dir: Directorio que contiene los archivos de imagen
            batch_size: Número de muestras por lote
            image_shape: Dimensiones de la imagen de destino (alto, ancho)
            n_channels: Número de canales de color (3 para RGB)
            shuffle: Si se deben mezclar las muestras al final de cada época
        """
        super().__init__(**kwargs)
        
        self.df = df
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.n_channels = n_channels
        self.shuffle = shuffle
        
        self.file_names = df['filename'].values 
        self.bbox_labels = df[['xmin', 'ymin', 'xmax', 'ymax']].values
        
        # Almacenar dimensiones originales para la normalización de la caja delimitadora
        self.original_widths = df['width'].values
        self.original_heights = df['height'].values
        
        self.on_epoch_end()

    def __len__(self):
        """Devuelve el número de lotes por época."""
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        indexes_in_batch = self.indexes[start:end]

        X, y = self.__data_generation(indexes_in_batch) 
        return X, y

    def on_epoch_end(self):
        """Se llama al final de la época para mezclar los datos si es necesario."""
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes_in_batch):
        """Carga y preprocesa un lote de imágenes y etiquetas desde el disco."""
        target_width, target_height = self.image_shape
        
        # Inicializar arrays vacíos para el lote
        X = np.empty((self.batch_size, target_height, target_width, self.n_channels), dtype=np.float32)
        
        # Forma de la salida: (BATCH, TAM_GRID, TAM_GRID, OUTPUT_DIM)
        y = np.zeros((self.batch_size, TAM_GRID, TAM_GRID, OUTPUT_DIM), dtype=np.float32) 
        
        for i, idx in enumerate(indexes_in_batch): 
            file_name = self.file_names[idx]
            bbox = self.bbox_labels[idx]
            original_w = self.original_widths[idx]
            original_h = self.original_heights[idx]

            # --- Procesamiento de la Imagen (X) ---
            img_path = os.path.join(self.data_dir, file_name)
            img = cv2.imread(img_path)
            if img is None:
                # Si la imagen falta o está corrupta, rellenar con ceros
                X[i,] = np.zeros((target_height, target_width, self.n_channels), dtype=np.float32)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.image_shape)
                X[i,] = img.astype('float32') / 255.0

            # --- Procesamiento de la Etiqueta (Y) - Formato de caja delimitadora basado en grid ---
            
            # Normalizar a [0, 1]
            xmin_norm = bbox[0] / original_w
            ymin_norm = bbox[1] / original_h
            xmax_norm = bbox[2] / original_w
            ymax_norm = bbox[3] / original_h
            
            # Convertir a (centro_x, centro_y, ancho, alto)
            cx_norm = (xmin_norm + xmax_norm) / 2
            cy_norm = (ymin_norm + ymax_norm) / 2
            w_norm = xmax_norm - xmin_norm
            h_norm = ymax_norm - ymin_norm
            
            # Determinar la celda del grid donde cae el centro de la caja delimitadora
            cell_x = int(cx_norm * TAM_GRID)
            cell_y = int(cy_norm * TAM_GRID)
            
            # Comprobación de límites
            if cell_x >= TAM_GRID or cell_y >= TAM_GRID: 
                continue

            # Normalizar el centro relativo a la celda del grid (coordenadas locales)
            cx_local = (cx_norm * TAM_GRID) - cell_x
            cy_local = (cy_norm * TAM_GRID) - cell_y
            
            # Asignar el ground truth a la primera ancla (simplificado: usando solo una ancla por celda)
            anchor_index = 0 
            start_idx = 5 * anchor_index 
            
            # [confidence, cx_local, cy_local, w_norm, h_norm]
            y[i, cell_y, cell_x, start_idx : start_idx + 5] = [1.0, cx_local, cy_local, w_norm, h_norm]
            
            # Etiqueta de clase (si NUM_CLASES > 1, se colocaría en los últimos índices)
            if NUM_CLASES > 0:
                y[i, cell_y, cell_x, -1] = 1.0

        return X, y
