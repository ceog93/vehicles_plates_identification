'''
Módulo que define un generador de datos para cargar imágenes en lotes desde el disco.
Utiliza la clase Sequence de Keras para manejar grandes conjuntos de datos que no caben en memoria.
'''
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2 # Usaremos OpenCV para cargar y redimensionar imágenes

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
        """Carga y preprocesa el lote de imágenes y etiquetas desde el disco."""
        
        target_width, target_height = self.image_shape
        
        # Inicializar arrays vacíos para el lote
        # X: Input (Imágenes)
        X = np.empty((self.batch_size, target_height, target_width, self.n_channels), dtype=np.float32)
        # Y: Output (BBoxes normalizados)
        y = np.empty((self.batch_size, 4), dtype=np.float32) 

        # Cargar y preprocesar cada imagen
        for i, idx in enumerate(indexes_in_batch): # Iterar sobre los índices
            
            # --- 1. Obtener Datos por Índice ---
            file_name = self.file_names[idx]
            bbox = self.bbox_labels[idx]
            original_w = self.original_widths[idx]
            original_h = self.original_heights[idx]

            img_path = os.path.join(self.data_dir, file_name)
            img = cv2.imread(img_path)
            
            # Manejo básico de imagen no cargada
            if img is None:
                X[i,] = np.zeros((target_height, target_width, self.n_channels), dtype=np.float32)
                y[i,] = [0.0, 0.0, 0.0, 0.0]
                continue
                
            # --- 2. Procesamiento de IMAGEN (X) ---
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_shape)
            X[i,] = img.astype('float32') / 255.0

            # --- 3. Procesamiento y Normalización de ETIQUETA (Y) ---
            
            # Normalización a [0, 1] (Coordenada / Dimensión Original)
            xmin_norm = bbox[0] / original_w
            ymin_norm = bbox[1] / original_h
            xmax_norm = bbox[2] / original_w
            ymax_norm = bbox[3] / original_h
            
            # Almacenar el BBox normalizado
            y[i,] = [xmin_norm, ymin_norm, xmax_norm, ymax_norm]

        return X, y

# Elimina o deja sin usar la función load_processed_split()
# Def load_processed_split(data_dir: str, df_split: pd.DataFrame):
#    ... esta función ya no es necesaria