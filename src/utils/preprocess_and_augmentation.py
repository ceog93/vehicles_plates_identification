import os
import random
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# ============================
# CONFIGURACIÓN (Importadas de src.config)
# ============================
from src.config import RAW_DATASET_DIR, TRAIN_DATA_DIR, VALIDATION_DATA_DIR
from src.config import RAW_DATA_LABELS_CSV, PROCESSED_DATA_LABELS_CSV
from src.config import IMG_SIZE

# Semilla para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================
# FUNCIÓN: PREPROCESAR, AUMENTAR Y GUARDAR (Modular y Completa)
# ============================

def preprocess_and_save_data_modular(
    raw_img_dir=RAW_DATASET_DIR,            # Directorio de imágenes crudas
    raw_csv_path=RAW_DATA_LABELS_CSV,       # Ruta del CSV con etiquetas crudas
    train_dir=TRAIN_DATA_DIR,               # Directorio para imágenes de entrenamiento procesadas
    val_dir=VALIDATION_DATA_DIR,            # Directorio para imágenes de validación procesadas
    processed_csv_path=PROCESSED_DATA_LABELS_CSV, # Ruta para guardar el CSV procesado
    img_size=IMG_SIZE,                      # Tamaño al que se redimensionarán las imágenes
    test_size=0.25,                         # Partición de datos para validación (25%)
    seed=SEED,                              # Semilla para reproducibilidad
    # Parámetros de Augmentation Modulares
    rotation_range=(-15, 15),               # Rotación en grados
    scale_range=(0.7, 1.3),                 # Escala (zoom)
    translate_range=(-0.10, 0.10),          # Traslación en porcentaje
    shear_range=(-10, 10),                  # Cizallamiento (shear) en grados
    brightness_factor=(0.4, 1.6),           # Ajuste de brillo
    blur_kernel_size_range=(1, 5),          # Kernel de Desenfoque Gaussiano (números impares)
    gaussian_noise_std_range=(0.0, 0.1),    # <-- Desviación estándar del ruido gaussiano
    num_augmentations=3                     # CANTIDAD DE COPIAS AUMENTADAS POR IMAGEN ORIGINAL
):
    '''
    Carga el dataset RAW (sin procesar), aplica Data Augmentation (aumento de datos), divide y guarda
    las imágenes redimensionadas y el CSV actualizado en las rutas PROCESADAS.
    '''
    print("=========================================")
    print("         PREPROCESAMIENTO Y AUGMENTATION MODULAR")
    print("=========================================")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    df_raw = pd.read_csv(raw_csv_path)
    unique_filenames = df_raw['filename'].unique()
    num_original_images = len(unique_filenames)
    temp_data = []
    
    print(f"     Imágenes originales encontradas: {num_original_images}")
    print("     ⏳ Procesando, aumentando y recopilando datos...")
    
    for i, filename in enumerate(tqdm(unique_filenames, desc="Generando Aumentaciones")):
        img_path = os.path.join(raw_img_dir, filename)
        group = df_raw[df_raw['filename'] == filename]
        
        if not os.path.exists(img_path): continue
            
        img = cv2.imread(img_path)
        if img is None: continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2] # Dimensiones originales
        cx, cy = w // 2, h // 2 
        
        row = group.iloc[0] 
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        
        # --- 2.1 Procesar Imagen Original (Redimensionada) ---
        img_resized_orig = cv2.resize(img, img_size)
        
        temp_data.append({
            'filename': filename,
            'image': img_resized_orig,
            'width': w, 'height': h,
            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
            'augmented': False
        })
        
        # --- 2.2 Generar Imagen Aumentada ---
        for k in range(num_augmentations):
            # Puntos de la caja delimitadora original (4 esquinas)
            pts_bb = np.array([
                [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
            ], dtype=np.float32)

            # 1. Generar Transformación Geométrica (Rotación, Escala, Traslación, Shear)
            #
            angle = random.uniform(*rotation_range) # Ángulo de rotación
            scale = random.uniform(*scale_range)   # Factor de escala
            M_rot = cv2.getRotationMatrix2D((cx, cy), angle, scale) # Matriz de rotación + escala
            
            tx = random.uniform(*translate_range) * w
            ty = random.uniform(*translate_range) * h
            M_rot[0, 2] += tx
            M_rot[1, 2] += ty
            
            shear_angle = random.uniform(*shear_range)
            shear_factor = np.tan(np.deg2rad(shear_angle))
            
            M_shear = np.float32([[1, shear_factor, 0], [0, 1, 0]])
            M_combined = M_shear @ np.vstack([M_rot, [0, 0, 1]]) 
            M_combined = M_combined[:2, :] 

            # 2. Aplicar la Transformación Geométrica Combinada
            img_aug = cv2.warpAffine(img, M_combined, (w, h))
            pts_bb_aug = cv2.transform(pts_bb[None, :, :], M_combined)[0]
            
            # Recalcular la nueva Bounding Box
            xmin_aug = int(np.min(pts_bb_aug[:, 0]))
            ymin_aug = int(np.min(pts_bb_aug[:, 1]))
            xmax_aug = int(np.max(pts_bb_aug[:, 0]))
            ymax_aug = int(np.max(pts_bb_aug[:, 1]))
            
            # Recortar (Clip)
            xmin_aug = np.clip(xmin_aug, 0, w)
            ymin_aug = np.clip(ymin_aug, 0, h)
            xmax_aug = np.clip(xmax_aug, 0, w)
            ymax_aug = np.clip(ymax_aug, 0, h)

            # 3. Aplicar Transformaciones Fotométricas
            
            # Brillo
            bright_factor = random.uniform(*brightness_factor)
            img_aug = img_aug.astype(np.float32) 
            img_aug = np.clip(img_aug * bright_factor, 0, 255).astype(np.uint8)
            
            # Desenfoque Gaussiano (Blur)
            min_k, max_k = blur_kernel_size_range
            ksize = random.choice([k for k in range(min_k, max_k + 1) if k % 2 != 0])
            if ksize > 1:
                img_aug = cv2.GaussianBlur(img_aug, (ksize, ksize), 0)
            
            # --- 3.3 Añadir Ruido Gaussiano Aleatorio ---
            
            # Seleccionar una desviación estándar aleatoria dentro del rango configurado
            std_noise = random.uniform(*gaussian_noise_std_range)

            if std_noise > 0:
                
                # 1. Convertir a float32 para las operaciones
                img_aug_float = img_aug.astype(np.float32)
                
                # 2. Generar el Ruido: Media 0 y la desviación estándar aleatoria (std_noise * 255)
                # Generamos el ruido en el rango [0, 255]
                noise = np.random.normal(0, std_noise * 255.0, img_aug_float.shape).astype(np.float32)
                
                # 3. Aplicar ruido y asegurar que los valores de píxel se mantengan entre [0, 255]
                img_aug = np.clip(img_aug_float + noise, 0, 255).astype(np.uint8)
            
            # 4. Redimensionar y guardar metadatos
            img_resized_aug = cv2.resize(img_aug, img_size)
            basename, ext = os.path.splitext(filename) # Separar nombre y extensión
            aug_filename = f"{basename}_aug_{k:02d}_{i:04d}{ext}" # Nuevo nombre para la imagen aumentada
            
            temp_data.append({
                'filename': aug_filename,
                'image': img_resized_aug,
                'width': w, 'height': h,
                'xmin': xmin_aug, 'ymin': ymin_aug, 'xmax': xmax_aug, 'ymax': ymax_aug,
                'augmented': True
            })

    # Resto de la función (División y Guardado)
    df_temp = pd.DataFrame(temp_data)
    num_total_images = len(df_temp)
    print(f"         Imágenes originales encontradas: {num_original_images}")
    print(f"         Total de imágenes (originales + aumentadas): {num_total_images}")

    # 3. Dividir el dataset
    df_train_meta, df_val_meta = train_test_split(df_temp, test_size=test_size, random_state=seed, shuffle=True)
    
    print(f"         División: Entrenamiento ({len(df_train_meta)} imágenes), Validación ({len(df_val_meta)} imágenes)")
    
    # 4. Guardar imágenes y construir el CSV final
    final_csv_data = []

    def save_images_and_update_csv(df_split, data_dir, split_type):
        for idx, row in df_split.iterrows():
            img_to_save = row['image']
            img_bgr = cv2.cvtColor(img_to_save.astype(np.uint8), cv2.COLOR_RGB2BGR) 
            output_path = os.path.join(data_dir, row['filename'])
            cv2.imwrite(output_path, img_bgr)

            final_csv_data.append({
                'filename': row['filename'],
                'width': row['width'], 
                'height': row['height'],
                'class': 'license_plate', 
                'xmin': row['xmin'],
                'ymin': row['ymin'],
                'xmax': row['xmax'],
                'ymax': row['ymax'],
                'split': split_type,
                'augmented': row['augmented']
            })

    print("         💾 Guardando imágenes de entrenamiento...")
    save_images_and_update_csv(df_train_meta, train_dir, 'train')
    
    print("         💾 Guardando imágenes de validación...")
    save_images_and_update_csv(df_val_meta, val_dir, 'valid')
    
    # 5. Guardar el nuevo archivo CSV unificado
    df_processed = pd.DataFrame(final_csv_data)
    df_processed.to_csv(processed_csv_path, index=False)
    
    print(f"         ✅ Datos procesados y guardados en:\n         - Entrenamiento: {train_dir}\n         - Validación: {val_dir}\n         - CSV: {processed_csv_path}")
    print("=========================================")
    input("\nPresione enter para continuar...\n")
    return df_processed