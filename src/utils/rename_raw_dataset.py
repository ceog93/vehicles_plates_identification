import pandas as pd
import os

# -------------------------- CONFIGURACIÓN DE RUTAS --------------------------
# DIRECTORIO BASE: La carpeta donde se ejecuta este script. Si está en la misma carpeta que las imágenes
# y el CSV, puedes dejarlo como '.'
#BASE_DIR = '.' 

# 1. RUTA donde se encuentra el archivo CSV
# Ejemplo: 'data/labels' si el CSV está en BASE_DIR/data/labels
#CARPETA_CSV = os.path.join(BASE_DIR, 'data') 
from src.config import LABELS_DIR as CARPETA_CSV
# 2. RUTA donde se encuentran los archivos de imagen.
# Ejemplo: 'data/images' si las imágenes están en BASE_DIR/data/images
#CARPETA_IMAGENES = os.path.join(BASE_DIR, 'data', 'raw_images')
from src.config import RAW_DATASET_DIR
# -------------------------- CONFIGURACIÓN DE ARCHIVOS -----------------------
# 3. Nombre del archivo CSV de entrada
NOMBRE_CSV = '_raw_data_labels.csv'
CSV_PATH_IN = os.path.join(CARPETA_CSV, NOMBRE_CSV)

# 4. Nombre del nuevo archivo CSV de salida
NOMBRE_CSV_SALIDA = '_raw_data_labels_renamed.csv'
CSV_PATH_OUT = os.path.join(CARPETA_CSV, NOMBRE_CSV_SALIDA)

# 5. Nombre de la columna con los nombres de archivo ORIGINALES
COLUMNA_NOMBRE_IMAGEN = 'filename' 

# 6. Formato del nuevo nombre (Ejemplo: PLACA_000001.jpg)
FORMATO_NUEVO_NOMBRE = 'PLACA_{:06d}.jpg' 

# ----------------------------------------------------------------------------

def renombrar_y_actualizar_raw_dataset(csv_path_in = CSV_PATH_IN, img_dir = RAW_DATASET_DIR, col_imagen = COLUMNA_NOMBRE_IMAGEN, formato_nuevo = FORMATO_NUEVO_NOMBRE, csv_path_out = CSV_PATH_OUT):
    """
    Lee el CSV desde su ruta, renombra los archivos de imagen en su ruta
    y actualiza TODAS las filas en el CSV con el nuevo nombre, guardándolo 
    en una nueva ruta.
    """
    print(f"Buscando CSV en: {csv_path_in}")
    print(f"Buscando imágenes en: {img_dir}\n")
    
    try:
        # 1. Cargar el archivo CSV
        df = pd.read_csv(csv_path_in)
        
        if col_imagen not in df.columns:
            print(f"Error: La columna '{col_imagen}' no se encontró en el CSV.")
            return

        # 2. Identificar nombres de archivo ÚNICOS para procesar una sola vez
        nombres_unicos_originales = df[col_imagen].unique()
        print(f"Se encontraron {len(nombres_unicos_originales)} archivos de imagen únicos para renombrar.")

        # Diccionario para mapear (nombre_antiguo -> nombre_nuevo)
        mapa_renombrado = {}
        contador = 1
        archivos_fallidos = []

        # 3. Iterar sobre los nombres únicos para realizar el renombrado físico
        for nombre_viejo in nombres_unicos_originales:
            
            # Generar el nuevo nombre de archivo estandarizado
            nombre_nuevo = formato_nuevo.format(contador)
            
            # Construir las rutas COMPLETAS para os.rename()
            ruta_vieja = os.path.join(img_dir, nombre_viejo)
            ruta_nueva = os.path.join(img_dir, nombre_nuevo)
            
            try:
                # Renombrar el archivo físico
                os.rename(ruta_vieja, ruta_nueva)
                
                # Guardar el mapeo del nombre puro (sin ruta de carpeta)
                mapa_renombrado[nombre_viejo] = nombre_nuevo
                print(f"Renombrado: {nombre_viejo} -> {nombre_nuevo}")
                
                contador += 1
            
            except FileNotFoundError:
                archivos_fallidos.append(nombre_viejo)
                # Mantiene el nombre original en el CSV si el archivo no existe
                mapa_renombrado[nombre_viejo] = nombre_viejo 
                print(f"ADVERTENCIA: Archivo físico {ruta_vieja} no encontrado. Se mantendrá el nombre en el CSV.")
            except Exception as e:
                archivos_fallidos.append(nombre_viejo)
                mapa_renombrado[nombre_viejo] = nombre_viejo
                print(f"ERROR al procesar {nombre_viejo}: {e}")
        
        # 4. Actualizar la columna 'filename' en TODO el DataFrame
        df[col_imagen] = df[col_imagen].map(mapa_renombrado)
        
        # 5. Guardar el nuevo CSV actualizado
        df.to_csv(csv_path_out, index=False)
        
        print("\n--- PROCESO TERMINADO EXITOSAMENTE ---")
        print(f"Archivos renombrados: {contador - 1}")
        print(f"Archivos no encontrados/fallidos: {len(archivos_fallidos)}")
        print(f"CSV actualizado guardado como: {csv_path_out}")
        input("\nPresione Enter para continuar...")
        
    except FileNotFoundError:
        print(f"\nERROR FATAL: El archivo CSV de entrada no fue encontrado en: {csv_path_in}")
        input("Presione Enter para continuar...")
    except Exception as e:
        print(f"Ocurrió un error general: {e}")
        input("Presione Enter para continuar...")