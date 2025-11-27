'''
Visor web para el dataset utilizando FiftyOne
'''
import fiftyone as fo
import pandas as pd
import os
from tqdm import tqdm
from src.config import PROCESED_DATA_DIR
from src.config import PROCESSED_DATA_LABELS_CSV

def view_web_dataset(img_dir=PROCESED_DATA_DIR, csv_path=PROCESSED_DATA_LABELS_CSV):
    '''
    Visor web para el dataset utilizando FiftyOne.
    Lanza el visor utilizando las rutas de imagen y CSV especificadas, 
    manejando subcarpetas ('train' y 'validation')
    '''
    # --- CORRECCIÓN: Deshabilitar paginador de consola ---
    # Esto evita que la consola se bloquee en un visor de texto (como 'less')
    fo.config.disable_pagination = True

    # --- Cargar CSV ---
    print("\n")
    print("########################################")
    print("      VISOR WEB DEL DATASET")
    print("########################################\n")
    print(f"      1/4. Cargando anotaciones desde CSV: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"      ❌ ERROR: Archivo CSV no encontrado en {csv_path}")
        return

    print("          1.1 Datos de cajas delimitadoras cargados.")
    print(f"            1.1.1 Total de anotaciones cargadas: {len(df)}")
    print(f"            1.1.2 Total de imágenes únicas: {df['filename'].nunique()}")
    print(f"            1.1.3 Clases presentes:", df['class'].unique())
    
    # --- Mapeo de split a nombre de carpeta ---
    # La columna 'split' tiene 'train' o 'valid', y las carpetas se llaman 'train' y 'validation'
    split_to_folder_map = {
        'train': 'train',
        'valid': 'validation' # Usar el nombre de la carpeta de config.py
    }

    # --- Crear dataset vacío ---
    dataset_name = "placas_colombia_processed" 
    
    print(f"      2/4. Creando dataset FiftyOne: {dataset_name}...")
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)

    # --- Crear muestras ---
    print(f"      3/4. Agregando muestras al dataset desde {img_dir}...")
    
    samples_to_add = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Añadiendo Muestras a FiftyOne"):
        
        # LÓGICA CLAVE: Construir la ruta completa manejando la subcarpeta
        split_type = row.get('split')
        folder_name = split_to_folder_map.get(split_type)
        
        if not folder_name:
            continue # Omitir si la fila no tiene un split válido

        # CONSTRUIR LA RUTA ABSOLUTA: /data/processed_data/train/imagen.jpg
        img_path = os.path.join(img_dir, folder_name, row["filename"])
        
        if not os.path.exists(img_path):
            continue

        width, height = row["width"], row["height"]
        
        if width <= 0 or height <= 0:
             continue
        
        # Normalizar las coordenadas de la Bounding Box
        bbox = [
            row["xmin"] / width,
            row["ymin"] / height,
            (row["xmax"] - row["xmin"]) / width,
            (row["ymax"] - row["ymin"]) / height
        ]

        detection = fo.Detection(label=row["class"], bounding_box=bbox)
        
        # Agregar los tags para poder filtrar por split y por imagen aumentada
        tags = []
        if pd.notna(row['split']):
            tags.append(row['split']) # 'train' o 'valid'
            
        if row.get('augmented', False): # Por defecto False si no existe
            tags.append('augmented')
        
        sample = fo.Sample(filepath=img_path, ground_truth=fo.Detections(detections=[detection]), tags=tags)
        samples_to_add.append(sample)
        
    dataset.add_samples(samples_to_add)
    
    print(f"\n      3.1 Total de muestras en el dataset: {len(dataset)}")
    
    if len(dataset) == 0:
        print("      ❌ ERROR: No se pudieron agregar muestras. Verifique las rutas de las imágenes.")
        return

    print("      4/4. Lanzando visor web FiftyOne...")
    # Mensaje modificado para WSL: se indica al usuario que abra la URL manualmente.
    print(f"\nVisor listo. Copie y pegue esta URL en su navegador de Windows: http://localhost:{fo.config.default_app_port}")
    print("\nPresione **Ctrl+C** en la terminal para **cerrar el visor** cuando termine.\n")
    
    # --- Guardar y lanzar visor ---
    dataset.persistent = True
    # Se agrega 'auto=False' para que no intente abrir el navegador, lo cual falla en WSL.
    session = fo.launch_app(dataset, auto=False)
    session.wait()
    print("\nVisor FiftyOne cerrado.")
    input("Presione Enter para continuar...")