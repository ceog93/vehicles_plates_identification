import fiftyone as fo
import pandas as pd
import os

def view_web_dataset():
    # --- Paths ---
    img_dir = "dataset/images"
    csv_path = "dataset/_annotations.csv"

    # --- Cargar CSV ---
    print("\n")
    print("########################################")
    print("      VISOR WEB DEL DATASET")
    print("########################################\n")
    print("     1/4. Cargando anotaciones desde CSV...")
    df = pd.read_csv(csv_path)
    print("         1.1 Datos de cajas delimitadoras cargados.")
    print(f"            1.1.1 Total de anotaciones cargadas: {len(df)}")
    print(f"            1.1.2 Total de imágenes únicas: {df['filename'].nunique()}")
    print(f"            1.1.3 Clases presentes:", df['class'].unique())

    # --- Crear dataset vacío ---
    print("     2/4. Creando dataset FiftyOne...")
    dataset_name = "placas_colombia"
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)

    # --- Crear muestras ---
    print("     3/4. Agregando muestras al dataset...")
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row["filename"])
        if not os.path.exists(img_path):
            continue

        width, height = row["width"], row["height"]

        bbox = [
            row["xmin"] / width,
            row["ymin"] / height,
            (row["xmax"] - row["xmin"]) / width,
            (row["ymax"] - row["ymin"]) / height
        ]

        detection = fo.Detection(label=row["class"], bounding_box=bbox)
        sample = fo.Sample(filepath=img_path, ground_truth=fo.Detections(detections=[detection]))
        dataset.add_sample(sample)
    print(f"        3.1 Total de muestras en el dataset: {len(dataset)}")
    print("     4/4. Lanzando visor web...")
    print("\nPresione Ctrl+C en la terminal para cerrar el visor cuando termine.\n")
    # --- Guardar y lanzar visor ---
    dataset.persistent = True
    session = fo.launch_app(dataset)
    session.wait()
