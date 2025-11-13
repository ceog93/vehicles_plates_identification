import fiftyone as fo
import pandas as pd
import os

# --- Paths ---
img_dir = "dataset/images"
csv_path = "dataset/_annotations.csv"

# --- Cargar CSV ---
df = pd.read_csv(csv_path)

# --- Crear dataset vacío ---
dataset_name = "placas_colombia"
if dataset_name in fo.list_datasets():
    fo.delete_dataset(dataset_name)

dataset = fo.Dataset(dataset_name)

# --- Crear muestras ---
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

# --- Guardar y lanzar visor ---
dataset.persistent = True
session = fo.launch_app(dataset)
session.wait()
