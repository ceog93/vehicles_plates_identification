# src/generate_dataset.py
"""
Crea un dataset sintético inicial de placas colombianas.
"""

import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from .config import DATA_DIR

def generar_codigo_placa():
    """Genera un texto aleatorio con formato tipo placa colombiana."""
    letras = ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    numeros = ''.join(random.choices("0123456789", k=3))
    return letras + numeros

def crear_placa_ancha():
    """Crea imagen tipo placa colombiana (fondo amarillo con letras negras)."""
    ancho, alto = 300, 100
    placa = np.full((alto, ancho, 3), (0, 220, 255), dtype=np.uint8)  # Amarillo
    img_pil = Image.fromarray(placa)
    draw = ImageDraw.Draw(img_pil)

    texto = generar_codigo_placa()
    try:
        # Usa Arial si está disponible, o la fuente por defecto
        font = ImageFont.truetype("arial.ttf", 50)
    except:
        font = ImageFont.load_default()

    # Calcula el tamaño del texto correctamente (compatibilidad Pillow ≥ 10)
    bbox = draw.textbbox((0, 0), texto, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    draw.text(((ancho - w) / 2, (alto - h) / 2 - 5), texto, font=font, fill=(0, 0, 0))
    return np.array(img_pil)

def agregar_fondo_y_ruido(img):
    """Coloca la placa sobre un fondo aleatorio y agrega ruido."""
    fondo = np.full((224, 224, 3), np.random.randint(0, 255, size=3), dtype=np.uint8)
    h, w, _ = img.shape
    x = random.randint(0, max(1, 224 - w))
    y = random.randint(0, max(1, 224 - h))

    # Asegurar que la placa no salga del borde
    fondo[y:y+h, x:x+w] = img[:min(h, 224 - y), :min(w, 224 - x)]
    
    ruido = np.random.normal(0, 25, fondo.shape).astype(np.int16)
    fondo_ruidoso = np.clip(fondo.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    return fondo_ruidoso

def generar_dataset(num_train=200, num_val=40):
    """Genera imágenes sintéticas de entrenamiento y validación."""
    rutas = {
        "train": {
            "placa": os.path.join(DATA_DIR, "train", "placa"),
            "no_placa": os.path.join(DATA_DIR, "train", "no_placa")
        },
        "val": {
            "placa": os.path.join(DATA_DIR, "val", "placa"),
            "no_placa": os.path.join(DATA_DIR, "val", "no_placa")
        }
    }

    # Crear carpetas si no existen
    for tipo in rutas:
        for clase in rutas[tipo]:
            os.makedirs(rutas[tipo][clase], exist_ok=True)

    print("[INFO] Generando dataset sintético...")

    for tipo, carpetas in rutas.items():
        n = num_train if tipo == "train" else num_val
        for i in range(n):
            # Imagen con placa
            placa = crear_placa_ancha()
            img_final = agregar_fondo_y_ruido(placa)
            path = os.path.join(carpetas["placa"], f"placa_{i}.jpg")
            cv2.imwrite(path, img_final)

            # Imagen sin placa (solo fondo)
            fondo = np.full((224, 224, 3), np.random.randint(0, 255, size=3), dtype=np.uint8)
            path2 = os.path.join(carpetas["no_placa"], f"no_placa_{i}.jpg")
            cv2.imwrite(path2, fondo)

    print("[✔] Dataset sintético generado exitosamente.")
    print(f"Ubicación: {DATA_DIR}")
