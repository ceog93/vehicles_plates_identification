# src/generate_datasheet.py
'''
Crea un dataset sintético inicial de placas colombianas
'''

# src/generate_dataset.py
import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from src.config import DATA_DIR

def generar_codigo_placa():
    letras = ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    numeros = ''.join(random.choices("0123456789", k=3))
    return letras + numeros

def crear_placa_ancha():
    """Crea imagen tipo placa colombiana (fondo amarillo con letras negras)."""
    ancho, alto = 300, 100
    placa = np.full((alto, ancho, 3), (0, 220, 255), dtype=np.uint8)  # Amarillo
    draw = ImageDraw.Draw(Image.fromarray(placa))
    texto = generar_codigo_placa()

    try:
        font = ImageFont.truetype("arial.ttf", 50)
    except:
        font = ImageFont.load_default()

    img_pil = Image.fromarray(placa)
    draw = ImageDraw.Draw(img_pil)
    w, h = draw.textsize(texto, font=font)
    draw.text(((ancho - w) / 2, (alto - h) / 2 - 5), texto, font=font, fill=(0, 0, 0))
    placa = np.array(img_pil)
    return placa

def agregar_fondo_y_ruido(img):
    """Coloca la placa sobre un fondo aleatorio y agrega ruido."""
    fondo = np.full((224, 224, 3), np.random.randint(0, 255, size=3), dtype=np.uint8)
    x = random.randint(10, 100)
    y = random.randint(50, 120)
    h, w, _ = img.shape
    fondo[y:y+h, x:x+w] = img
    ruido = np.random.normal(0, 25, fondo.shape).astype(np.uint8)
    return cv2.addWeighted(fondo, 0.8, ruido, 0.2, 0)

def generar_dataset(num_train=200, num_val=40):
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

    # Crear carpetas
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
