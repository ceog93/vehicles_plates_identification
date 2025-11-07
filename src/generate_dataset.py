# src/generate_dataset.py
"""
Genera dataset sintético normativo de placas colombianas (fondo amarillo, letras negras).
"""

import os
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .config import DATA_DIR

# === Colores normativos aproximados ===
COLOR_FONDO = (255, 211, 0)   # Amarillo reflectivo (RGB)
COLOR_TEXTO = (0, 0, 0)       # Negro
COLOR_BORDE = (0, 0, 0)       # Negro

# === Dimensiones estándar ===
WIDTH, HEIGHT = 335, 170   # proporción ~2:1

# === Ciudades colombianas (ejemplo) ===
CIUDADES = [
    "Bogotá D.C.", "Medellín", "Cali", "Barranquilla", "Cartagena", "Bucaramanga", "Pereira", "Manizales",
    "Cúcuta", "Santa Marta", "Ibagué", "Neiva", "Villavicencio", "Tunja", "Popayán", "Sincelejo", "Montería",
    "Armenia", "Pasto", "Valledupar", "Riohacha", "Soledad", "Palmira", "Itagüí", "Rionegro", "Envigado", "Soacha"
]

def generar_codigo_placa():
    letras = ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    numeros = ''.join(random.choices("0123456789", k=3))
    return letras + numeros

def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def generar_placa_img(codigo, ciudad):
    img = Image.new("RGB", (WIDTH, HEIGHT), COLOR_FONDO)
    draw = ImageDraw.Draw(img)

    # Marco con esquinas levemente redondeadas
    radio = 8
    draw.rounded_rectangle(
        [(5, 5), (WIDTH - 5, HEIGHT - 5)],
        radius=radio,
        outline=COLOR_BORDE,
        width=6
    )

    # Cargar fuentes
    try:
        font = ImageFont.truetype("arialbd.ttf", 95)  # texto grande
        font_city = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()
        font_city = ImageFont.load_default()

    # Texto principal (placa)
    text_w, text_h = get_text_size(draw, codigo, font)
    draw.text(
        ((WIDTH - text_w) / 2, (HEIGHT - text_h) / 2 - 15),
        codigo,
        font=font,
        fill=COLOR_TEXTO
    )

    # Ciudad inferior
    city_w, city_h = get_text_size(draw, ciudad, font_city)
    draw.text(
        ((WIDTH - city_w) / 2, HEIGHT - city_h - 15),
        ciudad,
        font=font_city,
        fill=COLOR_TEXTO
    )

    # Convertir de RGB (Pillow) a BGR (OpenCV)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def agregar_fondo_y_ruido(placa_img):
    fondo = np.full((300, 600, 3), np.random.randint(0, 255, size=3), dtype=np.uint8)
    h, w, _ = placa_img.shape

    if w > fondo.shape[1] or h > fondo.shape[0]:
        escala = min(fondo.shape[1] / w, fondo.shape[0] / h) * 0.9
        placa_img = cv2.resize(placa_img, (int(w * escala), int(h * escala)))

    y = random.randint(30, fondo.shape[0] - placa_img.shape[0] - 10)
    x = random.randint(30, fondo.shape[1] - placa_img.shape[1] - 10)

    fondo[y:y+placa_img.shape[0], x:x+placa_img.shape[1]] = placa_img

    ruido = np.random.normal(0, 25, fondo.shape).astype(np.uint8)
    return cv2.addWeighted(fondo, 0.85, ruido, 0.15, 0)

def generar_dataset(num_train=200, num_val=40):
    rutas = {
        "train": {"placa": os.path.join(DATA_DIR, "train", "placa"),
                  "no_placa": os.path.join(DATA_DIR, "train", "no_placa")},
        "val": {"placa": os.path.join(DATA_DIR, "val", "placa"),
                "no_placa": os.path.join(DATA_DIR, "val", "no_placa")}
    }

    for tipo in rutas:
        for clase in rutas[tipo]:
            os.makedirs(rutas[tipo][clase], exist_ok=True)

    print("[INFO] Generando dataset sintético (placas normativas)...")

    for tipo, carpetas in rutas.items():
        n = num_train if tipo == "train" else num_val
        for i in range(n):
            codigo = generar_codigo_placa()
            ciudad = random.choice(CIUDADES)

            placa_img = generar_placa_img(codigo, ciudad)
            img_final = agregar_fondo_y_ruido(placa_img)

            cv2.imwrite(os.path.join(carpetas["placa"], f"placa_{i}.jpg"), img_final)

            # Fondo sin placa
            fondo = np.full((300, 600, 3), np.random.randint(0, 255, size=3), dtype=np.uint8)
            cv2.imwrite(os.path.join(carpetas["no_placa"], f"no_placa_{i}.jpg"), fondo)

    print("✅ Dataset sintético generado exitosamente.")
    print(f"Ubicación: {DATA_DIR}")
