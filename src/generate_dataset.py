# src/generate_dataset.py
"""
Genera dataset sintético normativo de placas colombianas.
Basado en la Resolución 5228 de 2016 (amarillo reflectivo, letras negras, borde negro).
"""

import os
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .config import DATA_DIR

# === Colores normativos aproximados ===
COLOR_FONDO = (255, 225, 40)   # Amarillo reflectivo
COLOR_TEXTO = (0, 0, 0)        # Negro
COLOR_BORDE = (0, 0, 0)        # Borde negro

# === Dimensiones estándar (px) ===
WIDTH, HEIGHT = 335, 170   # proporción real 2:1 aprox

# === Ciudades colombianas (100 aleatorias) ===
CIUDADES = [
    "Bogotá", "Medellín", "Cali", "Barranquilla", "Cartagena", "Bucaramanga", "Pereira", "Manizales", "Cúcuta", "Santa Marta",
    "Ibagué", "Neiva", "Villavicencio", "Tunja", "Popayán", "Sincelejo", "Montería", "Armenia", "Pasto", "Florencia",
    "Valledupar", "Riohacha", "Yopal", "Arauca", "Quibdó", "Leticia", "Mocoa", "San José del Guaviare", "Puerto Carreño",
    "Inírida", "Soacha", "Palmira", "Bello", "Itagüí", "Envigado", "Rionegro", "Girardot", "Zipaquirá", "Sogamoso",
    "Duitama", "Facatativá", "Fusagasugá", "Tuluá", "Cartago", "Buga", "Cereté", "Lorica", "Malambo", "Soledad",
    "Jamundí", "La Dorada", "Pitalito", "Yumbo", "Chía", "Cajicá", "Funza", "Mosquera", "Madrid", "Copacabana",
    "La Estrella", "Dosquebradas", "Villamaría", "Chiquinquirá", "Ocaña", "Apartadó", "Turbo", "Barrancabermeja",
    "Floridablanca", "Piedecuesta", "Sabanalarga", "Magangué", "Sahagún", "Montelíbano", "El Banco", "Aguachica",
    "Ciénaga", "Turbaco", "Cereté", "Planeta Rica", "San Andrés", "Providencia", "Melgar", "Espinal", "Garzón",
    "La Vega", "Guateque", "Pamplona", "Puerto Asís", "Tame", "Guamal", "San Gil", "Barbosa", "Socorro", "Mompox",
    "Ipiales", "Tumaco", "La Plata", "Chigorodó", "Candelaria", "Florida", "Caicedonia", "Roldanillo"
]

# === Generador de código de placa ===
def generar_codigo_placa():
    letras = ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    numeros = ''.join(random.choices("0123456789", k=3))
    return letras + numeros

# === Función auxiliar para compatibilidad Pillow ===
def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

# === Crea placa colombiana simulada ===
def generar_placa_img(codigo, ciudad):
    img = Image.new("RGB", (WIDTH, HEIGHT), COLOR_FONDO)
    draw = ImageDraw.Draw(img)

    # Marco negro rectangular con esquinas ligeramente redondeadas
    radio = 8
    draw.rounded_rectangle(
        [(5, 5), (WIDTH - 5, HEIGHT - 5)],
        radius=radio,
        outline=COLOR_BORDE,
        width=6
    )

    # Tipografías
    try:
        font = ImageFont.truetype("arialbd.ttf", 90)
        font_city = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
        font_city = ImageFont.load_default()

    # Texto principal (código)
    text_w, text_h = get_text_size(draw, codigo, font)
    draw.text(((WIDTH - text_w) / 2, (HEIGHT - text_h) / 2 - 20), codigo, font=font, fill=COLOR_TEXTO)

    # Ciudad inferior
    city_w, city_h = get_text_size(draw, ciudad, font_city)
    draw.text(((WIDTH - city_w) / 2, HEIGHT - city_h - 15), ciudad, font=font_city, fill=COLOR_TEXTO)

    return np.array(img)

# === Añade fondo aleatorio y ruido ===
def agregar_fondo_y_ruido(placa_img):
    fondo = np.full((300, 600, 3), np.random.randint(0, 255, size=3), dtype=np.uint8)
    h, w, _ = placa_img.shape

    if w > fondo.shape[1] or h > fondo.shape[0]:
        escala = min(fondo.shape[1] / w, fondo.shape[0] / h) * 0.9
        placa_img = cv2.resize(placa_img, (int(w * escala), int(h * escala)))

    y = random.randint(50, fondo.shape[0] - placa_img.shape[0] - 10)
    x = random.randint(50, fondo.shape[1] - placa_img.shape[1] - 10)

    fondo[y:y+placa_img.shape[0], x:x+placa_img.shape[1]] = placa_img

    ruido = np.random.normal(0, 25, fondo.shape).astype(np.uint8)
    return cv2.addWeighted(fondo, 0.85, ruido, 0.15, 0)

# === Generador de dataset completo ===
def generar_dataset(num_train=200, num_val=40):
    rutas = {
        "train": {"placa": os.path.join(DATA_DIR, "train", "placa"),
                  "no_placa": os.path.join(DATA_DIR, "train", "no_placa")},
        "val": {"placa": os.path.join(DATA_DIR, "val", "placa"),
                "no_placa": os.path.join(DATA_DIR, "val", "no_placa")}
    }

    # Crear carpetas
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

