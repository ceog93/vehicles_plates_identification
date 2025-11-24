import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Configuración
OUTPUT_DIR = "data_ocr"
IMG_SIZE = (32, 32) # Tamaño pequeño estándar para OCR
SAMPLES_PER_CLASS = 1000 # 1000 imágenes por cada letra/número
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FONT_PATH = "ruta/a/tu/fuente/placa.ttf" # ¡USA LA MISMA FUENTE QUE TU GENERADOR DE PLACAS!

def generar_caracteres():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Cargar fuente (ajusta el tamaño para que quepa en 32x32)
    try:
        font = ImageFont.truetype(FONT_PATH, 28)
    except:
        print("Fuente no encontrada, usando default (bajará la precisión)")
        font = ImageFont.load_default()

    for char in CHARS:
        # Crear carpeta para cada clase (ej: data_ocr/A, data_ocr/B)
        class_dir = os.path.join(OUTPUT_DIR, char)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"Generando clase: {char}")
        
        for i in range(SAMPLES_PER_CLASS):
            # Crear imagen de fondo (puedes añadir ruido aquí para hacerlo robusto)
            img = Image.new('L', IMG_SIZE, color=random.randint(200, 255)) # Fondo gris claro
            draw = ImageDraw.Draw(img)
            
            # Centrar texto
            bbox = draw.textbbox((0, 0), char, font=font)
            w_text, h_text = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (IMG_SIZE[0] - w_text) / 2
            y = (IMG_SIZE[1] - h_text) / 2
            
            # Dibujar caracter con variaciones leves de posición
            draw.text((x + random.randint(-2, 2), y + random.randint(-2, 2)), 
                      char, font=font, fill=random.randint(0, 50))
            
            # Guardar
            img.save(os.path.join(class_dir, f"{char}_{i}.png"))

if __name__ == "__main__":
    generar_caracteres()