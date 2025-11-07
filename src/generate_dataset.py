# src/generate_dataset.py
"""
Generador avanzado de dataset sintético de placas colombianas.
Incluye: fondo amarillo normativo, doble borde, texto escalable y centrado,
sombra/contorno en texto, óvalo "CO", y tornillos simulados.
Compatible con versiones modernas de Pillow (usa textbbox).
"""

import os
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .config import DATA_DIR, IMG_SIZE

# Colores (RGB)
COLOR_FONDO = (255, 211, 0)   # amarillo reflectivo aproximado
COLOR_TEXTO = (0, 0, 0)       # negro
COLOR_BORDE = (0, 0, 0)       # negro
COLOR_OVALO = (230, 230, 230) # color interior leve para óvalo CO

# Dimensiones placa en px (proporción ~2:1)
WIDTH, HEIGHT = 335, 170

# Lista de ciudades (ejemplos)
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
    """Devuelve (w,h) del texto compatible con Pillow modernas/viejas."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def load_font(preferred_sizes):
    """
    Intenta cargar una fuente TrueType común en varios tamaños.
    preferred_sizes: lista de tamaños (ej: [95, 80, 60])
    Devuelve tuple (font_path_used, ImageFont instance for first size in list)
    """
    # posibles rutas comunes en Linux/Win/Mac
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf"
    ]
    for path in candidates:
        for size in preferred_sizes:
            try:
                font = ImageFont.truetype(path, size)
                return path, font
            except Exception:
                continue
    # fallback: font por defecto
    return None, ImageFont.load_default()

def draw_text_with_outline(draw, pos, text, font, fill, outline_fill=(255,255,255), outline_width=2):
    """
    Dibuja texto con contorno (outline) y sombra sutil.
    Usamos varios offsets para asegurar compatibilidad con versiones antiguas.
    """
    x, y = pos
    # Outline (negra o fill de outline_fill) - dibujar en offsets
    for dx in range(-outline_width, outline_width+1):
        for dy in range(-outline_width, outline_width+1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x+dx, y+dy), text, font=font, fill=outline_fill)
    # Sombra sutil abajo-derecha
    draw.text((x+2, y+2), text, font=font, fill=(60,60,60))
    # Texto principal
    draw.text((x, y), text, font=font, fill=fill)

def generar_placa_img(codigo, ciudad):
    """Genera la imagen RGB (Pillow) de la placa con detalles."""
    # Lienzo base
    img = Image.new("RGB", (WIDTH, HEIGHT), COLOR_FONDO)
    draw = ImageDraw.Draw(img)

    # Doble borde con esquinas ligeramente redondeadas (radio pequeño)
    radio = 10
    outer_thick = 6
    inner_thick = 3
    draw.rounded_rectangle([(2,2),(WIDTH-3, HEIGHT-3)], radius=radio, outline=COLOR_BORDE, width=outer_thick)
    inset = 2 + outer_thick + 1
    draw.rounded_rectangle([(inset, inset),(WIDTH-inset-1, HEIGHT-inset-1)], radius=radio-3, outline=COLOR_BORDE, width=inner_thick)

    # Óvalo "CO" a la izquierda (simula logo/filigrana mínima)
    oval_w, oval_h = 44, 30
    oval_x = 12
    oval_y = 16
    draw.ellipse([(oval_x, oval_y), (oval_x+oval_w, oval_y+oval_h)], fill=COLOR_OVALO, outline=COLOR_BORDE, width=2)
    # texto "CO" dentro del óvalo
    _, font_oval = load_font([16, 14])
    co_w, co_h = get_text_size(draw, "CO", font_oval)
    draw.text((oval_x + (oval_w-co_w)/2, oval_y + (oval_h-co_h)/2 - 1), "CO", font=font_oval, fill=COLOR_BORDE)

    # Tornillos simulados (círculos pequeños) en 4 esquinas internas
    screw_r = 6
    screw_color = (30,30,30)
    screw_positions = [
        (WIDTH - 18, 18), (WIDTH - 18, HEIGHT - 18),
        (18, HEIGHT - 18), (18, 18)
    ]
    for (cx, cy) in screw_positions:
        draw.ellipse([(cx - screw_r, cy - screw_r), (cx + screw_r, cy + screw_r)], fill=screw_color, outline=(60,60,60), width=2)

    # Cargar fuente principal con varios tamaños posibles
    font_path_used, _ = load_font([120, 100, 90, 80])
    # ahora determinaremos dinámicamente el tamaño máximo de fuente que quepa
    # empezamos con un tamaño grande y reducimos hasta que quepa en el ancho interior
    max_text_width = WIDTH - 40  # dejar márgenes laterales
    # intentar tamaños decrecientes
    for size in [120, 110, 100, 95, 90, 80, 72, 64, 56, 48, 40]:
        try:
            font_try = ImageFont.truetype(font_path_used, size) if font_path_used else ImageFont.load_default()
        except Exception:
            font_try = ImageFont.load_default()
        tw, th = get_text_size(draw, codigo, font_try)
        if tw <= max_text_width:
            font_principal = font_try
            break
    else:
        # fallback si ninguno ajusta completamente
        font_principal = font_try

    # Fuente ciudad (más pequeña)
    for size_city in [36, 32, 28, 24, 20]:
        try:
            font_city = ImageFont.truetype(font_path_used, size_city) if font_path_used else ImageFont.load_default()
        except Exception:
            font_city = ImageFont.load_default()
        cw, ch = get_text_size(draw, ciudad.upper(), font_city)
        if cw <= max_text_width:
            break

    # Posicionar y dibujar texto principal con outline
    tw, th = get_text_size(draw, codigo, font_principal)
    x_text = (WIDTH - tw) / 2
    y_text = (HEIGHT / 2 - th / 2) - 10
    # Dibujar con contorno y sombra (outline en blanco ligero para simular relieve)
    draw_text_with_outline(draw, (x_text, y_text), codigo, font_principal, fill=COLOR_TEXTO, outline_fill=(230,230,230), outline_width=2)

    # Dibujar ciudad abajo (en mayúsculas)
    ciudad_txt = ciudad.upper()
    cw, ch = get_text_size(draw, ciudad_txt, font_city)
    x_city = (WIDTH - cw) / 2
    y_city = HEIGHT - ch - 12
    draw_text_with_outline(draw, (x_city, y_city), ciudad_txt, font_city, fill=COLOR_TEXTO, outline_fill=(240,240,240), outline_width=1)

    # Añadir efecto reflectivo leve (gradiente blanco translúcido desde arriba)
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (255,255,255,0))
    ov_draw = ImageDraw.Draw(overlay)
    for i in range(0, 60):
        alpha = int(60 * (1 - i/60))  # decae
        ov_draw.rectangle([(10+i,10+i),(WIDTH-10-i, HEIGHT//2)], fill=(255,255,255,alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # Convertir a numpy BGR para OpenCV
    np_img = np.array(img)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    return bgr

def agregar_fondo_y_ruido(placa_bgr, target_shape=(300,600)):
    """Coloca la placa BGR sobre un fondo mayor y agrega ruido/reflejos."""
    target_h, target_w = target_shape
    # fondo tipo carro (gris oscuro con variación)
    base_color = np.random.randint(80,180,size=3).tolist()
    fondo = np.full((target_h, target_w, 3), base_color, dtype=np.uint8)

    h, w, _ = placa_bgr.shape
    # escalar si necesario para caber dentro del fondo con márgenes
    max_w = int(target_w * 0.7)
    max_h = int(target_h * 0.6)
    if w > max_w or h > max_h:
        escala = min(max_w / w, max_h / h)
        new_w = max(1, int(w * escala))
        new_h = max(1, int(h * escala))
        placa_bgr = cv2.resize(placa_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    x = random.randint(20, target_w - w - 20)
    y = random.randint(20, target_h - h - 20)
    fondo[y:y+h, x:x+w] = placa_bgr

    # añadir ruido gaussiano ligero y viñeteo suave
    ruido = np.random.normal(0, 12, fondo.shape).astype(np.int16)
    fondo = np.clip(fondo.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

    # viñeteo (oscurecer bordes)
    Y, X = np.ogrid[:target_h, :target_w]
    center_x, center_y = target_w/2, target_h/2
    distance = ((X - center_x)**2 + (Y - center_y)**2)
    max_dist = (center_x**2 + center_y**2)
    mask = 1 - (distance / max_dist)
    mask = np.clip(mask, 0.4, 1.0)
    for c in range(3):
        fondo[:,:,c] = (fondo[:,:,c].astype(np.float32) * mask).astype(np.uint8)

    return fondo

def generar_dataset(num_train=200, num_val=40):
    rutas = {
        "train": {"placa": os.path.join(DATA_DIR, "train", "placa"),
                  "no_placa": os.path.join(DATA_DIR, "train", "no_placa")},
        "val": {"placa": os.path.join(DATA_DIR, "val", "placa"),
                "no_placa": os.path.join(DATA_DIR, "val", "no_placa")}
    }

    # crear carpetas
    for tipo in rutas:
        for clase in rutas[tipo]:
            os.makedirs(rutas[tipo][clase], exist_ok=True)

    print("[INFO] Generando dataset sintético (placas normativas con detalles)...")

    for tipo, carpetas in rutas.items():
        n = num_train if tipo == "train" else num_val
        for i in range(n):
            codigo = generar_codigo_placa()
            ciudad = random.choice(CIUDADES)

            placa_bgr = generar_placa_img(codigo, ciudad)
            img_final = agregar_fondo_y_ruido(placa_bgr)

            ruta = os.path.join(carpetas["placa"], f"placa_{tipo}_{i}.jpg")
            cv2.imwrite(ruta, img_final)

            # imagen sin placa (solo fondo)
            fondo_solo = np.full((300, 600, 3), np.random.randint(80,220,size=3), dtype=np.uint8)
            cv2.imwrite(os.path.join(carpetas["no_placa"], f"no_placa_{tipo}_{i}.jpg"), fondo_solo)

    print("✅ Dataset sintético generado exitosamente.")
    print(f"Ubicación: {DATA_DIR}")
