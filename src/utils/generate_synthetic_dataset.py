# src/generate_dataset.py
"""
Generador de dataset sintético de placas colombianas (mejorado).
- 3 letras + logo + 3 números (logo placeholder o imagen).
- Letras y números escalados por separado para ajustarse al ancho disponible.
- Perforaciones alargadas (ranuras) simuladas en puntos de fijación.
- Doble borde, fondo amarillo, contorno y ligero efecto reflectivo.
Compatible con Pillow moderno (textbbox) y versiones antiguas.
"""

import os
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.config import DATA_DIR, IMG_SIZE, LOGO_PATH

# Parámetros visuales
COLOR_FONDO = (255, 211, 0)   # amarillo reflectivo aproximado (RGB)
COLOR_TEXTO = (0, 0, 0)
COLOR_BORDE = (0, 0, 0)

# Tamaño placa (px) — mantiene proporción aproximada 330x160 mm -> 335x170 px
WIDTH, HEIGHT = 335, 170

# Lista de ciudades (ejemplo)
CIUDADES = [
    "BOGOTA D.C.", "MEDELLIN", "CALI", "BARRANQUILLA", "CARTAGENA",
    "BUCARAMANGA", "PEREIRA", "MANIZALES", "CUCUTA", "SANTA MARTA",
    "IBAGUE", "NEIVA", "VILLAVICENCIO", "TUNJA", "POPAYAN", "SINCELEJO",
    "MONTERIA", "ARMENIA", "PASTO", "VALLEDUPAR", "RIOHACHA", "SOLEDAD",
    "PALMIRA", "ITAGUI", "RIONEGRO", "ENVIGADO", "SOACHA"
]

# Ruta opcional a un archivo PNG del logo oficial (si lo tienes, pon la ruta aquí)
#LOGO_PATH = "/images/logo_placa.png"

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

def load_font(paths_sizes=[(os.path.expanduser("~/.fonts/fe-font/FE-FONT.TTF"), 120,),
                           ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 120),
                           ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 120),
                           ("C:\\Windows\\Fonts\\arialbd.ttf", 120)]):
    for path, size in paths_sizes:
        try:
            font = ImageFont.truetype(path, size)
            print(f"[INFO] Fuente cargada desde {path}.")
            return path
        except Exception:
            print(f"[WARN] No se pudo cargar la fuente desde {path}.")
            continue
    return None

def draw_text_with_outline(draw, pos, text, font, fill, outline_fill=(230,230,230), outline_width=2):
    x, y = pos
    for dx in range(-outline_width, outline_width+1):
        for dy in range(-outline_width, outline_width+1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x+dx, y+dy), text, font=font, fill=outline_fill)
    # ligera sombra
    draw.text((x+2, y+2), text, font=font, fill=(60,60,60))
    draw.text((x, y), text, font=font, fill=fill)
    
def draw_stretched_text(base_image, position, text, font, fill_color, target_height):
    """
    Dibuja un texto en una imagen base, estirándolo verticalmente para alcanzar
    una altura objetivo sin cambiar su ancho.
    """
    # 1. Crear una imagen temporal para renderizar el texto
    # Usamos un tamaño grande para asegurar que el renderizado sea de alta calidad.
    temp_draw = ImageDraw.Draw(Image.new("RGBA", (2000, 2000))) 
    
    # 2. Medir el texto con la fuente original
    #current_width, current_height = font.getsize(text)
    
    bbox = font.getbbox(text)
    # bbox es (left, top, right, bottom)
    current_width = bbox[2] - bbox[0]
    current_height = bbox[3] - bbox[1]
    
    # 3. Renderizar el texto en la imagen temporal
    temp_image = Image.new("RGBA", (current_width, current_height), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp_image)
    temp_draw.text((0, 0), text, font=font, fill=fill_color)
    
    # 4. Calcular el factor de estiramiento y la nueva altura/ancho
    if current_height == 0:
        return
        
    stretch_factor = target_height / current_height
    new_width = int(current_width) # Mismo ancho
    new_height = int(target_height) # Nueva altura
    
    # 5. Redimensionar/Estirar la imagen del texto
    # Usamos Image.LANCZOS para mejor calidad
    stretched_text = temp_image.resize((new_width, new_height), Image.LANCZOS)
    
    # 6. Pegar la imagen estirada en la imagen base (la placa)
    base_image.paste(stretched_text, (int(position[0]), int(position[1])), stretched_text)

def generar_placa_img(codigo, ciudad):
    """
    Genera placa BGR (OpenCV) con:
    - código: 'AAA000' (3 letras + 3 números)
    - ciudad: texto bajo la placa
    """
    # Preparar canvas (Pillow)
    img = Image.new("RGB", (WIDTH, HEIGHT), COLOR_FONDO)
    draw = ImageDraw.Draw(img)

    # Marco doble (esquinas levemente rectas)
    radio = 8
    outer_w = 6
    inner_w = 3
    draw.rounded_rectangle([(2,2),(WIDTH-3, HEIGHT-3)], radius=radio, outline=COLOR_BORDE, width=outer_w)
    inset = 2 + outer_w + 1
    #draw.rounded_rectangle([(inset, inset),(WIDTH-inset-1, HEIGHT-inset-1)], radius=max(2, radio-3), outline=COLOR_BORDE, width=inner_w)

    # ################################################# PERFOARACIONES PLACA  #################################################
    
    # Perforaciones alargadas (ranuras) en zonas de fijación: simulamos 4 ranuras
    # Cada ranura: rounded rectangle (alargada), orientada horizontalmente en parte superior,
    # y en parte inferior también (puedes cambiar orientación).
    def draw_slot(center_x, center_y, w_slot, h_slot, radius=6, fill=(230,230,230), outline=(60,60,60)):
        x0, y0 = center_x - w_slot//2, center_y - h_slot//2
        x1, y1 = center_x + w_slot//2, center_y + h_slot//2
        draw.rounded_rectangle([(x0,y0),(x1,y1)], radius=radius, fill=fill, outline=outline, width=2)

    # ubicaciones (algo internas, a 18 px del borde)
    slots = [
        (int(WIDTH * 0.12), 20),               # superior izquierda
        (int(WIDTH * 0.88), 20),               # superior derecha
        (int(WIDTH * 0.12), HEIGHT - 20),      # inferior izquierda
        (int(WIDTH * 0.88), HEIGHT - 20)       # inferior derecha
    ]
    for (cx, cy) in slots:
        draw_slot(cx, cy, w_slot=20, h_slot=10, radius=4)

    # ################################################# CONFIGURACION PLACA  #################################################
    
    # Preparar fuentes (buscamos una existente)
    font_path = load_font()
    # Reservar espacio para logo central
    logo_w = 15 # tamaño base logo
    logo_gap = logo_w + 3  # espacio total reservado entre los grupos
    interior_width = WIDTH - 40  # margen lateral
    # width disponible para cada grupo (letras/números)
    group_max_w = (interior_width - logo_gap) / 2

    # ################################################# TEXTO LETRAS Y NUMEROS PRINCIPAL  #################################################
    
    # Elegir tamaño para letras (queremos que la altura ocupe ~65-72% del interior)
    # probaremos tamaños decrecientes hasta que el width de grupo encaje
    chosen_font_left = None
    chosen_font_right = None
    for size in [74]: #[140, 130, 120, 110, 100, 90, 80, 72, 64]:
        try:
            font_try = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        except Exception:
            font_try = ImageFont.load_default()
        # left: 3 letters (we measure as a string e.g. 'AAA')
        left_tw, left_th = get_text_size(draw, codigo[:3], font_try)
        right_tw, right_th = get_text_size(draw, codigo[3:], font_try)
        # scale down if either width exceeds group_max_w
        if left_tw <= group_max_w and right_tw <= group_max_w and left_th <= (HEIGHT * 0.75):
            chosen_font_left = font_try
            chosen_font_right = font_try
            break
    if chosen_font_left is None:
        # fallback: use final tried font
        chosen_font_left = font_try
        chosen_font_right = font_try

    # posición horizontal: left group start, logo center, right group start
    left_area_x0 = 20
    left_area_x1 = left_area_x0 + group_max_w
    right_area_x1 = WIDTH - 20
    right_area_x0 = right_area_x1 - group_max_w

    # medir texto definitivamente
    left_w, left_h = get_text_size(draw, codigo[:3], chosen_font_left)
    right_w, right_h = get_text_size(draw, codigo[3:], chosen_font_right)

    # Altura objetivo para el estiramiento (ej. 50% de la altura total)
    TARGET_TEXT_HEIGHT = int(HEIGHT * 0.50)
    DESPLAZAMIENTO_ARRIBA = 20
    
    # centrar verticalmente (ocupando la mitad-superior de la placa)
    #y_text = (HEIGHT * 0.5) - (TARGET_TEXT_HEIGHT / 2) - DESPLAZAMIENTO_ARRIBA
    y_text = (HEIGHT * 0.5) - max(left_h, right_h)/2 - 8 # 6 px de ajuste fino hacia arriba

    # left text center within left area
    x_left = left_area_x0 + (group_max_w - left_w)/2
    # right text center within right area
    x_right = right_area_x0 + (group_max_w - right_w)/2

    # -------------------------------------------------------------
    # Altura objetivo para el estiramiento (ej. 70% de la altura total)
    TARGET_TEXT_HEIGHT = int(HEIGHT * 0.40)
    
    # dibujar contorno + texto con sombra
    #draw_text_with_outline(draw, (x_left, y_text), codigo[:3], chosen_font_left, fill=COLOR_TEXTO, outline_fill=(240,240,240), outline_width=2)
    #draw_text_with_outline(draw, (x_right, y_text), codigo[3:], chosen_font_right, fill=COLOR_TEXTO, outline_fill=(240,240,240), outline_width=2)

    # dibujar contorno + texto sin sombra
    
    draw.text((x_left, y_text), codigo[:3], font=chosen_font_left, fill=COLOR_TEXTO)
    draw.text((x_right, y_text), codigo[3:], font=chosen_font_right, fill=COLOR_TEXTO)
    
    # MODIFICACIÓN: Usar la función de estiramiento
    
    #draw_stretched_text(img, (x_left, y_text), codigo[:3], chosen_font_left, COLOR_TEXTO, TARGET_TEXT_HEIGHT)
    #draw_stretched_text(img, (x_right, y_text), codigo[3:], chosen_font_right, COLOR_TEXTO, TARGET_TEXT_HEIGHT)
    
    # ################################################# LOGO PLACA  #################################################
    
    # Dibujar placeholder de logo central (si LOGO_PATH existe, pegar imagen; si no, dibujar un emblema circular)
    logo_center_x = (left_area_x1 + right_area_x0) / 2
    logo_center_y = HEIGHT * 0.5
    
    #reduction_factor = 0.65  # O usa cualquier otro factor (e.g., 0.75 para un 25% menos)
    #logo_w = int(logo_w * reduction_factor)    
    
    if LOGO_PATH and os.path.exists(LOGO_PATH):
        try:
            logo = Image.open(LOGO_PATH).convert("RGBA")
            logo = logo.resize((logo_w, logo_w), Image.LANCZOS)
            img.paste(logo, (int(logo_center_x - logo_w/2), int(logo_center_y - logo_w/2)), logo)
        except Exception:
            # fallback to drawn emblem
            draw.ellipse([(logo_center_x - logo_w/2, logo_center_y - logo_w/2),
                          (logo_center_x + logo_w/2, logo_center_y + logo_w/2)], outline=COLOR_TEXTO, width=2, fill=(230,230,230))
            draw.text((logo_center_x - 6, logo_center_y - 8), "M", fill=COLOR_TEXTO, font=ImageFont.load_default())
    else:
        # emblem placeholder: small filled circle with inner ring (simula logo)
        draw.ellipse([(logo_center_x - logo_w/2, logo_center_y - logo_w/2),
                      (logo_center_x + logo_w/2, logo_center_y + logo_w/2)], fill=(230,230,230), outline=COLOR_TEXTO, width=2)
        # letra dentro
        f_small = ImageFont.load_default()
        tw, th = get_text_size(draw, "M", f_small)
        draw.text((logo_center_x - tw/2, logo_center_y - th/2), "M", font=f_small, fill=COLOR_TEXTO)

    # ################################################# TEXTO CIUDAD PLACA  #################################################
    
    # Texto ciudad (más pequeño, centrado y cerca del borde inferior)
    city_txt = ciudad.upper()
    # elegir tamaño ciudad proporcional
    for size_city in [22]:#[36, 32, 30, 28, 26, 24]:
        try:
            font_city = ImageFont.truetype(font_path, size_city) if font_path else ImageFont.load_default()
        except Exception:
            font_city = ImageFont.load_default()
        cw, ch = get_text_size(draw, city_txt, font_city)
        if cw <= (WIDTH - 60):
            break
    # posición ciudad: más cerca del borde inferior (pero sin tocar)
    x_city = (WIDTH - cw) / 2
    y_city = HEIGHT - ch - 18
    # ciudad con sombra
    #draw_text_with_outline(draw, (x_city, y_city), city_txt, font_city, fill=COLOR_TEXTO, outline_fill=(240,240,240), outline_width=1)

    #Ciudad sin sombra
    draw.text((x_city, y_city), city_txt, font=font_city, fill=COLOR_TEXTO)
    
    # ligero efecto reflectivo (overlay blanco translúcido)
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (255,255,255,0))
    ov_draw = ImageDraw.Draw(overlay)
    for i in range(0, 50, 2):
        alpha = int(45 * (1 - i/50))
        ov_draw.rectangle([(10+i, 10+i), (WIDTH-10-i, HEIGHT//2)], fill=(255,255,255, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # convertir a numpy BGR para OpenCV
    np_img = np.array(img)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return bgr

def agregar_fondo_y_ruido(placa_bgr, target_shape=(300,600)):
    target_h, target_w = target_shape
    base_color = np.random.randint(80,180,size=3).tolist()
    fondo = np.full((target_h, target_w, 3), base_color, dtype=np.uint8)

    h, w = placa_bgr.shape[:2]
    max_w = int(target_w * 0.7)
    max_h = int(target_h * 0.6)
    if w > max_w or h > max_h:
        escala = min(max_w / w, max_h / h)
        placa_bgr = cv2.resize(placa_bgr, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)
        h, w = placa_bgr.shape[:2]

    x = random.randint(20, target_w - w - 20)
    y = random.randint(20, target_h - h - 20)
    fondo[y:y+h, x:x+w] = placa_bgr

    ruido = np.random.normal(0, 12, fondo.shape).astype(np.int16)
    fondo = np.clip(fondo.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

    # viñeteo
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

    for tipo in rutas:
        for clase in rutas[tipo]:
            os.makedirs(rutas[tipo][clase], exist_ok=True)

    print("[INFO] Generando dataset sintético (placas con logo y ranuras)...")

    for tipo, carpetas in rutas.items():
        n = num_train if tipo == "train" else num_val
        for i in range(n):
            codigo = generar_codigo_placa()
            ciudad = random.choice(CIUDADES)
            placa_bgr = generar_placa_img(codigo, ciudad)
            img_final = agregar_fondo_y_ruido(placa_bgr)
            ruta = os.path.join(carpetas["placa"], f"placa_{tipo}_{i}.jpg")
            cv2.imwrite(ruta, img_final)
            # fondo sin placa
            fondo_solo = np.full((300, 600, 3), np.random.randint(80,220,size=3), dtype=np.uint8)
            cv2.imwrite(os.path.join(carpetas["no_placa"], f"no_placa_{tipo}_{i}.jpg"), fondo_solo)

    print("✅ Dataset sintético generado.")
    print(f"Ubicación: {DATA_DIR}")
