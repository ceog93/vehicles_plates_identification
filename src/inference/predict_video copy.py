# src/inference/predict_video.py
import os
import cv2
import numpy as np
import tensorflow as tf
import easyocr
from datetime import datetime
from src.config import ROOT_MODEL_DIR


# ==============================================================
# 1. Carga segura del modelo
# ==============================================================
def load_model_safe(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modelo no encontrado: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✔ Modelo cargado: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Error cargando modelo: {e}")


# ==============================================================
# 2. Buscar el modelo más reciente en /models
# ==============================================================
def find_latest_model_in_models_dir(models_dir=ROOT_MODEL_DIR):

    if not os.path.exists(models_dir):
        raise RuntimeError("❌ No existe el directorio /models")

    files = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.lower().endswith(".keras")
    ]

    if not files:
        raise RuntimeError("❌ No hay modelos .keras en /models")
    latest = max(files, key=os.path.getmtime)
    print(f"✔ Modelo más reciente detectado: {latest}")
    return latest


# ==============================================================
# 3. PREPROCESAMIENTO
# ==============================================================
def preprocess_frame(frame, target_size=(416, 416)):
    resized = cv2.resize(frame, target_size)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)


# ==============================================================
# 4. Decodificación y Desempaquetado (CORREGIDO)
# ==============================================================
def decode_predictions(preds, frame_w, frame_h, conf_threshold=0.50):
    
    # 1. Reformar las predicciones para separarlas por ancla y valor
    # Forma actual: (1, 13, 13, 18)
    # Forma nueva: (1, 13, 13, 3, 6) donde 6 = [conf, cx, cy, w, h, cls]
    
    # Asumimos GRID_SIZE=13 y NUM_ANCHORS=3 de la configuración de tu modelo
    GRID_SIZE = 13
    NUM_ANCHORS = 3
    
    # Quitamos la dimensión del batch (1,)
    preds = preds[0] 
    
    # Aseguramos la forma esperada
    if preds.shape != (GRID_SIZE, GRID_SIZE, NUM_ANCHORS * 6):
        raise ValueError(f"❌ Forma de predicción inesperada: {preds.shape}. Esperado: ({GRID_SIZE}, {GRID_SIZE}, {NUM_ANCHORS*6})")
    
    # Reorganizar el tensor para separar las anclas
    preds = tf.reshape(preds, (GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 6))
    
    # 2. Extraer valores (todas las anclas, todas las celdas)
    raw_conf = preds[..., 0]   # Confianza
    raw_bbox = preds[..., 1:5] # Coordenadas normalizadas (cx, cy, w, h)
    
    all_detections = []
    
    # 3. Recorrer celdas y anclas para aplicar umbral y coordenadas
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            for a in range(NUM_ANCHORS):
                
                conf = raw_conf[gy, gx, a].numpy()
                
                if conf < conf_threshold:
                    continue
                    
                # Extraer las coordenadas normalizadas (0..1)
                cx_norm, cy_norm, w_norm, h_norm = raw_bbox[gy, gx, a].numpy()
                
                # Desplazar por celda de cuadrícula
                cx = (cx_norm + gx) / GRID_SIZE
                cy = (cy_norm + gy) / GRID_SIZE
                w = w_norm / GRID_SIZE # Usamos la w/h normalizada de la predicción
                h = h_norm / GRID_SIZE
                
                # Convertir (cx, cy, w, h) normalizado a (x1, y1, x2, y2) en píxeles
                x1 = int((cx - w / 2) * frame_w)
                y1 = int((cy - h / 2) * frame_h)
                x2 = int((cx + w / 2) * frame_w)
                y2 = int((cy + h / 2) * frame_h)
                
                # Asegurar que las coordenadas estén dentro del marco
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_w, x2)
                y2 = min(frame_h, y2)

                all_detections.append((x1, y1, x2, y2, float(conf)))

    # Nota: Aquí se necesitaría NMS (Non-Maximum Suppression) para eliminar duplicados
    # Pero para empezar, omitiremos NMS.

    return all_detections


# ==============================================================
# 5. Validación placa 🇨🇴
# ==============================================================
def clean_and_validate_plate(text):
    if not text:
        return None

    t = text.upper().replace(" ", "").replace("-", "").replace(".", "")

    if len(t) == 6 and t[:3].isalpha() and t[3:].isdigit():
        return t

    if len(t) == 6 and t[:3].isalpha() and t[3].isdigit() and t[4].isalpha() and t[5].isdigit():
        return t

    return None


def run_ocr_on_plate(crop, reader):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, detail=0)

    if not results:
        return None

    return clean_and_validate_plate(results[0])


# ==============================================================
# 6. Procesamiento de video
# ==============================================================
def detect_plates_in_video(model, input_video, output_video,
                           nms_thresh=0.45, min_conf=0.50,
                           min_area=1400, max_area=200000,
                           allow_vertical=False, ocr_lang="en",
                           show_window=True):

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"❌ No existe {input_video}")

    reader = easyocr.Reader([ocr_lang], gpu=False)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError("❌ No se pudo abrir el video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        inp = preprocess_frame(frame)
        preds = model.predict(inp, verbose=0)
        detections = decode_predictions(preds, W, H, conf_threshold=min_conf)

        for (x1, y1, x2, y2, conf) in detections:

            area = (x2 - x1) * (y2 - y1)
            if area < min_area or area > max_area:
                continue

            crop = frame[y1:y2, x1:x2]
            plate = run_ocr_on_plate(crop, reader)

            color = (0, 255, 0) if plate else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = plate if plate else f"{conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        out.write(frame)

        if show_window:
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✔ Video procesado: {output_video}")
    return output_video


# ==============================================================
# 7. FUNCIÓN PRINCIPAL — AHORA 100% COMPATIBLE CON main.py
# ==============================================================
def run_video_detection(model, input_video, output_folder,
                        nms_thresh=0.45, min_conf=0.50,
                        min_area=1400, max_area=200000,
                        allow_vertical=False, ocr_lang="en",
                        show_window=True):

    os.makedirs(output_folder, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"det_{ts}.mp4")

    return detect_plates_in_video(
        model=model,
        input_video=input_video,
        output_video=output_path,
        nms_thresh=nms_thresh,
        min_conf=min_conf,
        min_area=min_area,
        max_area=max_area,
        allow_vertical=allow_vertical,
        ocr_lang=ocr_lang,
        show_window=show_window
    )
