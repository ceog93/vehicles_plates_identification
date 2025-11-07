# src/infer:video.py
'''
Inerencia sobre video
'''

# src/infer_video.py

import cv2
import numpy as np
import tensorflow as tf
from .config import MODEL_PATH, IMG_SIZE, THRESHOLD

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_frame(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_frame(frame):
    pred = model.predict(preprocess_frame(frame))[0][0]
    return pred > THRESHOLD, pred

def run_video_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected, confidence = predict_frame(frame)
        label = f"Placa: {'SI' if detected else 'NO'} ({confidence:.2f})"
        color = (0, 255, 0) if detected else (0, 0, 255)

        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Detección de Placas", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
