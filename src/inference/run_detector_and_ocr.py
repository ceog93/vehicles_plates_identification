import os
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from src.config import MODEL_PATH, IMG_SIZE
from src.inference.char_segmentation import preprocess_plate, find_plate_polygon, warp_plate, segment_characters


def load_detector(detector_path=None):
    # try load Keras model (user model path in config)
    if detector_path is None:
        detector_path = MODEL_PATH
    if not os.path.exists(detector_path):
        raise FileNotFoundError(f"Detector model not found: {detector_path}")
    model = load_model(detector_path, compile=False)
    return model


def load_ocr(ocr_path):
    if not os.path.exists(ocr_path):
        raise FileNotFoundError(f"OCR model not found: {ocr_path}")
    return load_model(ocr_path)


def detect_boxes(detector, frame, input_size=(224,224)):
    # Simple wrapper: resize and predict then map back to original
    # Assumes detector returns normalized bbox [xmin,ymin,xmax,ymax]
    h0, w0 = frame.shape[:2]
    im = cv2.resize(frame, input_size)
    x = im.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    out = detector.predict(x)
    # format output handling: allow shapes (1,4) or (N,4)
    if out is None:
        return []
    out = np.array(out)
    if out.ndim == 2 and out.shape[0] == 1:
        boxes = out
    else:
        boxes = out

    # map normalized to pixel coords
    boxes_px = []
    for b in boxes:
        try:
            xmin = int(b[0] * w0)
            ymin = int(b[1] * h0)
            xmax = int(b[2] * w0)
            ymax = int(b[3] * h0)
            boxes_px.append((xmin, ymin, xmax, ymax))
        except Exception:
            continue
    return boxes_px


def ocr_predict(ocr_model, char_images, classes_map=None):
    if not char_images:
        return ""
    X = np.stack([img.astype(np.float32)/255.0 for img in char_images], axis=0)
    X = np.expand_dims(X, -1)  # (N,32,32,1)
    preds = ocr_model.predict(X)
    labels = np.argmax(preds, axis=1)
    if classes_map:
        inv_map = {v:k for k,v in classes_map.items()}
        chars = [inv_map.get(int(l), '?') for l in labels]
    else:
        # default mapping 0-9,A-Z
        charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chars = [charset[int(l)] if int(l) < len(charset) else '?' for l in labels]
    return "".join(chars)


def process_video(detector, ocr_model, input_path, output_path, show=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detect_boxes(detector, frame, input_size=(IMG_SIZE[0], IMG_SIZE[1]))
        for (xmin,ymin,xmax,ymax) in boxes:
            # expand bbox a little
            pad_w = int((xmax-xmin)*0.05)
            pad_h = int((ymax-ymin)*0.08)
            x0 = max(0, xmin - pad_w)
            y0 = max(0, ymin - pad_h)
            x1 = min(frame.shape[1], xmax + pad_w)
            y1 = min(frame.shape[0], ymax + pad_h)
            roi = frame[y0:y1, x0:x1]

            # preprocess and try polygon warp
            th = preprocess_plate(roi)
            poly = None
            try:
                poly = find_plate_polygon(th)
            except Exception:
                poly = None

            if poly is not None:
                warped = warp_plate(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), poly)
                th2 = preprocess_plate(warped)
                chars = segment_characters(th2)
            else:
                chars = segment_characters(th)

            text = ocr_predict(ocr_model, chars)

            # draw
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
            cv2.putText(frame, text, (x0, max(10,y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        out_vid.write(frame)
        if show:
            cv2.imshow('ocr', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_idx += 1

    cap.release()
    out_vid.release()
    if show:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=False)
    parser.add_argument('--ocr', default='saved_models/ocr_model.h5')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    input_path = args.input
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join('03_production','output_results', f"{base}_ocr.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    detector = load_detector()
    ocr = load_ocr(args.ocr)
    process_video(detector, ocr, input_path, output_path, show=args.show)
    print('Output saved to', output_path)


if __name__ == '__main__':
    main()
