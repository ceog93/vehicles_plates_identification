import cv2
import numpy as np

def preprocess_plate(img):
    """Preprocess plate ROI for segmentation.
    - img: BGR or grayscale plate ROI (numpy array)
    Returns binarized image (uint8, 0/255)
    """
    if img is None:
        return None
    # to gray
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # resize keeping aspect but with min width for processing
    h, w = gray.shape[:2]
    if w < 120:
        scale = 120 / float(w)
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

    # Denoise while keeping edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Contrast limited adaptive histogram equalization
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    # Morphological smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Adaptive threshold (good for variable lighting)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 9)

    # Remove small noise
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    return th


def find_plate_polygon(th_img):
    """Attempt to find a 4-point polygon (plate contour) in binarized image.
    Returns None or numpy array of 4 points.
    """
    contours, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 500:
            pts = approx.reshape(4,2)
            return order_points(pts)
    return None


def order_points(pts):
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4,2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_plate(gray, pts):
    # pts: ordered 4 points
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(pts, dst)
    warp = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))
    return warp


def segment_characters(th_img, original_gray=None, min_width=8, min_height=12):
    """Segment characters from binarized plate image.
    Returns list of char images (grayscale) ordered left-to-right.
    """
    # Find contours
    contours, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    h_img, w_img = th_img.shape[:2]

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        # Filters: area, relative height, aspect ratio
        if area < 80:
            continue
        if h < min_height or w < min_width:
            continue
        # reasonable aspect ratio for character
        ar = w / float(h)
        if ar > 1.2: # likely not a single char (wide)
            continue
        # ignore very tall or small
        if h < 0.3 * h_img:
            pass

        candidates.append((x,y,w,h))

    if not candidates:
        return []

    # sort left-to-right
    candidates = sorted(candidates, key=lambda b: b[0])

    chars = []
    for (x,y,w,h) in candidates:
        pad_x = max(1, int(0.05*w))
        pad_y = max(1, int(0.05*h))
        x0 = max(0, x-pad_x)
        y0 = max(0, y-pad_y)
        x1 = min(w_img, x+w+pad_x)
        y1 = min(h_img, y+h+pad_y)
        char = th_img[y0:y1, x0:x1]
        # resize to expected 32x32
        char = cv2.resize(char, (32,32), interpolation=cv2.INTER_LINEAR)
        # normalize to 0-1 if needed later
        chars.append(char)

    return chars
