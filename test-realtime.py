import cv2
import pandas as pd
import numpy as np

from skimage.feature import (
    graycomatrix,
    graycoprops
)

print("Realtime Deteksi Kain")
print("Tekan Q untuk keluar")

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

# =========================
# LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # mirror webcam
    frame = cv2.flip(
        frame,
        1
    )

    display = frame.copy()

    # =========================
    # RESIZE BIAR RINGAN
    # =========================

    small = cv2.resize(
        frame,
        (320,240)
    )

    # =========================
    # GRAYSCALE
    # =========================

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    # =========================
    # PREPROCESS
    # =========================

    blur = cv2.GaussianBlur(
        gray,
        (3,3),
        0
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(
        blur
    )

    # =========================
    # EDGE DETECTION
    # =========================

    edges = cv2.Canny(
        enhanced,
        40,
        120
    )

    kernel = np.ones(
        (3,3),
        np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    # =========================
    # FIND CONTOUR
    # =========================

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected = False
    best_score = 0
    best_rect = None

    # =========================
    # LOOP CONTOUR
    # =========================

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # =========================
        # FILTER NOISE
        # =========================

        if area < 20:
            continue

        if area > 2500:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        ratio = w / h

        # skip garis panjang
        if ratio > 6 or ratio < 0.15:
            continue

        # =========================
        # PADDING
        # =========================

        padding = 15

        x1 = max(
            0,
            x-padding
        )

        y1 = max(
            0,
            y-padding
        )

        x2 = min(
            gray.shape[1],
            x+w+padding
        )

        y2 = min(
            gray.shape[0],
            y+h+padding
        )

        # =========================
        # ROI
        # =========================

        roi = gray[
            y1:y2,
            x1:x2
        ]

        if roi.size == 0:
            continue

        roi = cv2.resize(
            roi,
            (128,128)
        )

        # =========================
        # EXTRA PREPROCESS
        # =========================

        roi_blur = cv2.GaussianBlur(
            roi,
            (3,3),
            0
        )

        # =========================
        # GLCM
        # =========================

        glcm = graycomatrix(
            roi_blur,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )

        contrast = graycoprops(
            glcm,
            'contrast'
        )[0,0]

        homogeneity = graycoprops(
            glcm,
            'homogeneity'
        )[0,0]

        energy = graycoprops(
            glcm,
            'energy'
        )[0,0]

        # =========================
        # SIMPLE ANOMALY SCORE
        # =========================

        score = (
            contrast * 0.6
            + (1 - homogeneity) * 0.3
            + (1 - energy) * 0.1
        )

        # =========================
        # DETECT ANOMALY
        # =========================

        if score > 2.5:

            if score > best_score:

                best_score = score

                best_rect = (
                    x,
                    y,
                    w,
                    h
                )

                detected = True

    # =========================
    # DRAW RESULT
    # =========================

    if detected and best_rect is not None:

        bx, by, bw, bh = best_rect

        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240

        x1 = int(
            bx * scale_x
        )

        y1 = int(
            by * scale_y
        )

        x2 = int(
            (bx+bw) * scale_x
        )

        y2 = int(
            (by+bh) * scale_y
        )

        # =========================
        # BOUNDING BOX
        # =========================

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0,0,255),
            3
        )

        # =========================
        # LABEL
        # =========================

        cv2.putText(
            display,
            'RUSAK',
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

    else:

        cv2.putText(
            display,
            'NORMAL',
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

    # =========================
    # SHOW
    # =========================

    cv2.imshow(
        'Realtime Deteksi Kain',
        display
    )

    cv2.imshow(
        'Edges',
        edges
    )

    # =========================
    # EXIT
    # =========================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# RELEASE
# =========================

cap.release()
cv2.destroyAllWindows()