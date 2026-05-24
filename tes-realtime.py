import cv2
import pickle
import pandas as pd
import numpy as np

from skimage.feature import (
    graycomatrix,
    graycoprops
)

# =========================
# LOAD MODEL
# =========================

with open('model_knn_kain.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('scaler_knn_kain.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("Model berhasil dimuat!")
print("Tekan Q untuk keluar")

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    display = frame.copy()

    # =========================
    # RESIZE
    # =========================

    small = cv2.resize(
        frame,
        (320,240)
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    # =========================
    # PREPROCESS
    # =========================

    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    kernel = np.ones(
        (3,3),
        np.uint8
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.dilate(
        thresh,
        kernel,
        iterations=1
    )

    # =========================
    # FIND CONTOURS
    # =========================

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected = False
    best_conf = 0
    best_point = None
    best_rect = None

    # =========================
    # LOOP CONTOUR
    # =========================

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # filter noise
        if area < 40:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # filter terlalu besar
        if area > 5000:
            continue

        ratio = w / h

        # filter lipatan panjang
        if ratio > 5 or ratio < 0.2:
            continue

        # =========================
        # ROI
        # =========================

        roi = gray[
            y:y+h,
            x:x+w
        ]

        if roi.size == 0:
            continue

        roi = cv2.resize(
            roi,
            (128,128)
        )

        # =========================
        # GLCM
        # =========================

        glcm = graycomatrix(
            roi,
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

        asm = graycoprops(
            glcm,
            'ASM'
        )[0,0]

        mean = np.mean(roi)

        fitur = pd.DataFrame(
            [[
                mean,
                contrast,
                homogeneity,
                asm,
                energy
            ]],
            columns=[
                'GLCM_Mean',
                'GLCM_Contrast',
                'GLCM_Homogeneity',
                'GLCM_ASM',
                'GLCM_Energy'
            ]
        )

        fitur_scaled = scaler.transform(
            fitur
        )

        hasil = knn_model.predict(
            fitur_scaled
        )[0]

        conf = np.max(
            knn_model.predict_proba(
                fitur_scaled
            )
        )

        # =========================
        # JIKA RUSAK
        # =========================

        if hasil != 'normal':

            if conf > best_conf:

                best_conf = conf

                center_x = x + w // 2
                center_y = y + h // 2

                best_point = (
                    center_x,
                    center_y
                )
                
                best_rect = (
                    x, y, w, h
                )

                detected = True

    # =========================
    # DRAW RESULT
    # =========================

    if detected and best_point is not None and best_rect is not None:

        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240

        bx, by, bw, bh = best_rect

        x1 = int(bx * scale_x)
        y1 = int(by * scale_y)
        x2 = int((bx + bw) * scale_x)
        y2 = int((by + bh) * scale_y)

        # bounding box merah
        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        # tulisan
        cv2.putText(
            display,
            'RUSAK',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# RELEASE
# =========================

cap.release()
cv2.destroyAllWindows()