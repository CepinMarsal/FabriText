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

    h, w = frame.shape[:2]

    # =========================
    # ROI TENGAH
    # =========================

    box_size = 100

    x1 = w//2 - box_size//2
    y1 = h//2 - box_size//2

    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[
        y1:y2,
        x1:x2
    ]

    # =========================
    # PREPROCESS
    # =========================

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    gray = clahe.apply(gray)

    # Blur
    gray = cv2.GaussianBlur(
        gray,
        (3,3),
        0
    )

    # =========================
    # GLCM
    # =========================

    glcm = graycomatrix(
        gray,
        distances=[1,2,3],
        angles=[
            0,
            np.pi/4,
            np.pi/2
        ],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        'contrast'
    ).mean()

    homogeneity = graycoprops(
        glcm,
        'homogeneity'
    ).mean()

    energy = graycoprops(
        glcm,
        'energy'
    ).mean()

    asm = graycoprops(
        glcm,
        'ASM'
    ).mean()

    mean = np.mean(gray)

    # =========================
    # FEATURE
    # =========================

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

    # =========================
    # SCALER
    # =========================

    fitur_scaled = scaler.transform(
        fitur
    )

    # =========================
    # PREDIKSI
    # =========================

    hasil = knn_model.predict(
        fitur_scaled
    )[0]

    confidence = np.max(
        knn_model.predict_proba(
            fitur_scaled
        )
    )

    # =========================
    # EDGE CHECK
    # =========================

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = np.sum(
        edges > 0
    ) / (box_size * box_size)

    # =========================
    # FINAL RESULT
    # =========================

    if (
        hasil != 'normal'
        and confidence > 0.55
        and edge_density > 0.015
    ):

        warna = (0,0,255)
        text = 'RUSAK'

    else:

        warna = (0,255,0)
        text = 'NORMAL'

    # =========================
    # DRAW BOX
    # =========================

    cv2.rectangle(
        display,
        (x1,y1),
        (x2,y2),
        warna,
        3
    )

    cv2.putText(
        display,
        text,
        (x1, y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        warna,
        3
    )

    # =========================
    # DEBUG
    # =========================

    cv2.putText(
        display,
        f'Conf: {confidence:.2f}',
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        display,
        f'Edge: {edge_density:.3f}',
        (20,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    # =========================
    # SHOW
    # =========================

    cv2.imshow(
        'Realtime Deteksi Kain',
        display
    )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# =========================
# RELEASE
# =========================

cap.release()

cv2.destroyAllWindows()