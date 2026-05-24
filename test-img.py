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

# =========================
# LOAD IMAGE
# =========================

image_path = 'test-stain2.jpg'

img = cv2.imread(image_path)

if img is None:
    print("Gambar gagal dibaca!")
    exit()

img = cv2.resize(
    img,
    (900,600)
)

display = img.copy()

# =========================
# PREPROCESS
# =========================

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

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
    iterations=2
)

# =========================
# FIND CONTOURS
# =========================

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# =========================
# CARI CONTOUR TERBESAR
# =========================

best_contour = None
best_area = 0

for contour in contours:

    area = cv2.contourArea(contour)

    # Filter noise kecil
    if area < 300:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    # Filter bentuk terlalu tipis/panjang
    ratio = w / h

    if ratio > 8 or ratio < 0.2:
        continue

    if area > best_area:
        best_area = area
        best_contour = contour

# =========================
# JIKA ADA CONTOUR
# =========================

if best_contour is not None:

    x, y, w, h = cv2.boundingRect(
        best_contour
    )

    padding = 20

    x = max(0, x-padding)
    y = max(0, y-padding)

    w = min(
        w + padding*2,
        img.shape[1]-x
    )

    h = min(
        h + padding*2,
        img.shape[0]-y
    )

    roi = gray[
        y:y+h,
        x:x+w
    ]

    roi = cv2.resize(
        roi,
        (256,256)
    )

    # =========================
    # GLCM FEATURE
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

    hasil_asli = knn_model.predict(
        fitur_scaled
    )[0]

    # =========================
    # NORMAL
    # =========================

    if hasil_asli == 'normal':

        cv2.putText(
            display,
            'NORMAL',
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

        print("STATUS : NORMAL")

    # =========================
    # RUSAK
    # =========================

    else:

        cv2.rectangle(
            display,
            (x,y),
            (x+w,y+h),
            (0,0,255),
            3
        )

        cv2.putText(
            display,
            'RUSAK',
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

        print("STATUS : RUSAK")

# =========================
# JIKA TIDAK ADA CONTOUR
# =========================

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

    print("STATUS : NORMAL")

# =========================
# SHOW RESULT
# =========================

cv2.imshow(
    'Deteksi Kondisi Kain',
    display
)

cv2.waitKey(0)

cv2.destroyAllWindows()