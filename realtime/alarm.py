import cv2
import pickle
import pandas as pd
import numpy as np

from skimage.feature import (
    graycomatrix,
    graycoprops
)

# LOAD MODEL
with open(
    '../results/model_knn_kain.pkl',
    'rb'
) as f:

    knn_model = pickle.load(f)

with open(
    '../results/scaler_knn_kain.pkl',
    'rb'
) as f:

    scaler = pickle.load(f)

# DETECT DEFECT
def detect_defect(frame):

    detected = False
    best_conf = 0
    best_rect = None

    # RESIZE
    small = cv2.resize(
        frame,
        (320,240)
    )

    # GRAYSCALE
    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    # BLUR
    blur = cv2.GaussianBlur(
        gray,
        (3,3),
        0
    )

    # EDGE
    edges = cv2.Canny(
        blur,
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

    # FIND CONTOURS
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # LOOP CONTOUR
    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 20:
            continue

        if area > 2500:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        ratio = w / h

        if ratio > 6 or ratio < 0.15:
            continue

        roi = gray[
            y:y+h,
            x:x+w
        ]

        if roi.size == 0:
            continue

        roi = cv2.resize(
            roi,
            (256,256)
        )

        # GLCM
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

        mean = np.mean(
            roi
        )

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

        if hasil != 'normal':

            if conf > best_conf:

                best_conf = conf

                best_rect = (
                    x,
                    y,
                    w,
                    h
                )

                detected = True

    return (
        detected,
        best_rect,
        best_conf,
        edges
    )