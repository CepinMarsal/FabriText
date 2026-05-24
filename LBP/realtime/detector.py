import cv2
import pickle
import numpy as np

from skimage.feature import (
    local_binary_pattern
)

# =========================
# LOAD MODEL
# =========================

with open(
    '../models/model_knn_lbp.pkl',
    'rb'
) as f:

    knn_model = pickle.load(f)

# =========================
# PARAMETER
# =========================

RADIUS = 1

N_POINTS = 8 * RADIUS

METHOD = 'uniform'

n_bins = 10

# =========================
# DETECTOR
# =========================

def detect_defect(frame):

    detected = False
    best_rect = None
    best_conf = 0
    prediction = 'normal'

    h, w, _ = frame.shape

    size = min(h,w)

    start_x = int(w/2 - size/2)
    start_y = int(h/2 - size/2)

    end_x = start_x + size
    end_y = start_y + size

    roi = frame[
        start_y:end_y,
        start_x:end_x
    ]

    roi_resized = cv2.resize(
        roi,
        (256,256)
    )

    gray = cv2.cvtColor(
        roi_resized,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (9,9),
        0
    )

    lbp = local_binary_pattern(
        gray,
        N_POINTS,
        RADIUS,
        METHOD
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0,10),
        density=True
    )

    prediction = knn_model.predict(
        [hist]
    )[0]

    conf = np.max(
        knn_model.predict_proba(
            [hist]
        )
    )

    if prediction != 'normal':

        detected = True

        bg = cv2.GaussianBlur(
            gray,
            (61,61),
            0
        )

        diff = cv2.absdiff(
            gray,
            bg
        )

        _, thresh = cv2.threshold(
            diff,
            15,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones(
            (5,5),
            np.uint8
        )

        thresh = cv2.dilate(
            thresh,
            kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        scale_factor = size / 256.0

        for cnt in contours:

            area = cv2.contourArea(
                cnt
            )

            if area > 100:

                x, y, w, h = cv2.boundingRect(
                    cnt
                )

                abs_x = start_x + int(x * scale_factor)
                abs_y = start_y + int(y * scale_factor)

                abs_w = int(w * scale_factor)
                abs_h = int(h * scale_factor)

                best_rect = (
                    abs_x,
                    abs_y,
                    abs_w,
                    abs_h
                )

                best_conf = conf

                break

    return (
        detected,
        best_rect,
        best_conf,
        prediction
    )