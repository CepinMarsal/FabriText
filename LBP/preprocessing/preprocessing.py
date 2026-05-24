import cv2
import numpy as np

from skimage.feature import (
    local_binary_pattern
)

# =========================
# PARAMETER LBP
# =========================

RADIUS = 1

N_POINTS = 8 * RADIUS

METHOD = 'uniform'

# =========================
# PREPROCESS IMAGE
# =========================

def preprocess_image(img_path):

    img = cv2.imread(
        img_path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        return None

    img = cv2.resize(
        img,
        (256,256)
    )

    # blur untuk mengurangi noise kain

    img = cv2.GaussianBlur(
        img,
        (9,9),
        0
    )

    # ekstraksi lbp

    lbp = local_binary_pattern(
        img,
        N_POINTS,
        RADIUS,
        METHOD
    )

    return lbp