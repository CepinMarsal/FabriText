import cv2
import numpy as np

def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (200, 200))

    gray = cv2.GaussianBlur(gray, (5,5), 0)

    kernel = np.array([
        [0, -1, 0],
        [-1, 5,-1],
        [0, -1, 0]
    ])

    gray = cv2.filter2D(gray, -1, kernel)

    return gray