import cv2

# PREPROCESS IMAGE
def preprocess_image(img):
    # GRAYSCALE
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # RESIZE
    gray = cv2.resize(
        gray,
        (256,256)
    )

    return gray