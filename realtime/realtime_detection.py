import cv2
import joblib
import numpy as np

from skimage.feature import graycomatrix, graycoprops

# =========================
# LOAD MODEL
# =========================

model = joblib.load("model/knn_model.pkl")

# =========================
# FUNCTION GLCM
# =========================

def extract_glcm_features(image):

    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return [[
        contrast,
        homogeneity,
        energy,
        correlation
    ]]

# =========================
# START CAMERA
# =========================

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ukuran frame
    h, w, _ = frame.shape

    # ROI tengah
    x1 = int(w * 0.3)
    y1 = int(h * 0.3)

    x2 = int(w * 0.7)
    y2 = int(h * 0.7)

    roi = frame[y1:y2, x1:x2]

    # preprocessing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 200))

    # ekstraksi fitur
    features = extract_glcm_features(gray)

    # prediksi
    prediction = model.predict(features)[0]

    # warna box
    color = (0, 255, 0)

    if prediction == "rusak":
        color = (0, 0, 255)

    # tampilkan box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # tampilkan label
    cv2.putText(
        frame,
        f"Kondisi: {prediction}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    # tampilkan ROI
    cv2.imshow("ROI", gray)

    # tampilkan kamera
    cv2.imshow("Fabric Defect Detection", frame)

    # keluar tekan q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()