import cv2
import joblib
import numpy as np

from skimage.feature import graycomatrix, graycoprops

# =========================
# LOAD MODEL & SCALER
# =========================

model = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# =========================
# FUNCTION GLCM
# =========================

def extract_glcm_features(image):

    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

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

    h, w, _ = frame.shape

    # =========================
    # ROI LEBIH KECIL
    # =========================

    x1 = int(w * 0.42)
    y1 = int(h * 0.42)

    x2 = int(w * 0.58)
    y2 = int(h * 0.58)

    roi = frame[y1:y2, x1:x2]

    # =========================
    # PREPROCESSING
    # =========================

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (200, 200))

    # blur untuk mengurangi noise
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # sharpening untuk memperjelas tekstur
    kernel = np.array([
        [0, -1, 0],
        [-1, 5,-1],
        [0, -1, 0]
    ])

    gray = cv2.filter2D(gray, -1, kernel)

    # =========================
    # FEATURE EXTRACTION
    # =========================

    features = extract_glcm_features(gray)

    # scaler
    features = scaler.transform(features)

    # =========================
    # PREDIKSI
    # =========================

    prediction = model.predict(features)[0]

    # =========================
    # WARNA BOX
    # =========================

    color = (0, 255, 0)

    if prediction == "rusak":
        color = (0, 0, 255)

    # =========================
    # TAMPILKAN FITUR
    # =========================

    raw_features = extract_glcm_features(gray)[0]

    cv2.putText(
        frame,
        f"Contrast: {raw_features[0]:.4f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Homogeneity: {raw_features[1]:.4f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Energy: {raw_features[2]:.4f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Correlation: {raw_features[3]:.4f}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    # =========================
    # BOX & LABEL
    # =========================

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        frame,
        f"Hasil: {prediction}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    # =========================
    # SHOW
    # =========================

    cv2.imshow("ROI", gray)
    cv2.imshow("Fabric Defect Detection", frame)

    # keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()