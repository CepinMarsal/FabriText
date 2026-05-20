import cv2
import joblib
import time
import os
import numpy as np

from preprocessing.preprocess import preprocess_image
from segmentation.fabric_segment import detect_fabric
from features.feature_utils import extract_glcm_features
from realtime.ui_utils import draw_status
from realtime.alarm import play_alarm

# =========================
# LOAD MODEL
# =========================

model = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# =========================
# OUTPUT FOLDER
# =========================

os.makedirs("output/screenshots", exist_ok=True)

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

prediction_history = []

last_alarm_time = 0

# =========================
# MAIN LOOP
# =========================

while True:

    start = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # =========================
    # ROI TENGAH
    # =========================

    roi = detect_fabric(frame)

    if roi is not None:

        original_roi = roi.copy()

        # =========================
        # PREPROCESS
        # =========================

        processed = preprocess_image(roi)

        # =========================
        # GLCM FEATURES
        # =========================

        features = extract_glcm_features(processed)

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        probabilities = model.predict_proba(features_scaled)[0]

        confidence = max(probabilities) * 100

        # =========================
        # DETECT DEFECT
        # =========================

        gray = cv2.cvtColor(original_roi, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # cari area gelap
        thresh = cv2.threshold(
            blur,
            60,
            255,
            cv2.THRESH_BINARY_INV
        )[1]

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        defect_found = False

        for cnt in contours:

            area = cv2.contourArea(cnt)

            # filter noise kecil
            if area > 40:

                x, y, w, h = cv2.boundingRect(cnt)

                ratio = w / float(h)

                # filter bentuk defect
                if 0.2 < ratio < 5:

                    defect_found = True

                    prediction = "rusak"

                    # bounding box defect
                    cv2.rectangle(
                        roi,
                        (x, y),
                        (x+w, y+h),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        roi,
                        "DEFECT",
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,0,255),
                        2
                    )

        # =========================
        # STABILIZER
        # =========================

        prediction_history.append(prediction)

        if len(prediction_history) > 10:
            prediction_history.pop(0)

        prediction = max(
            set(prediction_history),
            key=prediction_history.count
        )

        # =========================
        # DRAW STATUS
        # =========================

        draw_status(frame, prediction, confidence)

        # =========================
        # SHOW FEATURES
        # =========================

        raw = features[0]

        cv2.putText(
            frame,
            f"Contrast : {raw[0]:.2f}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Homogeneity : {raw[1]:.2f}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Energy : {raw[2]:.2f}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Correlation : {raw[3]:.2f}",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        # =========================
        # ALARM
        # =========================

        current_time = time.time()

        if prediction == "rusak":

            if current_time - last_alarm_time > 3:

                play_alarm()

                filename = f"output/screenshots/rusak_{int(current_time)}.jpg"

                cv2.imwrite(filename, frame)

                last_alarm_time = current_time

        # =========================
        # SHOW ROI
        # =========================

        cv2.imshow("Processed Fabric", roi)

    # =========================
    # FPS
    # =========================

    end = time.time()

    fps = 1 / (end - start)

    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    # =========================
    # GUIDE
    # =========================

    cv2.putText(
        frame,
        "Arahkan kain ke tengah kamera",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    # =========================
    # SHOW
    # =========================

    cv2.imshow("Fabric Defect Detection", frame)

    # =========================
    # EXIT
    # =========================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# RELEASE
# =========================

cap.release()
cv2.destroyAllWindows()