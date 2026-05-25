import os
import cv2
import pickle
import pandas as pd
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template

from skimage.feature import (
    graycomatrix,
    graycoprops
)

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = os.path.join('results', 'model_knn_kain.pkl')
SCALER_PATH = os.path.join('results', 'scaler_knn_kain.pkl')

knn_model = None
scaler = None

try:
    with open(MODEL_PATH, 'rb') as f:
        knn_model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Read file as numpy array
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Failed to read image"}), 400

        img = cv2.resize(img, (900, 600))
        display = img.copy()

        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 7
        )
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / float(h)
            if ratio > 8 or ratio < 0.2:
                continue
            if area > best_area:
                best_area = area
                best_contour = contour

        status = "NORMAL"

        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(w + padding * 2, img.shape[1] - x)
            h = min(h + padding * 2, img.shape[0] - y)

            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (256, 256))

            # GLCM Feature
            glcm = graycomatrix(
                roi, distances=[1], angles=[0], levels=256,
                symmetric=True, normed=True
            )
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            asm = graycoprops(glcm, 'ASM')[0, 0]
            mean = np.mean(roi)

            fitur = pd.DataFrame(
                [[mean, contrast, homogeneity, asm, energy]],
                columns=['GLCM_Mean', 'GLCM_Contrast', 'GLCM_Homogeneity', 'GLCM_ASM', 'GLCM_Energy']
            )

            fitur_scaled = scaler.transform(fitur)
            hasil_asli = knn_model.predict(fitur_scaled)[0]

            if hasil_asli == 'normal':
                cv2.putText(display, 'NORMAL', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                status = "NORMAL"
            else:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(display, 'RUSAK', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                status = "RUSAK"
        else:
            cv2.putText(display, 'NORMAL', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            status = "NORMAL"

        # Convert to base64
        _, buffer = cv2.imencode('.jpg', display)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": status,
            "image_base64": f"data:image/jpeg;base64,{img_base64}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data"}), 400

        image_data = data['image']
        # Remove 'data:image/jpeg;base64,' if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 to image
        img_bytes = base64.b64decode(image_data)
        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Mirror the frame to match what the user sees in camera
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = frame.shape[:2]

        # ROI - exactly as in realtime.py
        box_size = 100
        x1 = w // 2 - box_size // 2
        y1 = h // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]

        # PREPROCESS
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # GLCM
        glcm = graycomatrix(
            gray,
            distances=[1, 2, 3],
            angles=[0, np.pi/4, np.pi/2],
            levels=256,
            symmetric=True,
            normed=True
        )

        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        asm = graycoprops(glcm, 'ASM').mean()
        mean = np.mean(gray)

        # FEATURE
        fitur = pd.DataFrame(
            [[mean, contrast, homogeneity, asm, energy]],
            columns=['GLCM_Mean', 'GLCM_Contrast', 'GLCM_Homogeneity', 'GLCM_ASM', 'GLCM_Energy']
        )

        # SCALER
        fitur_scaled = scaler.transform(fitur)

        # PREDIKSI
        hasil = knn_model.predict(fitur_scaled)[0]
        confidence = np.max(knn_model.predict_proba(fitur_scaled))

        # EDGE CHECK
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (box_size * box_size)

        status_text = "NORMAL"

        # FINAL RESULT LOGIC
        if hasil != 'normal' and confidence > 0.55 and edge_density > 0.015:
            warna = (0, 0, 255)
            text = 'RUSAK'
            status_text = "RUSAK"
        else:
            warna = (0, 255, 0)
            text = 'NORMAL'

        # DRAW BOX
        cv2.rectangle(display, (x1, y1), (x2, y2), warna, 3)
        cv2.putText(display, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, warna, 3)

        # DEBUG INFO
        cv2.putText(display, f'Conf: {confidence:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f'Edge: {edge_density:.3f}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', display)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": status_text,
            "image_base64": f"data:image/jpeg;base64,{img_base64}",
            "confidence": f"{confidence:.2f}",
            "edge_density": f"{edge_density:.3f}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
