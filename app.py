import os
import cv2
import pickle
import pandas as pd
import numpy as np
import base64
import winsound  # Diperlukan untuk triger alarm suara Bip Windows
import traceback
from flask import Flask, request, jsonify, render_template

from skimage.feature import (
    graycomatrix,
    graycoprops
)

# Mengimpor fungsi deteksi otomatis dari sub-folder realtime Anda (sesuai skrip validasi)
from realtime.alarm import detect_defect

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# ==========================================
# KONFIGURASI PATH DAN LOAD MODEL MALAIKAT
# ==========================================
MODEL_PATH = os.path.join('results', 'model_knn_kain.pkl')
SCALER_PATH = os.path.join('results', 'scaler_knn_kain.pkl')

knn_model = None
scaler = None

try:
    with open(MODEL_PATH, 'rb') as f:
        knn_model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("=== Model & Scaler Berhasil Dimuat! ===")
except Exception as e:
    print(f"=== GAGAL MEMUAT MODEL: {e} ===")


# ==========================================
# JALUR 1: HALAMAN UTAMA (INDEX)
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')


# ==========================================
# JALUR 2: PREDIKSI VIA UNGGAH FOTO (UPLOAD)
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Membaca file gambar ke format numpy array (OpenCV)
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Failed to read image"}), 400

        img = cv2.resize(img, (900, 600))
        display = img.copy()

        # --- Tahap Preprocessing Citra ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 7
        )
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # --- Deteksi Kontur Cacat ---
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

            # --- Ekstraksi Fitur GLCM ---
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

            # Normalisasi & Prediksi Model
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

        # Mengubah hasil gambar kembali ke Base64 string teks
        _, buffer = cv2.imencode('.jpg', display)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": status,
            "image_base64": f"data:image/jpeg;base64,{img_base64}"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==========================================
# JALUR 3: PREDIKSI LIVE STREAMING (MENGGUNAKAN LOGIKA TEST-REALTIME)
# ==========================================
@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data"}), 400

        image_data = data['image']
        # Hilangkan header data:image/jpeg;base64 jika ada
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Mengubah data teks string base64 kembali menjadi matriks gambar OpenCV
        img_bytes = base64.b64decode(image_data)
        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Balikkan orientasi kamera (Mirror effect sesuai test-realtime)
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # --- EXECUTE DETECTION (Memanggil fungsi dari realtime.alarm) ---
        detected, best_rect, conf, edges = detect_defect(frame)

        status_text = "NORMAL"

        # --- LOGIKA DRAW RESULT & ALARM (Meniru persis skrip test-realtime Anda) ---
        if detected and best_rect is not None:
            bx, by, bw, bh = best_rect

            # Penskalaan koordinat dari resolusi proses biner (320x240) ke resolusi kamera web asli
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 240

            x1 = int(bx * scale_x)
            y1 = int(by * scale_y)
            x2 = int((bx + bw) * scale_x)
            y2 = int((by + bh) * scale_y)

            # Gambar Kotak Merah Dinamis & Keterangan Cacat
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(display, 'RUSAK', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display, f'Confidence: {conf:.2f}', (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            status_text = "RUSAK"

            # MEMBUNYIKAN SUARA ALARM (Bip internal Windows)
            winsound.Beep(1000, 300)
        else:
            # Jika kondisi kain aman / normal
            cv2.putText(display, 'NORMAL', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Menghitung kerapatan tepi dari citra biner 'edges' untuk visualisasi dashboard monitor
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Mengubah hasil frame OpenCV menjadi teks Base64 untuk dikirim balik ke Web Browser
        _, buffer = cv2.imencode('.jpg', display)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": status_text,
            "image_base64": f"data:image/jpeg;base64,{img_base64}",
            "confidence": f"{conf:.2f}",
            "edge_density": f"{edge_density:.3f}"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Meluncurkan Aplikasi Server Lokal
if __name__ == '__main__':
    app.run(debug=True, port=5000)