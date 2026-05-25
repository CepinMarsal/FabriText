import os
import cv2
import pickle
import pandas as pd
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier

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


# =========================
# TRAIN LBP MODEL
# =========================
LBP_DATASET_PATH = os.path.join('LBP', 'dataset')
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'
LBP_N_BINS = 10

lbp_model = None

try:
    all_features = []
    all_labels = []
    valid_labels = ['hole', 'lines', 'normal', 'stain', 'line']

    for root, dirs, files in os.walk(LBP_DATASET_PATH):
        label = os.path.basename(root)
        if label in valid_labels:
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (256, 256))
                    img = cv2.GaussianBlur(img, (9, 9), 0)
                    lbp = local_binary_pattern(img, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
                    hist, _ = np.histogram(lbp.ravel(), bins=LBP_N_BINS, range=(0, 10), density=True)
                    all_features.append(hist)
                    all_labels.append('line' if label == 'lines' else label)

    if len(all_features) > 0:
        lbp_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        lbp_model.fit(all_features, all_labels)
        print(f"LBP Model berhasil dilatih! ({len(all_features)} sampel)")
    else:
        print("LBP: Tidak ada data dataset ditemukan")
except Exception as e:
    print(f"Gagal melatih LBP model: {e}")


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


def detect_defect(frame):
    detected = False
    best_conf = 0
    best_rect = None

    small = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20 or area > 2500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / h
        if ratio > 6 or ratio < 0.15:
            continue

        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        roi = cv2.resize(roi, (256, 256))
        glcm = graycomatrix(roi, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
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
        hasil = knn_model.predict(fitur_scaled)[0]
        conf = np.max(knn_model.predict_proba(fitur_scaled))

        if hasil != 'normal' and conf > best_conf:
            best_conf = conf
            best_rect = (x, y, w, h)
            detected = True

    return detected, best_rect, best_conf, edges


@app.route('/predict_frame_fixed', methods=['POST'])
def predict_frame_fixed():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data"}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = frame.shape[:2]

        box_size = 100
        x1 = w // 2 - box_size // 2
        y1 = h // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        glcm = graycomatrix(gray, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        asm = graycoprops(glcm, 'ASM').mean()
        mean = np.mean(gray)

        fitur = pd.DataFrame(
            [[mean, contrast, homogeneity, asm, energy]],
            columns=['GLCM_Mean', 'GLCM_Contrast', 'GLCM_Homogeneity', 'GLCM_ASM', 'GLCM_Energy']
        )
        fitur_scaled = scaler.transform(fitur)
        hasil = knn_model.predict(fitur_scaled)[0]
        confidence = np.max(knn_model.predict_proba(fitur_scaled))

        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (box_size * box_size)

        if hasil != 'normal' and confidence > 0.55 and edge_density > 0.015:
            warna = (0, 0, 255)
            status_text = "RUSAK"
        else:
            warna = (0, 255, 0)
            status_text = "NORMAL"

        cv2.rectangle(display, (x1, y1), (x2, y2), warna, 3)
        cv2.putText(display, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, warna, 3)
        cv2.putText(display, f'Conf: {confidence:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', display)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": status_text,
            "image_base64": f"data:image/jpeg;base64,{img_base64}",
            "confidence": f"{confidence:.2f}"
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
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = frame.shape[:2]

        detected, best_rect, best_conf, edges = detect_defect(frame)

        if detected and best_rect is not None:
            bx, by, bw, bh = best_rect
            scale_x = w / 320
            scale_y = h / 240
            x1 = int(bx * scale_x)
            y1 = int(by * scale_y)
            x2 = int((bx + bw) * scale_x)
            y2 = int((by + bh) * scale_y)

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(display, 'RUSAK', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display, f'Confidence: {best_conf:.2f}', (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            status_text = "RUSAK"
        else:
            cv2.putText(display, 'NORMAL', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            status_text = "NORMAL"
            best_conf = 0.0

        _, buffer = cv2.imencode('.jpg', display)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": status_text,
            "image_base64": f"data:image/jpeg;base64,{img_base64}",
            "confidence": f"{best_conf:.2f}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/predict_frame_lbp', methods=['POST'])
def predict_frame_lbp():
    try:
        if lbp_model is None:
            return jsonify({"error": "LBP model belum siap"}), 500

        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data"}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        display = frame.copy()
        h, w = frame.shape[:2]
        size = min(h, w)
        start_x = int(w / 2 - size / 2)
        start_y = int(h / 2 - size / 2)
        end_x = start_x + size
        end_y = start_y + size

        roi = frame[start_y:end_y, start_x:end_x]
        roi_resized = cv2.resize(roi, (256, 256))
        roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (9, 9), 0)

        lbp_live = local_binary_pattern(roi_gray, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
        hist_live, _ = np.histogram(lbp_live.ravel(), bins=LBP_N_BINS, range=(0, 10), density=True)

        prediction = lbp_model.predict([hist_live])[0]
        confidence = np.max(lbp_model.predict_proba([hist_live])) * 100

        if prediction != 'normal':
            bg = cv2.GaussianBlur(roi_gray, (61, 61), 0)
            diff = cv2.absdiff(roi_gray, bg)
            _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            scale_factor = size / 256.0
            cacat_ditemukan = False

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    abs_x = start_x + int(x * scale_factor)
                    abs_y = start_y + int(y * scale_factor)
                    abs_w = int(bw * scale_factor)
                    abs_h = int(bh * scale_factor)
                    cv2.rectangle(display, (abs_x, abs_y), (abs_x + abs_w, abs_y + abs_h), (0, 0, 255), 3)
                    if not cacat_ditemukan:
                        cv2.putText(display, f"{prediction.upper()} ({confidence:.0f}%)",
                                    (abs_x, abs_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cacat_ditemukan = True

            if not cacat_ditemukan:
                cv2.putText(display, f"KERUSAKAN: {prediction.upper()}",
                            (start_x + 10, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            status_text = prediction.upper()
        else:
            cv2.putText(display, "Kain: NORMAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            status_text = "NORMAL"

        _, buffer = cv2.imencode('.jpg', display)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": status_text,
            "image_base64": f"data:image/jpeg;base64,{img_base64}",
            "confidence": f"{confidence:.0f}%"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
