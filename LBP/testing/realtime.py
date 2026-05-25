import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier


# TAHAP 1: PRE-TRAINING MODEL KNN (LANGSUNG DARI DATASET ASLI)

# Path folder dataset asli kamu
base_dir_asli = r'C:\Users\Asus\Kampus\semester4\Pengolahan Citra Digital\LBP\dataset\dataset'
n_bins = 10

# Parameter ekstraksi LBP
RADIUS = 1
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

all_features = []
all_labels = []

print("=" * 60)
print("SISTEM DETEKSI CACAT KAIN REAL-TIME (LBP + KNN)")
print("=" * 60)
print("Sedang mengekstraksi LBP dan melatih KNN dari dataset asli, mohon tunggu...\n")

# Proses pembacaan dataset ASLI untuk menghindari distorsi normalisasi & kompresi gambar
for root, dirs, files in os.walk(base_dir_asli):
    label = os.path.basename(root)
    
    # Sinkronisasi label
    if label in ['hole', 'lines', 'normal', 'stain', 'line']: 
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                
                # Baca gambar ASLI dalam grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Resize ke 256x256 agar adil
                img = cv2.resize(img, (256, 256))
                
                # [PERBAIKAN] Tambahkan efek blur LEBIH KUAT untuk mematikan tekstur benang kain
                img = cv2.GaussianBlur(img, (9, 9), 0)
                
                # Ekstraksi LBP
                lbp = local_binary_pattern(img, N_POINTS, RADIUS, METHOD)
                
                # Ekstrak histogram langsung dari nilai LBP asli (0-9)
                hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, 10), density=True)
                
                all_features.append(hist)
                all_labels.append('line' if label == 'lines' else label)

if len(all_features) == 0:
    print(f"[ERROR] Tidak ada data gambar yang terbaca di: {base_dir_asli}")
    exit()

kolom_fitur = [f'LBP_{i}' for i in range(n_bins)]
df_dataset = pd.DataFrame(all_features, columns=kolom_fitur)
df_dataset['Jenis_Kerusakan'] = all_labels

X = df_dataset[kolom_fitur].values
y = df_dataset['Jenis_Kerusakan'].values

# Inisialisasi dan Pelatihan Model KNN
knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_model.fit(X, y)

print(f"-> BERHASIL: Model KNN siap digunakan!")
print(f"-> Total data yang dipelajari: {len(X)} sampel dari kategori {np.unique(y)}.\n")



# Mengaktifkan kamera bawaan laptop (0 = Kamera Utama)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Kamera laptop tidak dapat diakses atau sedang digunakan aplikasi lain!")
    exit()

print("=" * 60)
print("KAMERA AKTIF! Posisikan kain tepat di dalam kotak tengah.")
print("Tekan tombol 'q' pada keyboard untuk keluar dari program.")
print("=" * 60)

while True:
    # Membaca frame video dari kamera
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Gagal menangkap gambar dari kamera.")
        break
        
    # Ambil kotak maksimal dari layar (misal 480x480) agar skalanya tidak terlalu "zoom-in"
    h, w, _ = frame.shape
    size = min(h, w)
    start_x, start_y = int(w/2 - size/2), int(h/2 - size/2)
    end_x, end_y = start_x + size, start_y + size
    
    # Memotong area kotak tengah yang besar
    roi = frame[start_y:end_y, start_x:end_x]
    
    # [PENTING] Resize ROI ke 256x256 agar efek kompresi ukurannya sama persis dengan dataset
    roi_resized = cv2.resize(roi, (256, 256))
    
    # Mengubah gambar potongan menjadi Grayscale
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    
    # [PERBAIKAN] Tambahkan efek blur LEBIH KUAT untuk menyamarkan noise benang kain/kamera
    roi_gray = cv2.GaussianBlur(roi_gray, (9, 9), 0)
    
    # Ekstraksi LBP Real-time
    lbp_live = local_binary_pattern(roi_gray, N_POINTS, RADIUS, METHOD)
    
    # Ekstraksi statistik histogram dari matriks LBP langsung (range 0-10) tanpa cv2.normalize
    hist_live, _ = np.histogram(lbp_live.ravel(), bins=n_bins, range=(0, 10), density=True)
    
    # Mengumpankan parameter histogram live ke KNN untuk ditebak kelasnya
    prediction = knn_model.predict([hist_live])[0]
    
    # Menghitung probabilitas/tingkat keyakinan model
    probabilities = knn_model.predict_proba([hist_live])[0]
    confidence = np.max(probabilities) * 100
    
    # Penentuan visualisasi bounding box HANYA saat ada kerusakan
    if prediction != 'normal':
        warna_indikator = (0, 0, 255)  # Merah
        
        # --- TEKNIK LOKALISASI CACAT ---
        # Karena model KNN murni hanya mengklasifikasi gambar secara keseluruhan, 
        # kita gunakan OpenCV Background Subtraction untuk mencari letak PASTI noda/garis/lubang.
        
        # 1. Estimasi background kain dengan blur sangat tebal
        bg = cv2.GaussianBlur(roi_gray, (61, 61), 0)
        
        # 2. Cari selisih warna antara serat kain asli dan background (mencari anomali)
        diff = cv2.absdiff(roi_gray, bg)
        
        # 3. Thresholding: Ambil area yang perbedaannya cukup mencolok (anomali)
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        
        # 4. Morfologi untuk menggabungkan bintik cacat yang terpecah menjadi satu kesatuan
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # 5. Cari kontur (bentuk/area) dari cacat tersebut
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cacat_ditemukan = False
        scale_factor = size / 256.0 # Faktor pengali untuk mengembalikan ukuran ke frame asli
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100: # Filter area kecil agar noise kamera tidak ikut dikotakin
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Kembalikan koordinat dari resolusi 256x256 ke resolusi frame layar Anda
                abs_x = start_x + int(x * scale_factor)
                abs_y = start_y + int(y * scale_factor)
                abs_w = int(w * scale_factor)
                abs_h = int(h * scale_factor)
                
                # Gambar kotak merah TEPAT menutupi area yang cacat
                cv2.rectangle(frame, (abs_x, abs_y), (abs_x + abs_w, abs_y + abs_h), warna_indikator, 3)
                
                if not cacat_ditemukan: # Tulis teks label pada cacat pertama saja agar rapi
                    text_display = f"RUSAK ({confidence:.0f}%)"
                    cv2.putText(frame, text_display, (abs_x, abs_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna_indikator, 2)
                    cacat_ditemukan = True
                    
        # Jika kebetulan areanya sangat tipis/tidak tertangkap kontur, peringatan tetap muncul
        if not cacat_ditemukan:
            cv2.putText(frame, f"RUSAK ({confidence:.0f}%)", (start_x + 10, start_y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, warna_indikator, 2)
    else:
        # Jika kain normal, tampilkan teks hijau di pojok kiri atas
        cv2.putText(frame, "NORMAL", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
    # Menampilkan window streaming kamera ke layar monitor
    cv2.imshow("Deteksi Kerusakan Kain Real-time (LBP + KNN)", frame)
    
    # Kondisi berhenti jika mendeteksi user menekan tombol huruf 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan resource kamera dan menutup jendela OpenCV saat selesai
cap.release()
cv2.destroyAllWindows()
print("\nKamera ditutup. Program pemrosesan selesai dengan aman.")