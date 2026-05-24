import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Path dataset asli dan path folder hasil yang akan dibuat
base_dir = 'dataset'
output_dir = r'C:\Users\Asus\Kampus\semester4\Pengolahan Citra Digital\LBP\hasil_LBP\dataset_hasil_lbp'

# Parameter LBP
RADIUS = 1
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

print("Mulai memproses LBP dan menyimpan ke folder baru...\n")

for root, dirs, files in os.walk(base_dir):
    label = os.path.basename(root)

    if label in ['hole', 'lines','normal','stain']:
        # Membuat folder tujuan baru jika belum ada (misal: /content/dataset_hasil_lbp/hole)
        target_folder = os.path.join(output_dir, label)
        os.makedirs(target_folder, exist_ok=True)

        print(f"Menyimpan hasil LBP untuk kategori: [{label}]")

        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)

                # 1. Baca gambar asli dalam grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (256, 256))

                # 2. Hitung LBP
                lbp = local_binary_pattern(img, N_POINTS, RADIUS, METHOD)

                # 3. Normalisasi nilai LBP (0-255) agar bisa disimpan kembali sebagai file gambar .png/.jpg
                lbp_normalized = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # 4. Simpan gambar hasil LBP ke folder baru
                output_path = os.path.join(target_folder, f"lbp_{file}")
                cv2.imwrite(output_path, lbp_normalized)

print(f"\nSelesai! Semua gambar hasil LBP disimpan di folder: {output_dir}")