import os            
import cv2           
import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Menggunakan path folder hasil LBP yang baru saja dibuat
base_dir_lbp = r'C:\Users\Asus\Kampus\semester4\Pengolahan Citra Digital\LBP\hasil_LBP\dataset_hasil_lbp'
n_bins = 10

all_features = []
all_labels = []

print("Membaca data dari folder hasil LBP untuk persiapan KNN...\n")

# Loop membaca gambar dari folder hasil LBP
for root, dirs, files in os.walk(base_dir_lbp):
    label = os.path.basename(root)

    if label in ['hole', 'line','normal','stain']:
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)

                # Baca gambar LBP yang sudah tersimpan (format grayscale)
                img_lbp = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_lbp is None:
                    continue

                # Hitung kembali histogram dari gambar LBP tersebut sebagai representasi tekstur
                # Karena gambar sudah disimpan dan dinormalisasi (0-255), kita bagi bin-nya ke rentang skala 256 piksel
                hist, _ = np.histogram(img_lbp.ravel(), bins=n_bins, range=(0, 256), density=True)

                all_features.append(hist)
                all_labels.append(label)

# Satukan menjadi DataFrame
kolom_fitur = [f'LBP_{i}' for i in range(n_bins)]
df_dataset = pd.DataFrame(all_features, columns=kolom_fitur)
df_dataset['Jenis_Kerusakan'] = all_labels

# Split Data & Training KNN
X = df_dataset[kolom_fitur].values
y = df_dataset['Jenis_Kerusakan'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Eksekusi KNN
knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

# Menampilkan Hasil Akhir
print(f"=======================================")
print(f"AKURASI KNN SETELAH EKSPOR FOLDER: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"=======================================\n")

# Cetak Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.title('Confusion Matrix - Sumber Data Folder LBP')
plt.ylabel('Kategori Asli')
plt.xlabel('Hasil Prediksi KNN')
plt.show()