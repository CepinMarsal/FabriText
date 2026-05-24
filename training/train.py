import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATASET
# =========================

csv_path = '../results/hasil_ekstraksi_glcm_kelas.csv'

df = pd.read_csv(csv_path)

print("DATASET BERHASIL DIBACA")

print(f"Total data : {len(df)}")

# =========================
# FITUR DAN LABEL
# =========================

X = df[
    [
        'GLCM_Mean',
        'GLCM_Contrast',
        'GLCM_Homogeneity',
        'GLCM_ASM',
        'GLCM_Energy'
    ]
]

y = df['Kelas']

print()
print("FITUR DAN LABEL BERHASIL DIPISAH")

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print()
print("SPLIT DATA SELESAI")

print(f"Data training : {len(X_train)}")
print(f"Data testing  : {len(X_test)}")

# =========================
# NORMALISASI
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print()
print("NORMALISASI DATA SELESAI")

# =========================
# TRAINING KNN
# =========================

nilai_k = 5

knn_model = KNeighborsClassifier(
    n_neighbors=nilai_k
)

knn_model.fit(
    X_train_scaled,
    y_train
)

print()
print("TRAINING MODEL KNN SELESAI")

print(f"Nilai K : {nilai_k}")

# =========================
# PREDIKSI
# =========================

y_pred = knn_model.predict(
    X_test_scaled
)

print()
print("PREDIKSI DATA TESTING SELESAI")

# =========================
# AKURASI
# =========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("HASIL AKURASI MODEL")

print(
    f"Akurasi Model KNN : "
    f"{accuracy * 100:.2f}%"
)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_test,
    y_pred
)

print()
print("CONFUSION MATRIX")

print(cm)

kelas = [
    'hole',
    'lines',
    'normal',
    'stain'
]

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    linewidths=1,
    linecolor='black',
    xticklabels=kelas,
    yticklabels=kelas
)

plt.title(
    'Confusion Matrix KNN'
)

plt.xlabel(
    'Predicted Class'
)

plt.ylabel(
    'True Class'
)

# =========================
# SAVE CONFUSION MATRIX
# =========================

plt.savefig(
    '../results/confusion_matrix_knn.png'
)

plt.show()

print()
print("CONFUSION MATRIX BERHASIL DISIMPAN")

# =========================
# CLASSIFICATION REPORT
# =========================

report = classification_report(
    y_test,
    y_pred
)

print()
print("CLASSIFICATION REPORT")

print(report)

# =========================
# SAVE CLASSIFICATION REPORT
# =========================

with open(
    '../results/classification_report.txt',
    'w'
) as f:

    f.write(report)

print()
print("CLASSIFICATION REPORT BERHASIL DISIMPAN")

# =========================
# SAVE MODEL
# =========================

with open(
    '../results/model_knn_kain.pkl',
    'wb'
) as f:

    pickle.dump(
        knn_model,
        f
    )

print()
print("MODEL KNN BERHASIL DISIMPAN")

# =========================
# SAVE SCALER
# =========================

with open(
    '../results/scaler_knn_kain.pkl',
    'wb'
) as f:

    pickle.dump(
        scaler,
        f
    )

print("SCALER BERHASIL DISIMPAN")

# =========================
# INFORMASI TAMBAHAN
# =========================

print()
print("RINGKASAN HASIL TRAINING")

print(f"Jumlah Total Data  : {len(df)}")
print(f"Jumlah Training    : {len(X_train)}")
print(f"Jumlah Testing     : {len(X_test)}")
print(f"Nilai K            : {nilai_k}")
print(f"Akurasi            : {accuracy*100:.2f}%")

print()
print("File yang dihasilkan :")
print("- model_knn_kain.pkl")
print("- scaler_knn_kain.pkl")
print("- confusion_matrix_knn.png")
print("- classification_report.txt")

print()
print("PROGRAM SELESAI")