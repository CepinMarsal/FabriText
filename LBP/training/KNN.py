import os
import time
import pickle
import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from sklearn.neighbors import (
    KNeighborsClassifier
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# FOLDER
# =========================

os.makedirs(
    '../models',
    exist_ok=True
)

os.makedirs(
    '../results',
    exist_ok=True
)

# =========================
# LOAD CSV
# =========================

csv_path = (
    '../results/hasil_ekstraksi_lbp.csv'
)

df = pd.read_csv(
    csv_path
)

print("DATASET BERHASIL DIBACA")

# =========================
# FITUR
# =========================

kolom_fitur = [
    col for col in df.columns
    if col.startswith('LBP_')
]

X = df[kolom_fitur].values

y = df['Kelas'].values

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# TRAINING
# =========================

start = time.time()

knn_model = KNeighborsClassifier(
    n_neighbors=3,
    metric='euclidean'
)

knn_model.fit(
    X_train,
    y_train
)

end = time.time()

training_time = end - start

# =========================
# PREDICT
# =========================

y_pred = knn_model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print(
    f"AKURASI : "
    f"{accuracy*100:.2f}%"
)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(
    figsize=(6,5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=knn_model.classes_,
    yticklabels=knn_model.classes_
)

plt.title(
    'Confusion Matrix LBP + KNN'
)

plt.savefig(
    '../results/confusion_matrix_knn.png'
)

plt.show()

# =========================
# REPORT
# =========================

report = classification_report(
    y_test,
    y_pred
)

with open(
    '../results/classification_report.txt',
    'w'
) as f:

    f.write(
        "HASIL EVALUASI LBP + KNN\n\n"
    )

    f.write(
        f"Akurasi : "
        f"{accuracy*100:.2f}%\n"
    )

    f.write(
        f"Waktu Training : "
        f"{training_time:.4f} detik\n\n"
    )

    f.write(report)

print()
print(
    "REPORT BERHASIL DISIMPAN"
)

# =========================
# SAVE MODEL
# =========================

with open(
    '../models/model_knn_lbp.pkl',
    'wb'
) as f:

    pickle.dump(
        knn_model,
        f
    )

print(
    "MODEL BERHASIL DISIMPAN"
)