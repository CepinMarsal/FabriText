import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("features/features.csv")

# fitur
X = df[['contrast', 'homogeneity', 'energy', 'correlation']]

# label
y = df['label']

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# STANDARD SCALER
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# TRAIN KNN
# =========================

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

# =========================
# PREDIKSI
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUASI
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\n===== HASIL EVALUASI =====")
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# SIMPAN MODEL & SCALER
# =========================

joblib.dump(model, "model/knn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel dan scaler berhasil disimpan!")