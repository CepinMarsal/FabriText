import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("features/features.csv")

X = df.drop("label", axis=1)
y = df["label"]

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL
# =========================

model = KNeighborsClassifier(
    n_neighbors=3
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print(f"Akurasi : {acc * 100:.2f}%")

# =========================
# SAVE
# =========================

joblib.dump(model, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model berhasil disimpan")