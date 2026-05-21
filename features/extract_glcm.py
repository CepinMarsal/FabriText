import os
import cv2
import pandas as pd

from preprocessing.preprocess import preprocess_image
from features.feature_utils import extract_glcm_features

dataset_path = "dataset"

data = []

labels = ["normal", "rusak"]

for label in labels:

    folder = os.path.join(dataset_path, label)

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        image = cv2.imread(path)

        if image is None:
            continue

        processed = preprocess_image(image)

        features = extract_glcm_features(processed)[0]

        data.append([
            features[0],
            features[1],
            features[2],
            features[3],
            features[4],
            features[5],
            label
        ])

columns = [
    "contrast",
    "homogeneity",
    "energy",
    "correlation",
    "dissimilarity",
    "asm",
    "label"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("features.csv", index=False)

print("Features berhasil disimpan")