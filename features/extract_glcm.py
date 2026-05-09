import cv2
import os
import pandas as pd

from skimage.feature import graycomatrix, graycoprops

# folder dataset hasil preprocessing
dataset_path = "dataset_processed"

# list untuk menyimpan data
data = []

# loop tiap kelas
for label in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, label)

    # loop tiap gambar
    for filename in os.listdir(folder_path):

        img_path = os.path.join(folder_path, filename)

        # baca gambar grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # buat GLCM
        glcm = graycomatrix(
            img,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )

        # ambil fitur
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        # simpan ke list
        data.append([
            contrast,
            homogeneity,
            energy,
            correlation,
            label
        ])

# buat dataframe
df = pd.DataFrame(data, columns=[
    'contrast',
    'homogeneity',
    'energy',
    'correlation',
    'label'
])

# simpan CSV
os.makedirs("features", exist_ok=True)

df.to_csv("features/features.csv", index=False)

print("Feature extraction selesai!")