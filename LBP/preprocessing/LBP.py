import os
import cv2
import numpy as np
import pandas as pd

from preprocessing import (
    preprocess_image
)

# =========================
# DATASET
# =========================

base_dir = '../dataset'

# =========================
# OUTPUT
# =========================

os.makedirs(
    '../results',
    exist_ok=True
)

output_csv = (
    '../results/hasil_ekstraksi_lbp.csv'
)

# =========================
# PARAMETER
# =========================

n_bins = 10

all_features = []
all_labels = []

print("EKSTRAKSI FITUR LBP DIMULAI\n")

# =========================
# LOOP DATASET
# =========================

for root, dirs, files in os.walk(base_dir):

    label = os.path.basename(root)

    if label in [
        'hole',
        'lines',
        'normal',
        'stain'
    ]:

        print(f"[{label}]")

        for file in files:

            if file.endswith(
                (
                    '.png',
                    '.jpg',
                    '.jpeg'
                )
            ):

                img_path = os.path.join(
                    root,
                    file
                )

                lbp = preprocess_image(
                    img_path
                )

                if lbp is None:
                    continue

                # =========================
                # HISTOGRAM
                # =========================

                hist, _ = np.histogram(
                    lbp.ravel(),
                    bins=n_bins,
                    range=(0,10),
                    density=True
                )

                all_features.append(
                    hist
                )

                all_labels.append(
                    label
                )

# =========================
# DATAFRAME
# =========================

kolom_fitur = [
    f'LBP_{i}'
    for i in range(n_bins)
]

df = pd.DataFrame(
    all_features,
    columns=kolom_fitur
)

df['Kelas'] = all_labels

# =========================
# SAVE CSV
# =========================

df.to_csv(
    output_csv,
    index=False
)

print()
print("EKSTRAKSI SELESAI")

print(
    f"CSV berhasil disimpan:\n{output_csv}"
)