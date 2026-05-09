import cv2
import os

# folder input
input_folder = "dataset"

# folder output
output_folder = "dataset_processed"

# ukuran gambar
IMG_SIZE = 200

# buat folder output kalau belum ada
os.makedirs(output_folder, exist_ok=True)

# loop tiap kelas
for label in os.listdir(input_folder):

    input_path = os.path.join(input_folder, label)
    output_path = os.path.join(output_folder, label)

    os.makedirs(output_path, exist_ok=True)

    # loop gambar
    for filename in os.listdir(input_path):

        img_path = os.path.join(input_path, filename)

        # baca gambar
        img = cv2.imread(img_path)

        # skip kalau error
        if img is None:
            continue

        # resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # simpan
        save_path = os.path.join(output_path, filename)
        cv2.imwrite(save_path, gray)

print("Preprocessing selesai!")