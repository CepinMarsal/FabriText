import numpy as np
import cv2
import os
import pandas as pd

# FAST GLCM
def fast_glcm(
    img,
    vmin=0,
    vmax=255,
    levels=8,
    kernel_size=5,
    distance=1.0,
    angle=0.0
):

    mi, ma = vmin, vmax
    ks = kernel_size

    h, w = img.shape

    # QUANTIZATION
    bins = np.linspace(
        mi,
        ma + 1,
        levels + 1
    )

    gl1 = np.digitize(
        img,
        bins
    ) - 1

    # SHIFT IMAGE
    dx = distance * np.cos(
        np.deg2rad(angle)
    )

    dy = distance * np.sin(
        np.deg2rad(-angle)
    )

    mat = np.array([
        [1.0, 0.0, -dx],
        [0.0, 1.0, -dy]
    ], dtype=np.float32)

    gl2 = cv2.warpAffine(
        gl1,
        mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE
    )

    # MEMBUAT GLCM
    glcm = np.zeros(
        (levels, levels, h, w),
        dtype=np.uint8
    )

    for i in range(levels):
        for j in range(levels):

            mask = (
                (gl1 == i) &
                (gl2 == j)
            )

            glcm[i, j, mask] = 1
            
    # FILTERING
    kernel = np.ones(
        (ks, ks),
        dtype=np.uint8
    )

    for i in range(levels):
        for j in range(levels):

            glcm[i, j] = cv2.filter2D(
                glcm[i, j],
                -1,
                kernel
            )

    return glcm.astype(np.float32)

# GLCM MEAN
def fast_glcm_mean(
    img,
    vmin=0,
    vmax=255,
    levels=8,
    ks=5,
    distance=1.0,
    angle=0.0
):

    h, w = img.shape

    glcm = fast_glcm(
        img,
        vmin,
        vmax,
        levels,
        ks,
        distance,
        angle
    )

    mean = np.zeros(
        (h, w),
        dtype=np.float32
    )

    for i in range(levels):
        for j in range(levels):

            mean += (
                glcm[i, j] * i / (levels)**2
            )

    return mean

# GLCM CONTRAST
def fast_glcm_contrast(
    img,
    vmin=0,
    vmax=255,
    levels=8,
    ks=5,
    distance=1.0,
    angle=0.0
):

    h, w = img.shape

    glcm = fast_glcm(
        img,
        vmin,
        vmax,
        levels,
        ks,
        distance,
        angle
    )

    contrast = np.zeros(
        (h, w),
        dtype=np.float32
    )

    for i in range(levels):
        for j in range(levels):

            contrast += (
                glcm[i, j] *
                (i - j) ** 2
            )

    return contrast

# GLCM HOMOGENEITY
def fast_glcm_homogeneity(
    img,
    vmin=0,
    vmax=255,
    levels=8,
    ks=5,
    distance=1.0,
    angle=0.0
):

    h, w = img.shape

    glcm = fast_glcm(
        img,
        vmin,
        vmax,
        levels,
        ks,
        distance,
        angle
    )

    homogeneity = np.zeros(
        (h, w),
        dtype=np.float32
    )

    for i in range(levels):
        for j in range(levels):

            homogeneity += (
                glcm[i, j] /
                (1. + (i - j) ** 2)
            )

    return homogeneity

# GLCM ASM & ENERGY
def fast_glcm_ASM(
    img,
    vmin=0,
    vmax=255,
    levels=8,
    ks=5,
    distance=1.0,
    angle=0.0
):

    h, w = img.shape
    glcm = fast_glcm(
        img,
        vmin,
        vmax,
        levels,
        ks,
        distance,
        angle
    )

    asm = np.zeros(
        (h, w),
        dtype=np.float32
    )

    for i in range(levels):
        for j in range(levels):
            asm += glcm[i, j] ** 2
    energy = np.sqrt(asm)
    return asm, energy

# MAIN PROGRAM
if __name__ == '__main__':
    # PARAMETER GLCM
    levels = 8
    ks = 5
    mi, ma = 0, 255

    # PATH DATASET

    base_folder_path = 'dataset'
    
    # LIST HASIL EKSTRAKSI
    data_fitur = []

    # VALIDASI FOLDER
    if not os.path.exists(base_folder_path):
        print(f"Folder '{base_folder_path}' tidak ditemukan!")
    else:
        print("MEMULAI EKSTRAKSI FITUR GLCM")

        # DETEKSI KELAS
        daftar_kelas = [

            d for d in os.listdir(base_folder_path)

            if os.path.isdir(
                os.path.join(
                    base_folder_path,
                    d
                )
            )
        ]

        print(f"Kelas terdeteksi: {daftar_kelas}")

        # LOOP TIAP KELAS
        for kelas in daftar_kelas:

            folder_kelas_path = os.path.join(
                base_folder_path,
                kelas
            )

            all_files = os.listdir(
                folder_kelas_path
            )

            print()
            print(f"Memproses kelas [{kelas}]")
            print(f"Jumlah file: {len(all_files)}")

            # LOOP FILE GAMBAR
            for file_name in all_files:
                if file_name.lower().endswith(
                    (
                        '.png',
                        '.jpg',
                        '.jpeg',
                        '.bmp',
                        '.tif'
                    )
                ):

                    img_path = os.path.join(
                        folder_kelas_path,
                        file_name
                    )

                    # BACA GAMBAR
                    img = cv2.imread(
                        img_path,
                        cv2.IMREAD_GRAYSCALE
                    )

                    if img is None:
                        print(
                            f"Gagal membaca: {file_name}"
                        )
                        continue

                    # RESIZE
                    img = cv2.resize(
                        img,
                        (256, 256)
                    )

                    # EKSTRAKSI FITUR
                    f_mean = fast_glcm_mean(
                        img,
                        mi,
                        ma,
                        levels,
                        ks
                    )

                    f_contrast = fast_glcm_contrast(
                        img,
                        mi,
                        ma,
                        levels,
                        ks
                    )

                    f_homo = fast_glcm_homogeneity(
                        img,
                        mi,
                        ma,
                        levels,
                        ks
                    )

                    f_asm, f_energy = fast_glcm_ASM(
                        img,
                        mi,
                        ma,
                        levels,
                        ks
                    )

                    # SIMPAN DATA
                    row_data = {

                        'Nama_File': file_name,

                        'Kelas': kelas,

                        'GLCM_Mean':
                            np.mean(f_mean),

                        'GLCM_Contrast':
                            np.mean(f_contrast),

                        'GLCM_Homogeneity':
                            np.mean(f_homo),

                        'GLCM_ASM':
                            np.mean(f_asm),

                        'GLCM_Energy':
                            np.mean(f_energy)
                    }

                    data_fitur.append(
                        row_data
                    )

                    print(
                        f"Berhasil: {file_name}"
                    )

        # SAVE CSV
        if data_fitur:
            df = pd.DataFrame(data_fitur)

            output_csv = (
                'hasil_ekstraksi_glcm_kelas.csv'
            )

            df.to_csv(
                output_csv,
                index=False
            )

            print()
            print("EKSTRAKSI SELESAI")

            print(
                f"CSV berhasil disimpan: "
                f"{output_csv}"
            )

            print()
            print("Sampel Data:")
            print(df.head())

        else:

            print(
                "Tidak ada gambar "
                "yang berhasil diproses."
            )