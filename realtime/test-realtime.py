import cv2
import os
import time
import winsound

from datetime import datetime

from alarm import detect_defect

# =========================
# FOLDER HASIL
# =========================

save_folder = '../results/hasil_deteksi'

if not os.path.exists(save_folder):

    os.makedirs(save_folder)

print("Realtime Deteksi Kain")
print("Q = Keluar")
print("S = Simpan Screenshot")

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

prev_time = 0

# =========================
# LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(
        frame,
        1
    )

    display = frame.copy()

    # =========================
    # DETECTION
    # =========================

    detected, best_rect, conf, edges = detect_defect(
        frame
    )

    # =========================
    # DRAW RESULT
    # =========================

    if detected and best_rect is not None:

        bx, by, bw, bh = best_rect

        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240

        x1 = int(
            bx * scale_x
        )

        y1 = int(
            by * scale_y
        )

        x2 = int(
            (bx+bw) * scale_x
        )

        y2 = int(
            (by+bh) * scale_y
        )

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0,0,255),
            3
        )

        cv2.putText(
            display,
            'RUSAK',
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

        cv2.putText(
            display,
            f'Confidence: {conf:.2f}',
            (x1, y2+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )

        winsound.Beep(1000, 300)

    else:

        cv2.putText(
            display,
            'NORMAL',
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

    # =========================
    # FPS
    # =========================

    current_time = time.time()

    fps = 1 / (
        current_time - prev_time
    )

    prev_time = current_time

    cv2.putText(
        display,
        f'FPS: {int(fps)}',
        (30,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    # =========================
    # TIMESTAMP
    # =========================

    now = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    cv2.putText(
        display,
        now,
        (10,470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1
    )

    # =========================
    # SHOW
    # =========================

    cv2.imshow(
        'Realtime Fabric Defect Detection',
        display
    )

    cv2.imshow(
        'Edges',
        edges
    )

    # =========================
    # KEYBOARD
    # =========================

    key = cv2.waitKey(1)

    if key == ord('s'):

        filename = datetime.now().strftime(
            "deteksi_%Y%m%d_%H%M%S.jpg"
        )

        path = os.path.join(
            save_folder,
            filename
        )

        cv2.imwrite(
            path,
            display
        )

        print(f"Gambar disimpan: {path}")

    if key == ord('q'):
        break

# =========================
# RELEASE
# =========================

cap.release()

cv2.destroyAllWindows()