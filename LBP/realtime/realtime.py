import cv2
import os
import time
import winsound

from datetime import datetime

from detector import detect_defect

# =========================
# FOLDER HASIL
# =========================

save_folder = '../results/hasil_deteksi'

os.makedirs(
    save_folder,
    exist_ok=True
)

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

# =========================
# PRINT INFO
# =========================

print("=" * 60)
print(" REALTIME FABRIC DEFECT DETECTION ")
print("          LBP + KNN")
print("=" * 60)

print("Q = Keluar")
print("S = Simpan Screenshot")

print("=" * 60)

# =========================
# FPS
# =========================

prev_time = 0

# =========================
# ALARM TIMER
# =========================

last_alarm_time = 0

# =========================
# FRAME SKIP
# =========================

frame_count = 0

# =========================
# LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # MIRROR
    # =========================

    frame = cv2.flip(
        frame,
        1
    )

    display = frame.copy()

    # =========================
    # FRAME SKIP
    # =========================

    frame_count += 1

    if frame_count % 2 != 0:

        cv2.imshow(
            'Realtime Fabric Detection',
            display
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        continue

    # =========================
    # DETECTION
    # =========================

    detected, best_rect, conf, prediction = detect_defect(
        frame
    )

    # =========================
    # DRAW RESULT
    # =========================

    if detected and best_rect is not None:

        x, y, w, h = best_rect

        # =========================
        # BOUNDING BOX
        # =========================

        cv2.rectangle(
            display,
            (x,y),
            (x+w, y+h),
            (0,0,255),
            3
        )

        # =========================
        # LABEL
        # =========================

        cv2.putText(
            display,
            f'{prediction.upper()}',
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

        # =========================
        # CONFIDENCE
        # =========================

        cv2.putText(
            display,
            f'Confidence: {conf*100:.0f}%',
            (x, y+h+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )

        # =========================
        # STATUS
        # =========================

        cv2.putText(
            display,
            'STATUS: RUSAK',
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

        # =========================
        # ALARM
        # =========================

        current_alarm_time = time.time()

        if current_alarm_time - last_alarm_time > 2:

            winsound.MessageBeep()

            last_alarm_time = current_alarm_time

    else:

        # =========================
        # NORMAL
        # =========================

        cv2.putText(
            display,
            'STATUS: NORMAL',
            (20,40),
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
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
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
    # UI BORDER
    # =========================

    cv2.rectangle(
        display,
        (0,0),
        (639,479),
        (255,255,255),
        2
    )

    # =========================
    # SHOW
    # =========================

    cv2.imshow(
        'Realtime Fabric Detection',
        display
    )

    # =========================
    # KEYBOARD
    # =========================

    key = cv2.waitKey(1)

    # =========================
    # SAVE SCREENSHOT
    # =========================

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

        print()
        print(f"Gambar disimpan:")
        print(path)

    # =========================
    # EXIT
    # =========================

    if key == ord('q'):
        break

# =========================
# RELEASE
# =========================

cap.release()

cv2.destroyAllWindows()

print()
print("Program selesai.")