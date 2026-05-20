def detect_fabric(frame):

    h, w, _ = frame.shape

    # area tengah kamera
    x1 = int(w * 0.25)
    y1 = int(h * 0.20)

    x2 = int(w * 0.75)
    y2 = int(h * 0.80)

    roi = frame[y1:y2, x1:x2]

    return roi