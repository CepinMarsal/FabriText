import cv2

def draw_status(frame, prediction, confidence):

    color = (0, 255, 0)

    if prediction == "rusak":
        color = (0, 0, 255)

    cv2.putText(
        frame,
        f"Status : {prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Confidence : {confidence:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )