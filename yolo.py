from ultralytics import YOLO
import cv2

# Load model YOLOv8 hasil training kamu
model = YOLO("runs/detect/train5/weights/best.pt")

# Buka webcam (device 0 biasanya webcam bawaan)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO predict pada frame webcam
    results = model(frame)

    # results[0].plot() menghasilkan gambar dengan box deteksi
    img_with_boxes = results[0].plot()

    # Tampilkan frame dengan hasil deteksi
    cv2.imshow("YOLOv8 Mask Detection", img_with_boxes)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()