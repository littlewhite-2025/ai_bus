from ultralytics import YOLO

model = YOLO("runs/detect/seat_detector-2/weights/best.pt")

model.train(
    data="seat_dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    patience=23,
    name="seat_detector"
)
