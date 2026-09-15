from ultralytics import YOLO

model = YOLO("runs/detect/seat_detector-3/weights/best.pt")

model.predict(
    source="D:/test2/another_test/16.png",
    save=True,
    conf=0.4
)
