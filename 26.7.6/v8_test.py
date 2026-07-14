from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.info()

results = model(r"assets\training_files\L1.png", save=True)
print(results)
