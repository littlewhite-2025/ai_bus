from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.info()

results = model(r"C:/Users/a/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/python312/site-packages/ultralytics/assets/bus.jpg", save=True)
print(results)
