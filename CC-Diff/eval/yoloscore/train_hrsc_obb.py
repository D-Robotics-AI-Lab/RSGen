from ultralytics import YOLO
model = YOLO('yolov8m-obb.yaml')
results = model.train(data='config_hrsc.yaml', imgsz=512, epochs=50, batch=16, name='yolov8n')