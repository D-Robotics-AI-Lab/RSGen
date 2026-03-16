from ultralytics import YOLO

# Load a model
model = YOLO('/data/best_hrsc.pt')

# Customize validation settings
validation_results = model.val(data="config_hrsc.yaml", imgsz=512, name='yolov8n_val', batch=16)