from ultralytics import YOLO

# Load a model
model = YOLO('/data/dior_hbb.pt')

# Customize validation settings
validation_results = model.val(data="config.yaml", imgsz=800, name='yolov8n_val', batch=16)