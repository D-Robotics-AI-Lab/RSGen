from ultralytics import YOLO

# Load a model
model = YOLO('/data/dota_obbox.pt')

# Customize validation settings
validation_results = model.val(data="config_dota.yaml", imgsz=512, name='yolov8n_val', batch=16)