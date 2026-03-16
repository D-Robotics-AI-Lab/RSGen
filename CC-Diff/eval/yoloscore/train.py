from ultralytics import YOLO
model = YOLO('yolov8n.yaml')
#dota 512  rsvg 800  
#val : dota 512 rsvg 800
results = model.train(data='config_dota.yaml', imgsz=512, epochs=50, batch=16, name='yolov8n')