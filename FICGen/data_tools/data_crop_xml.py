import os
import cv2
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np

# 定义路径
# 请确保 image_dir 下是图片，xml_dir 下是对应的 xml 文件
image_dir = "/data/dior/train"
# 假设 xml 文件在这个目录下，如果不是，请修改为实际存放 xml 的路径
xml_dir = "/data/Annotations/Horizontal_Bounding_Boxes" 
output_dir = "/data/dior/patch"

os.makedirs(output_dir, exist_ok=True)

def parse_xml(xml_path):
    if not os.path.exists(xml_path):
        return []
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    return objects

counter = Counter()

image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
print(f"找到 {len(image_files)} 张图片，开始处理...")

for image_name in image_files:
    image_path = os.path.join(image_dir, image_name)
    
    xml_name = image_name.replace(".jpg", ".xml")
    xml_path = os.path.join(xml_dir, xml_name)
    
    if not os.path.exists(xml_path):
        print(f"Warning: No XML found for {image_name}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        continue
        
    image_height, image_width, _ = image.shape
    image_area = image_width * image_height
    
    objects = parse_xml(xml_path)
    
    for i, obj in enumerate(objects):
        class_name = obj['name']
        xmin, ymin, xmax, ymax = obj['bbox']
        
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image_width, xmax)
        ymax = min(image_height, ymax)
        
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        
        if bbox_width <= 0 or bbox_height <= 0:
            continue
            
        bbox_area = bbox_width * bbox_height
        
        bbox_ratio = bbox_area / image_area
        
        if bbox_ratio < 0.0005:
            continue
            
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        cropped_image = image[ymin:ymax, xmin:xmax]
        
        if cropped_image.size == 0:
            continue
            
        output_image_name = f"{image_name[:-4]}_{i}.jpg"
        output_image_path = os.path.join(class_dir, output_image_name)
        
        cv2.imwrite(output_image_path, cropped_image)
        counter[class_dir] += 1

print(counter)        

