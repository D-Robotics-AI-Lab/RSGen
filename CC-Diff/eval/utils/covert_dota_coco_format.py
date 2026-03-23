import os
import json
from PIL import Image
import numpy as np

def calculate_area(points):
    """
    Calculate polygon area using shoelace formula.
    points: list of [x1,y1,x2,y2,...] closed.
    """
    n = len(points) // 2
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[2*i] * points[2*j + 1]
        area -= points[2*j] * points[2*i + 1]
    area /= 2.0
    return abs(area)

def calculate_bbox(points):
    """
    Calculate axis-aligned bbox from points.
    points: [x1,y1,x2,y2,...]
    return [xmin, ymin, width, height]
    """
    xs = points[0::2]
    ys = points[1::2]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    return [xmin, ymin, xmax - xmin, ymax - ymin]

def dota_to_coco(ann_dir, img_dir, out_json):
    """
    Convert DOTA v1 TXT annotations to COCO JSON format.
    
    Args:
        ann_dir (str): Directory with TXT annotation files.
        img_dir (str): Directory with image files (assumed .png).
        out_json (str): Path to output JSON file.
    """
    # DOTA v1 categories
    categories = [
        {"id": 1, "name": "plane"},
        {"id": 2, "name": "ship"},
        {"id": 3, "name": "storage-tank"},
        {"id": 4, "name": "baseball-diamond"},
        {"id": 5, "name": "tennis-court"},
        {"id": 6, "name": "basketball-court"},
        {"id": 7, "name": "ground-track-field"},
        {"id": 8, "name": "harbor"},
        {"id": 9, "name": "bridge"},
        {"id": 10, "name": "large-vehicle"},
        {"id": 11, "name": "small-vehicle"},
        {"id": 12, "name": "helicopter"},
        {"id": 13, "name": "roundabout"},
        {"id": 14, "name": "soccer-ball-field"},
        {"id": 15, "name": "swimming-pool"}
    ]
    
    cat_name_to_id = {cat["name"]: cat["id"] for cat in categories}
    
    images = []
    annotations = []
    img_id = 0
    ann_id = 0
    
    for ann_file in sorted(os.listdir(ann_dir)):
        if not ann_file.endswith('.txt'):
            continue
        
        base_name = ann_file[:-4]  # remove .txt
        img_path = os.path.join(img_dir, base_name + '.png')  # assume .png, change if needed
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # Get image size
        with Image.open(img_path) as img:
            width, height = img.size
        
        images.append({
            "id": img_id,
            "file_name": base_name + '.png',
            "height": height,
            "width": width
        })
        
        # Read annotations
        with open(os.path.join(ann_dir, ann_file), 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 10:
                continue  # invalid line
            
            x1, y1, x2, y2, x3, y3, x4, y4, class_name, difficult = parts
            points = [float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)]
            # Close the polygon
            segmentation = points + [points[0], points[1]]
            
            bbox = calculate_bbox(points)
            area = calculate_area(points)
            
            cat_id = cat_name_to_id.get(class_name, None)
            if cat_id is None:
                continue  # unknown class
            
            iscrowd = 1 if int(difficult) == 1 else 0
            
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": area,
                "segmentation": [segmentation],  # list of lists for COCO
                "iscrowd": iscrowd
            })
            
            ann_id += 1
        
        img_id += 1
    
    coco_json = {
        "info": {
            "description": "DOTA v1 converted to COCO format",
            "version": "1.0",
            "year": 2025,
            "contributor": "Your Name",
            "date_created": "2025/09/15"
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": categories,
        "images": images,
        "annotations": annotations
    }
    
    with open(out_json, 'w') as f:
        json.dump(coco_json, f, indent=4)
    
    print(f"Conversion complete. Output saved to {out_json}")

# Example usage:
dota_to_coco('/home/users/xianbao01.hou/CC-Diff/dataset/filter_val/labelTxt', '/home/users/xianbao01.hou/CC-Diff/dataset/filter_val/images', '/home/users/xianbao01.hou/CC-Diff/dataset/filter_val/val_coco.json')