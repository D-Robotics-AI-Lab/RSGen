import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import HRSC_TRAIN_HED, HRSC_TRAIN_ANNO, HRSC_CROP_OUT

import cv2
import numpy as np
import math
import xml.etree.ElementTree as ET
from tqdm import tqdm

def get_poly_from_mbox(cx, cy, w, h, ang):
    """HRSC mbox -> flat 8 values (4 corners x,y)."""
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)
    
    dx1, dy1 = w / 2, h / 2
    dx2, dy2 = w / 2, -h / 2
    
    pts = [
        (cx + dx1 * cos_a - dy1 * sin_a, cy + dx1 * sin_a + dy1 * cos_a),
        (cx + dx2 * cos_a - dy2 * sin_a, cy + dx2 * sin_a + dy2 * cos_a),
        (cx - dx1 * cos_a + dy1 * sin_a, cy - dx1 * sin_a - dy1 * cos_a),
        (cx - dx2 * cos_a + dy2 * sin_a, cy - dx2 * sin_a - dy2 * cos_a)
    ]
    poly = []
    for p in pts:
        poly.extend([p[0], p[1]])
    return poly

def _load_hrsc_xml(xml_path):
    """Parse HRSC XML. Returns {'ann': {'bboxes': ..., 'labels': ...}}."""
    bboxes, labels = [], []
    if not os.path.isfile(xml_path):
        return dict(ann=dict(bboxes=np.zeros((0, 8)), labels=[]))

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = root.find('HRSC_Objects')
        if objects is not None:
            for obj in objects.findall('HRSC_Object'):
                label = "ship"
                
                cx = float(obj.find('mbox_cx').text)
                cy = float(obj.find('mbox_cy').text)
                w = float(obj.find('mbox_w').text)
                h = float(obj.find('mbox_h').text)
                ang = float(obj.find('mbox_ang').text)
                
                poly = get_poly_from_mbox(cx, cy, w, h, ang)
                bboxes.append(poly)
                labels.append(label)
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")

    bboxes = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 8), dtype=np.float32)
    return dict(ann=dict(bboxes=bboxes, labels=labels))

def get_hbb_from_poly(poly, img_w, img_h):
    """Axis-aligned crop box from 8-number polygon."""
    poly = np.array(poly).reshape(4, 2)
    x_min = np.min(poly[:, 0])
    y_min = np.min(poly[:, 1])
    x_max = np.max(poly[:, 0])
    y_max = np.max(poly[:, 1])

    x_min = max(0, int(np.floor(x_min)))
    y_min = max(0, int(np.floor(y_min)))
    x_max = min(img_w, int(np.ceil(x_max)))
    y_max = min(img_h, int(np.ceil(y_max)))
    
    return x_min, y_min, x_max, y_max

if __name__ == '__main__':
    img_dir = HRSC_TRAIN_HED
    anno_dir = HRSC_TRAIN_ANNO
    save_dir = HRSC_CROP_OUT
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_list = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.bmp', '.jpg', '.png'))])
    
    for img_name in tqdm(img_list, desc="Processing HRSC"):
        name_prefix = os.path.splitext(img_name)[0]
        anno_path = os.path.join(anno_dir, name_prefix + '.xml')
        
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img, _ = img.shape

        if not os.path.exists(anno_path):
            continue
        content = _load_hrsc_xml(anno_path)['ann']
        
        class_count = {} 

        for label, poly in zip(content['labels'], content['bboxes']):

            x_min, y_min, x_max, y_max = get_hbb_from_poly(poly, w_img, h_img)
            w_crop = x_max - x_min
            h_crop = y_max - y_min
            
            if w_crop <= 1 or h_crop <= 1:
                continue

            patch_img = img[y_min:y_max, x_min:x_max]

            mask_img = np.zeros((h_crop, w_crop), dtype=np.uint8)

            poly_np = np.array(poly).reshape(4, 2)
            poly_local = (poly_np - np.array([x_min, y_min])).astype(np.int32)

            cv2.fillPoly(mask_img, [poly_local], 255)

            foreground_img = cv2.bitwise_and(patch_img, patch_img, mask=mask_img)

            if label not in class_count:
                class_count[label] = 0
            class_count[label] += 1
            idx = class_count[label]

            paths = {
                'img_patch': os.path.join(save_dir, 'img_patch', label),
                'mask': os.path.join(save_dir, 'mask', label),
                'foreground': os.path.join(save_dir, 'foreground', label)
            }

            for p in paths.values():
                os.makedirs(p, exist_ok=True)

            save_name = f'{name_prefix}_{idx}.jpg'
            cv2.imwrite(os.path.join(paths['img_patch'], save_name), patch_img)
            cv2.imwrite(os.path.join(paths['mask'], f'{name_prefix}_{idx}.png'), mask_img)
            cv2.imwrite(os.path.join(paths['foreground'], save_name), foreground_img)

    print("HRSC Processing done.")
