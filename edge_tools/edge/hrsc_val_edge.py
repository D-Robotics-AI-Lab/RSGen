import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import HRSC_TRAIN_RESULT_ROOT, HRSC_VAL_IMG, HRSC_VAL_ANNO, HRSC_HED_REF_VAL_OUT

import cv2
import numpy as np
import json
import math
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ================= config =================
class Config:
    TRAIN_RESULT_ROOT = HRSC_TRAIN_RESULT_ROOT

    VAL_IMG_DIR = HRSC_VAL_IMG
    VAL_ANNO_DIR = HRSC_VAL_ANNO

    OUTPUT_DIR = HRSC_HED_REF_VAL_OUT

    VIS_DIR = os.path.join(OUTPUT_DIR, 'vis_obb')

    INDEX_JSON_PATH = 'hrsc_train_patch_index.json'

# ================= helpers =================

def mbox_to_poly(cx, cy, w, h, angle):
    """HRSC mbox -> flat 8 coords."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    dx1, dy1 = w / 2, h / 2
    dx2, dy2 = w / 2, -h / 2
    
    points = [
        (cx + dx1 * cos_a - dy1 * sin_a, cy + dx1 * sin_a + dy1 * cos_a),
        (cx + dx2 * cos_a - dy2 * sin_a, cy + dx2 * sin_a + dy2 * cos_a),
        (cx - dx1 * cos_a + dy1 * sin_a, cy - dx1 * sin_a - dy1 * cos_a),
        (cx - dx2 * cos_a + dy2 * sin_a, cy - dx2 * sin_a - dy2 * cos_a)
    ]
    poly = []
    for p in points:
        poly.extend([p[0], p[1]])
    return poly

def load_hrsc_xml(xml_path):
    """Parse HRSC XML."""
    objects = []
    if not os.path.exists(xml_path):
        return objects
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        hrsc_objects = root.find('HRSC_Objects')
        if hrsc_objects is not None:
            for obj in hrsc_objects.findall('HRSC_Object'):
                label = "ship" 
                
                cx = float(obj.find('mbox_cx').text)
                cy = float(obj.find('mbox_cy').text)
                w = float(obj.find('mbox_w').text)
                h = float(obj.find('mbox_h').text)
                ang = float(obj.find('mbox_ang').text)
                
                poly = mbox_to_poly(cx, cy, w, h, ang)
                objects.append({'poly': poly, 'label': label})
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        
    return objects

def order_points_quadrangle(pts):
    pts = np.array(pts).reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    center = rect[0]
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    sort_idx = np.argsort(angles)
    pts = pts[sort_idx]
    
    d01 = np.linalg.norm(pts[0] - pts[1])
    d12 = np.linalg.norm(pts[1] - pts[2])
    if d01 < d12:
        pts = np.roll(pts, -1, axis=0)
    return pts

def build_or_load_index(root_dir, json_path):
    if os.path.exists(json_path):
        print(f"[Info] Loading index from {json_path}...")
        with open(json_path, 'r') as f:
            return json.load(f)
            
    print("[Info] Building index (First run)...")
    index = {}
    mask_root = os.path.join(root_dir, 'mask')
    
    if not os.path.exists(mask_root):
        os.makedirs(mask_root, exist_ok=True)
        print(f"[Warning] Mask dir created but empty: {mask_root}")
        return {}
        
    categories = sorted(os.listdir(mask_root))
    for cat in tqdm(categories, desc="Indexing"):
        cat_dir = os.path.join(mask_root, cat)
        if not os.path.isdir(cat_dir): continue
        
        index[cat] = []
        for fname in os.listdir(cat_dir):
            if not fname.endswith(('.png', '.bmp', '.jpg')): continue
            
            mask_path = os.path.join(cat_dir, fname)
            mask = cv2.imread(mask_path, 0)
            if mask is None: continue
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            rect = cv2.minAreaRect(contours[0])
            w, h = rect[1]
            if min(w, h) < 1: continue
            ar = max(w, h) / min(w, h)
            
            index[cat].append({
                'id': os.path.splitext(fname)[0],
                'ar': float(ar)
            })
            
    with open(json_path, 'w') as f:
        json.dump(index, f)
    return index

def find_best_match(target_ar, category, index_db):
    if category not in index_db:
        category = 'ship'
        if category not in index_db: return None
        
    candidates = index_db[category]
    if not candidates: return None
    candidates.sort(key=lambda x: abs(x['ar'] - target_ar))
    return candidates[0]

# ================= main =================

def process_hrsc_reference():
    if not os.path.exists(Config.OUTPUT_DIR): os.makedirs(Config.OUTPUT_DIR)
    if not os.path.exists(Config.VIS_DIR): os.makedirs(Config.VIS_DIR)
    
    train_index = build_or_load_index(Config.TRAIN_RESULT_ROOT, Config.INDEX_JSON_PATH)
    
    xml_files = sorted([f for f in os.listdir(Config.VAL_ANNO_DIR) if f.endswith('.xml')])
    
    for xml_name in tqdm(xml_files, desc="Processing HRSC"):
        xml_path = os.path.join(Config.VAL_ANNO_DIR, xml_name)
        
        objects = load_hrsc_xml(xml_path)
        if not objects: continue
        
        img_stem = os.path.splitext(xml_name)[0]
        img_name = None
        for ext in ['.bmp', '.jpg', '.png', '.tif']:
            if os.path.exists(os.path.join(Config.VAL_IMG_DIR, img_stem + ext)):
                img_name = img_stem + ext
                break
        
        if img_name is None: continue
        img_path = os.path.join(Config.VAL_IMG_DIR, img_name)
        
        val_img = cv2.imread(img_path)
        if val_img is None: continue
        h_val, w_val = val_img.shape[:2]
        
        canvas = np.zeros((h_val, w_val, 3), dtype=np.uint8)
        
        for obj in objects:
            poly = obj['poly']
            label = obj['label']
            
            val_pts = np.array(poly).reshape(4, 2).astype(np.float32)
            rect_val = cv2.minAreaRect(val_pts)
            w_v, h_v = rect_val[1]
            if min(w_v, h_v) < 1: continue
            val_ar = max(w_v, h_v) / min(w_v, h_v)
            
            match_info = find_best_match(val_ar, label, train_index)
            if match_info is None: continue 
            
            src_id = match_info['id']
            src_patch_path = os.path.join(Config.TRAIN_RESULT_ROOT, 'img_patch', label, src_id + '.jpg')
            src_mask_path = os.path.join(Config.TRAIN_RESULT_ROOT, 'mask', label, src_id + '.png')
            
            if not (os.path.exists(src_patch_path) and os.path.exists(src_mask_path)):
                continue
                
            src_img = cv2.imread(src_patch_path)
            src_mask = cv2.imread(src_mask_path, 0)
            
            contours, _ = cv2.findContours(src_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            src_rect_raw = cv2.minAreaRect(contours[0])
            src_pts = cv2.boxPoints(src_rect_raw)
            
            src_pts_ordered = order_points_quadrangle(src_pts)
            val_pts_ordered = order_points_quadrangle(val_pts)
            
            H = cv2.getPerspectiveTransform(src_pts_ordered, val_pts_ordered)
            warped_patch = cv2.warpPerspective(src_img, H, (w_val, h_val))
            warped_mask = cv2.warpPerspective(src_mask, H, (w_val, h_val))
            
            _, bin_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
            bin_mask_3c = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)

            patch_clean = cv2.bitwise_and(warped_patch, bin_mask_3c)
            canvas = cv2.max(canvas, patch_clean)
        
        save_img_name = img_stem + ".jpg"
        cv2.imwrite(os.path.join(Config.OUTPUT_DIR, save_img_name), canvas)

        vis_img = canvas.copy()
        for obj in objects:
            pts = np.array(obj['poly']).reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(Config.VIS_DIR, save_img_name), vis_img)

    print(f"HRSC Reference Map Generation Done. Results in {Config.OUTPUT_DIR}")

if __name__ == '__main__':
    process_hrsc_reference()
