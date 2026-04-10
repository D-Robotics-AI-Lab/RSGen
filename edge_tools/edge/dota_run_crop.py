import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DOTA_TRAIN_HED, DOTA_LABELS_TRAIN, FILTER_RESULTS_OBB

import cv2
import numpy as np
from tqdm import tqdm

def _load_dota_txt(txtfile):
    """Load DOTA label txt. Returns {'ann': {'bboxes': ..., 'labels': ...}}."""
    bboxes, labels, diffs = [], [], []
    if txtfile is None or not os.path.isfile(txtfile):
        print(f"Can't find {txtfile}, treated as empty txtfile")
    else:
        with open(txtfile, 'r') as f:
            for line in f:
                if line.startswith('gsd'):
                    continue

                items = line.split(' ')
                if len(items) >= 9:
                    # first 8 numbers: polygon coords
                    bboxes.append([float(i) for i in items[:8]])
                    labels.append(items[8])
                    diffs.append(int(items[9]) if len(items) == 10 else 0)

    bboxes = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 8), dtype=np.float32)
    return dict(ann=dict(bboxes=bboxes, labels=labels))

def get_hbb_from_poly(poly, img_w, img_h):
    """Axis-aligned bounding box from 8-number polygon; clip to image bounds."""
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
    img_dir = DOTA_TRAIN_HED
    anno_dir = DOTA_LABELS_TRAIN
    save_dir = FILTER_RESULTS_OBB

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_list = sorted(os.listdir(img_dir))
    for img_name in tqdm(img_list):
        if not img_name.lower().endswith(('.png', '.jpg', '.bmp')):
            continue

        name_prefix = os.path.splitext(img_name)[0]
        anno_name = name_prefix + '.txt'
        anno_path = os.path.join(anno_dir, anno_name)
        
        # cv2 reads BGR
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img, _ = img.shape

        if not os.path.exists(anno_path):
            continue
        content = _load_dota_txt(anno_path)['ann']
        
        # per-class instance index for naming (e.g. plane_1.jpg, plane_2.jpg)
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
            poly_local = poly_np - np.array([x_min, y_min])

            poly_local = poly_local.astype(np.int32)

            cv2.fillPoly(mask_img, [poly_local], 255)

            foreground_img = cv2.bitwise_and(patch_img, patch_img, mask=mask_img)

            if label not in class_count:
                class_count[label] = 0
            class_count[label] += 1
            idx = class_count[label]

            save_folder_patch = os.path.join(save_dir, 'img_patch', label)
            os.makedirs(save_folder_patch, exist_ok=True)
            cv2.imwrite(os.path.join(save_folder_patch, f'{name_prefix}_{idx}.jpg'), patch_img)

            save_folder_mask = os.path.join(save_dir, 'mask', label)
            os.makedirs(save_folder_mask, exist_ok=True)
            cv2.imwrite(os.path.join(save_folder_mask, f'{name_prefix}_{idx}.png'), mask_img)

            save_folder_fg = os.path.join(save_dir, 'foreground', label)
            os.makedirs(save_folder_fg, exist_ok=True)
            cv2.imwrite(os.path.join(save_folder_fg, f'{name_prefix}_{idx}.jpg'), foreground_img)

    print("Processing done.")
