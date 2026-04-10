import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import FILTER_TRAIN_HED, FILTER_LABEL_TXT, FILTER_TRAIN_REF_ONLY

import cv2
import numpy as np
import os
from tqdm import tqdm

def parse_dota_txt(txt_path):
    """
    Parse DOTA label txt.
    Returns a list of (4, 2) vertex arrays.
    DOTA line format: x1 y1 ... x4 y4 category difficulty
    """
    polygons = []
    if not os.path.exists(txt_path):
        return polygons

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 8:
            continue

        try:
            coords = list(map(float, parts[:8]))
            poly = np.array(coords).reshape(-1, 2)
            polygons.append(poly)
        except ValueError:
            continue

    return polygons

def process_single_pair(hed_path, txt_path, save_path):
    """HED image + DOTA txt -> polygon mask, keep edges inside polygons only."""
    try:
        img = cv2.imread(hed_path)
        if img is None:
            return False, "failed to read image"

        h, w = img.shape[:2]

        polygons = parse_dota_txt(txt_path)

        mask = np.zeros((h, w), dtype=np.uint8)

        if len(polygons) > 0:
            int_polys = [p.astype(np.int32).reshape((-1, 1, 2)) for p in polygons]

            cv2.fillPoly(mask, int_polys, 255)

        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imwrite(save_path, result)
        return True, "ok"

    except Exception as e:
        return False, str(e)

def batch_process(hed_dir, txt_dir, output_dir):
    """Batch: HED dir + DOTA labelTxt dir -> masked edges."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output dir: {output_dir}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in os.listdir(hed_dir) if os.path.splitext(f)[1].lower() in valid_extensions]

    print(f"Found {len(image_files)} images, processing...")

    success_count = 0
    fail_count = 0

    for img_file in tqdm(image_files, desc="Processing"):
        name_stem, ext = os.path.splitext(img_file)

        hed_path = os.path.join(hed_dir, img_file)

        txt_filename = name_stem + ".txt"
        txt_path = os.path.join(txt_dir, txt_filename)
        save_path = os.path.join(output_dir, img_file)

        if not os.path.exists(txt_path):
            fail_count += 1
            continue

        success, msg = process_single_pair(hed_path, txt_path, save_path)
        if success:
            success_count += 1
        else:
            print(f"\n[error] {img_file}: {msg}")
            fail_count += 1

    print("-" * 30)
    print(f"Done. success: {success_count}, failed/skipped: {fail_count}")
    print(f"Output: {output_dir}")

# ========== paths (see paths.py) ==========
if __name__ == "__main__":
    input_hed_folder = FILTER_TRAIN_HED

    input_txt_folder = FILTER_LABEL_TXT

    output_masked_folder = FILTER_TRAIN_REF_ONLY

    batch_process(input_hed_folder, input_txt_folder, output_masked_folder)
