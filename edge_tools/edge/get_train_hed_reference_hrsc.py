import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import HRSC_TRAIN_HED, HRSC_TRAIN_ANNO, HRSC_TRAIN_REF_OBB

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import math
from tqdm import tqdm

def get_rotated_points(cx, cy, w, h, angle):
    """HRSC mbox -> 4 corner points."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    dx1, dy1 = w / 2, h / 2
    dx2, dy2 = w / 2, -h / 2

    pts = [
        (cx + dx1 * cos_a - dy1 * sin_a, cy + dx1 * sin_a + dy1 * cos_a),
        (cx + dx2 * cos_a - dy2 * sin_a, cy + dx2 * sin_a + dy2 * cos_a),
        (cx - dx1 * cos_a + dy1 * sin_a, cy - dx1 * sin_a - dy1 * cos_a),
        (cx - dx2 * cos_a + dy2 * sin_a, cy - dx2 * sin_a - dy2 * cos_a)
    ]

    return np.array(pts, dtype=np.int32)

def process_single_pair(hed_path, xml_path, save_path):
    """HED + HRSC XML -> rotated polygon mask, bitwise-and with HED."""
    try:
        img = cv2.imread(hed_path)
        if img is None:
            return False, "failed to read image"

        h_img, w_img = img.shape[:2]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        mask = np.zeros((h_img, w_img), dtype=np.uint8)

        objects_root = root.find('HRSC_Objects')
        if objects_root is not None:
            for obj in objects_root.findall('HRSC_Object'):
                try:
                    cx = float(obj.find('mbox_cx').text)
                    cy = float(obj.find('mbox_cy').text)
                    w = float(obj.find('mbox_w').text)
                    h = float(obj.find('mbox_h').text)
                    ang = float(obj.find('mbox_ang').text)

                    pts = get_rotated_points(cx, cy, w, h, ang)

                    cv2.fillPoly(mask, [pts], 255)
                except (AttributeError, ValueError):
                    continue

        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imwrite(save_path, result)
        return True, "ok"

    except Exception as e:
        return False, str(e)

def batch_process(hed_dir, xml_dir, output_dir):
    """Batch mask HED by HRSC rotated boxes."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output dir: {output_dir}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in os.listdir(hed_dir) if os.path.splitext(f)[1].lower() in valid_extensions]

    print(f"Found {len(image_files)} images (HRSC masking)...")

    success_count = 0
    fail_count = 0

    for img_file in tqdm(image_files, desc="Masking HED"):
        name_stem = os.path.splitext(img_file)[0]

        hed_path = os.path.join(hed_dir, img_file)
        xml_path = os.path.join(xml_dir, name_stem + ".xml")
        save_path = os.path.join(output_dir, img_file)

        if not os.path.exists(xml_path):
            fail_count += 1
            continue

        success, msg = process_single_pair(hed_path, xml_path, save_path)
        if success:
            success_count += 1
        else:
            print(f"\n[error] {img_file}: {msg}")
            fail_count += 1

    print("-" * 30)
    print(f"Done. success: {success_count}, failed/skipped: {fail_count}")

# ========== paths (see paths.py) ==========
if __name__ == "__main__":
    input_hed_folder = HRSC_TRAIN_HED

    input_xml_folder = HRSC_TRAIN_ANNO

    output_masked_folder = HRSC_TRAIN_REF_OBB

    batch_process(input_hed_folder, input_xml_folder, output_masked_folder)
