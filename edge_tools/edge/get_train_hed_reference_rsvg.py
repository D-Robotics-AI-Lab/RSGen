import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DIOR_TRAIN_HED, DIOR_ANNOTATIONS, DIOR_TRAIN_REF_ONLY

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

def process_single_pair(hed_path, xml_path, save_path):
    """HED + VOC-style XML (bndbox) -> mask HED inside boxes."""
    try:
        img = cv2.imread(hed_path)
        if img is None:
            return False, "failed to read image"

        h, w = img.shape[:2]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        mask = np.zeros((h, w), dtype=np.uint8)

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))

                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmax), min(h, ymax)

                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)

        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imwrite(save_path, result)
        return True, "ok"

    except Exception as e:
        return False, str(e)

def batch_process(hed_dir, xml_dir, output_dir):
    """Batch process folder of HED + XML."""
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
        xml_filename = name_stem + ".xml"
        xml_path = os.path.join(xml_dir, xml_filename)
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
    print(f"Output: {output_dir}")

# ========== paths (see paths.py) ==========
if __name__ == "__main__":
    input_hed_folder = DIOR_TRAIN_HED

    input_xml_folder = DIOR_ANNOTATIONS

    output_masked_folder = DIOR_TRAIN_REF_ONLY

    batch_process(input_hed_folder, input_xml_folder, output_masked_folder)
