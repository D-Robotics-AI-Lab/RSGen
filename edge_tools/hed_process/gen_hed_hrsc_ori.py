#obbox
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import HRSC_TRAIN_IMG, HRSC_TRAIN_HED

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from hed.init import HEDdetector
from hed.util import common_annotator_call

# ================= config =================
IMAGE_BASE_DIR = HRSC_TRAIN_IMG

OUTPUT_DIR = HRSC_TRAIN_HED

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
# ===========================================

# ================= HED model (singleton) =================
hed_model = None

def get_hed_model(device):
    global hed_model
    if hed_model is None:
        print("Loading HED model...")
        hed_model = HEDdetector.from_pretrained().to(device)
    return hed_model

def hed_pred(image, device):
    model = get_hed_model(device)
    image_np = np.array(image).astype(np.float32) / 255.0
    
    image_tensor = torch.from_numpy(image_np).unsqueeze(0).to(device)
    
    out = common_annotator_call(model, image_tensor)
    
    if isinstance(out, torch.Tensor):
        out = out.squeeze().cpu().numpy()
    
    if out.max() <= 1.0:
        out = (out * 255).astype(np.uint8)
    else:
        out = out.astype(np.uint8)
        
    return Image.fromarray(out)

# ================= main =================
def process_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(IMAGE_BASE_DIR):
        print(f"Error: Directory not found {IMAGE_BASE_DIR}")
        return

    all_files = os.listdir(IMAGE_BASE_DIR)
    image_files = [f for f in all_files if f.lower().endswith(VALID_EXTENSIONS)]
    
    print(f"Found {len(image_files)} images in {IMAGE_BASE_DIR}")

    for file_name in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(IMAGE_BASE_DIR, file_name)

        try:
            ori_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {file_name}: {e}")
            continue

        try:
            hed_image = hed_pred(ori_image, device)
        except Exception as e:
            print(f"HED failed for {file_name}: {e}")
            continue

        base_name_no_ext = os.path.splitext(file_name)[0]
        save_img_name = f"{base_name_no_ext}.png"
        save_img_path = os.path.join(OUTPUT_DIR, save_img_name)

        hed_image.save(save_img_path)

if __name__ == "__main__":
    process_dataset()
