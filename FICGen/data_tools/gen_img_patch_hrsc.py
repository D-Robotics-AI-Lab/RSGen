import os
import json
from PIL import Image
from tqdm import tqdm


JSONL_PATH = "metadata.jsonl" 
IMG_DIR = "/data/HRSC2016/Train/AllImages" 
SAVE_DIR = "img_patch"     # 

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def crop_objects():
    print(f"Loading {JSONL_PATH}...")
    
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_crops = 0
    for line in tqdm(lines, desc="Cropping"):
        data = json.loads(line)
        file_name = data['file_name']
        img_path = os.path.join(IMG_DIR, file_name)
        
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            w_img, h_img = img.size
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            continue

        bndboxes = data['bndboxes']
        
        file_stem = os.path.splitext(file_name)[0]

        for i, box in enumerate(bndboxes):
            if all(v == 0 for v in box):
                continue
            
            xmin = box[0] * w_img
            ymin = box[1] * h_img
            xmax = box[2] * w_img
            ymax = box[3] * h_img
            
            left = max(0, int(xmin))
            top = max(0, int(ymin))
            right = min(w_img, int(xmax))
            bottom = min(h_img, int(ymax))

            if right <= left or bottom <= top:
                continue

            crop = img.crop((left, top, right, bottom))
            
            save_name = f"{file_stem}_{i}.jpg"
            save_path = os.path.join(SAVE_DIR, save_name)
            
            crop.save(save_path)
            total_crops += 1

    print(f"\nFinished! Successfully cropped {total_crops} objects to {SAVE_DIR}")

if __name__ == "__main__":
    crop_objects()
