import os
import json
import argparse
import random
import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from edge.edge2edge import SDXLEdge2Edge 



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        backend = "nccl"
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        local_rank = 0
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def shard_list(data, rank, world_size):
    return data[rank::world_size]

# ==========================================
# 2. 主逻辑
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Edge2Edge refinement using pre-generated HED maps")
    
    # 路径参数
    parser.add_argument("--data_path", type=str, default="/data/images", help="Dataset root containing metadata.jsonl")
    parser.add_argument("--hed_root", type=str, default="/data/val", help="Directory containing pre-generated HED reference images")
    parser.add_argument("--out_dir", type=str, default="/data/edge", help="Directory to save the refined images")
    
    # 模型参数
    parser.add_argument("--edge2edge_model_path", type=str, default="/data/model/sd-xl")
    parser.add_argument("--edge2edge_lora_path", type=str, default="/data/lora.safetensors")
    
    # 生成参数
    parser.add_argument("--refiner_strength", type=float, default=0.6)
    parser.add_argument("--refiner_guidance_steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=512)
    
    return parser.parse_args()

def main():
    args = parse_args()
    rank, world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        vis_dir = os.path.join(args.out_dir, "vis_gt")
        os.makedirs(vis_dir, exist_ok=True)
        
    if world_size > 1:
        dist.barrier()

    metadata_path = os.path.join(args.data_path, "metadata.jsonl")
    data = []
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        if rank == 0: print(f"Error: Metadata file not found at {metadata_path}")
        return

    data_shard = shard_list(data, rank, world_size)
    if rank == 0:
        print(f"Total samples: {len(data)}. Processing {len(data_shard)} on Rank {rank}.")

    if rank == 0: print("Loading SDXL Refiner...")
    refiner = SDXLEdge2Edge(
        model_path=args.edge2edge_model_path,
        lora_path=args.edge2edge_lora_path,
        device=device
    )
    
    seed_everything(args.seed)
    
    MIN_AREA_THRESHOLD = 0.01

    for sample in tqdm(data_shard, disable=(rank != 0)):
        file_name = os.path.basename(sample["file_name"])
        save_path = os.path.join(args.out_dir, file_name)
        
        hed_img_path = os.path.join(args.hed_root, file_name)
        
        if not os.path.exists(hed_img_path):
            fname_no_ext = os.path.splitext(file_name)[0]
            hed_img_path_png = os.path.join(args.hed_root, fname_no_ext + ".png")
            if os.path.exists(hed_img_path_png):
                hed_img_path = hed_img_path_png
            else:
                if rank == 0: print(f"Warning: HED image not found: {hed_img_path}")
                continue
            
        try:
            init_image = Image.open(hed_img_path).convert("RGB")
            
            if init_image.size != (args.image_size, args.image_size):
                init_image = init_image.resize((args.image_size, args.image_size), Image.NEAREST)
        except Exception as e:
            print(f"Error loading image {hed_img_path}: {e}")
            continue

        prompt_str = sample["caption"]  # [Full_Caption, Class1, Class2...]
        categories = prompt_str[1:] 
        polys_norm = sample.get("obboxes", [])
        
        current_valid_boxes = []     
        current_valid_cats = []
        current_valid_obboxes = []
        current_restore_patches = []
        
        width, height = args.image_size, args.image_size
        
        for name, poly_n in zip(categories, polys_norm):
            if name == '': continue
            
            poly_np = np.array(poly_n).reshape(-1, 2)
            
            x_min, y_min = np.min(poly_np, axis=0)
            x_max, y_max = np.max(poly_np, axis=0)
            
            x_min_c, y_min_c = max(0.0, float(x_min)), max(0.0, float(y_min))
            x_max_c, y_max_c = min(1.0, float(x_max)), min(1.0, float(y_max))
            
            hbbox = [x_min_c, y_min_c, x_max_c, y_max_c]

            box_area = (x_max_c - x_min_c) * (y_max_c - y_min_c)
            
            if box_area < MIN_AREA_THRESHOLD:

                px1, py1 = int(x_min_c * width), int(y_min_c * height)
                px2, py2 = int(x_max_c * width), int(y_max_c * height)
                pw, ph = max(1, px2 - px1), max(1, py2 - py1)

                patch_crop = init_image.crop((px1, py1, px2, py2))
                

                current_restore_patches.append((patch_crop, px1, py1, pw, ph))


            current_valid_boxes.append(hbbox)
            current_valid_cats.append(name)
            current_valid_obboxes.append(poly_n) 

        refined_img = init_image
        
        if len(current_valid_boxes) > 0:
            try:
                refined_img = refiner.refine(
                    init_image=init_image,
                    hbboxes=current_valid_boxes,
                    obboxes=current_valid_obboxes,
                    caption=prompt_str[0], 
                    class_names=current_valid_cats,
                    strength=args.refiner_strength,
                    seed=args.seed,
                    num_inference_steps=50,
                    num_guidance_steps=args.refiner_guidance_steps,
                )
            except Exception as e:
                print(f"Refiner error on {file_name}: {e}")
                refined_img = init_image


        if len(current_restore_patches) > 0:
            if refined_img.size != (width, height):
                refined_img = refined_img.resize((width, height), Image.NEAREST)
            
            refined_np = np.array(refined_img)
            
            for patch, px, py, pw, ph in current_restore_patches:
                patch_np = np.array(patch) # (h, w, 3)

                py_end = min(py + ph, height)
                px_end = min(px + pw, width)
                
                h_eff = py_end - py
                w_eff = px_end - px
                
                if h_eff <= 0 or w_eff <= 0:
                    continue
                

                patch_crop_eff = patch_np[:h_eff, :w_eff]
                roi = refined_np[py:py_end, px:px_end]
                

                fused_roi = np.maximum(roi, patch_crop_eff)
                

                refined_np[py:py_end, px:px_end] = fused_roi
            

            refined_img = Image.fromarray(refined_np)


        refined_img.save(save_path)


        if polys_norm:
            vis_img = np.array(refined_img) 
            if len(vis_img.shape) == 2:
                 vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
            else:
                 vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

            for poly_n, name in zip(polys_norm, categories):
                poly_abs = np.array(poly_n).reshape(-1, 2) * args.image_size
                poly_int = poly_abs.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [poly_int], isClosed=True, color=(0, 255, 0), thickness=2)
            
            vis_save_path = os.path.join(args.out_dir, "vis_gt", file_name)
            cv2.imwrite(vis_save_path, vis_img)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    
    if rank == 0:
        print("Generation complete.")

if __name__ == "__main__":
    main()
