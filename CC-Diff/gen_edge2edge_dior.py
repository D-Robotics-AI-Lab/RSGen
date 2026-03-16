import os
import json
import argparse
import torch
import torch.distributed as dist
import numpy as np
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm

from ccdiff.utils import dict_of_images, hed_dict_of_images, find_nearest, seed_everything
from edge.edge2edge import SDXLEdge2Edge 

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

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-generate Edge2Edge refined images")
    
    # 路径参数
    parser.add_argument("--data_path", type=str, default="/data/val", help="Dataset root containing metadata.jsonl")
    parser.add_argument("--patch_root", type=str, default="/data/hed_patch", help="Root directory for HED patches")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the refined edge images")
    
    # --- 新增参数 ---
    parser.add_argument("--edge_path", type=str, default=None, help="If provided, load pre-existing edge images from this path instead of retrieving/stitching patches.")
    
    # 模型参数
    parser.add_argument("--edge2edge_model_path", type=str, default="/data/model/sd-xl")
    parser.add_argument("--edge2edge_lora_path", type=str, default="/data/lora.safetensors")
    
    # 生成参数
    parser.add_argument("--refiner_strength", type=float, default=0.6)
    parser.add_argument("--refiner_guidance_steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=512)
    
    parser.add_argument("--train_only", action="store_true", help="If True, skip Edge2Edge refinement and only output the canvas.")    
    return parser.parse_args()

def main():
    args = parse_args()
    rank, world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()

    metadata_path = os.path.join(args.data_path, "metadata.jsonl")
    data = []
    with open(metadata_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    data_shard = shard_list(data, rank, world_size)
    print(f"[Rank {rank}] Processing {len(data_shard)} samples...")

    todo_shard = []
    for sample in data_shard:
        file_name = os.path.basename(sample["file_name"])
        save_path = os.path.join(args.out_dir, file_name)
        if not (os.path.exists(save_path) and os.path.getsize(save_path) > 0):
            todo_shard.append(sample)
    
    skip_count = len(data_shard) - len(todo_shard)
    if skip_count > 0:
        print(f"[Rank {rank}] Automatic Resume: Skipped {skip_count} existing samples. Remaining: {len(todo_shard)}")

    if len(todo_shard) == 0:
        print(f"[Rank {rank}] All tasks completed. Skipping model loading.")
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    refiner = None
    if not args.train_only:
        print(f"[Rank {rank}] Loading SDXL Refiner...")
        refiner = SDXLEdge2Edge(
            model_path=args.edge2edge_model_path,
            lora_path=args.edge2edge_lora_path,
            device=device
        )
    else:
        print(f"[Rank {rank}] Running in Train Only mode (No Refiner).")
    
    seed_everything(args.seed)

    MIN_AREA_THRESHOLD = 0.01

    for sample in tqdm(data_shard, disable=(rank != 0)):
        file_name = os.path.basename(sample["file_name"])
        save_path = os.path.join(args.out_dir, file_name)

        if os.path.exists(save_path):
            continue

        prompt_str = sample["caption"]
        categories = prompt_str[1:] 
        bboxes = sample["bndboxes"]
        
        width, height = args.image_size, args.image_size
        
        # 初始化变量
        current_valid_boxes = []    
        current_valid_cats = []      
        current_restore_patches = [] 
        current_canvas = None
        
        
        use_pre_existing_edge = False
        if args.edge_path is not None:
            fname_no_ext = os.path.splitext(file_name)[0]
            edge_png_name = fname_no_ext + ".png"
            pre_edge_file_png = os.path.join(args.edge_path, edge_png_name)
            
            pre_edge_file_orig = os.path.join(args.edge_path, file_name)

            if os.path.exists(pre_edge_file_png):
                use_pre_existing_edge = True
                edge_load_path = pre_edge_file_png
            elif os.path.exists(pre_edge_file_orig):
                use_pre_existing_edge = True
                edge_load_path = pre_edge_file_orig
            else:
                # from IPython import embed; embed()
                print(f"[Rank {rank}] Warning: --edge_path provided but {edge_png_name} not found. Falling back to stitching.")

        if use_pre_existing_edge:
            try:
                fname_no_ext = os.path.splitext(file_name)[0]
                edge_png_name = os.path.join(args.edge_path, file_name)
                if not os.path.exists(edge_png_name):
                    edge_png_name = fname_no_ext + ".png"
                    # edge_png_name = fname_no_ext + ".png"
                current_canvas = Image.open(os.path.join(args.edge_path, edge_png_name)).convert("L")
                if current_canvas.size != (width, height):
                    current_canvas = current_canvas.resize((width, height), Image.NEAREST)
                
                for name, bbox in zip(categories, bboxes):
                    if name == '': continue
                    current_valid_boxes.append(bbox)
                    current_valid_cats.append(name)
            except Exception as e:
                print(f"Error loading edge from path {edge_png_name}: {e}")
                continue

        else:
            current_canvas = Image.new("L", (width, height), 0)
            
            for name, bbox in zip(categories, bboxes):
                if name == '':
                    continue
                
                current_valid_boxes.append(bbox)
                current_valid_cats.append(name)
                
                value = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
                if name not in hed_dict_of_images:
                    continue 
                    
                hed_chosen_file = list(hed_dict_of_images[name].keys())[find_nearest(list(hed_dict_of_images[name].values()), value)]
                patch_path = os.path.join(args.patch_root, name, hed_chosen_file)
                
                try:
                    hed_source_image = Image.open(patch_path).convert("L")
                except Exception as e:
                    print(f"Error loading patch {patch_path}: {e}")
                    continue

                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                w_box = max(1, x2 - x1)
                h_box = max(1, y2 - y1)
                
                ref_edge_resized = hed_source_image.resize((w_box, h_box), resample=Image.NEAREST)
                current_canvas.paste(ref_edge_resized, (x1, y1))
                
                box_area_ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if not args.train_only and box_area_ratio < MIN_AREA_THRESHOLD:
                    current_restore_patches.append((ref_edge_resized, x1, y1, w_box, h_box))

        refined_img = current_canvas 
        
        if not args.train_only:
            if len(current_valid_boxes) > 0:
                try:
                    refined_img = refiner.refine(
                        init_image=current_canvas,
                        hbboxes=current_valid_boxes, 
                        obboxes=None, 
                        caption=prompt_str[0],
                        class_names=current_valid_cats, 
                        strength=args.refiner_strength,
                        seed=args.seed,
                        num_inference_steps=50, 
                        num_guidance_steps=args.refiner_guidance_steps,
                    )
                except Exception as e:
                    print(f"[Rank {rank}] Refiner error on {file_name}: {e}")
                    refined_img = current_canvas
            
            if len(current_restore_patches) > 0:
                if refined_img.size != (width, height):
                    refined_img = refined_img.resize((width, height), Image.NEAREST)
                
                refined_img = refined_img.convert("L")
                refined_np = np.array(refined_img)

                for patch, px, py, pw, ph in current_restore_patches:
                    patch_np = np.array(patch)
                    
                    py_end = min(py + ph, height)
                    px_end = min(px + pw, width)
                    h_eff = py_end - py
                    w_eff = px_end - px
                    
                    if h_eff <= 0 or w_eff <= 0: continue
                        
                    patch_crop = patch_np[:h_eff, :w_eff]
                    roi = refined_np[py:py_end, px:px_end]
                    
                    fused_roi = np.maximum(roi, patch_crop)
                    refined_np[py:py_end, px:px_end] = fused_roi
                
                refined_img = Image.fromarray(refined_np)
            

        refined_img.save(save_path)
        vis_img_np = np.array(refined_img.convert("L"))
        vis_img_bgr = cv2.cvtColor(vis_img_np, cv2.COLOR_GRAY2BGR)

        for bbox, name in zip(bboxes, categories):
            if name == '': 
                continue
            
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
            
            cv2.rectangle(vis_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            



        vis_save_path = os.path.join(args.out_dir, "vis_gt", file_name)
        cv2.imwrite(vis_save_path, vis_img_bgr)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    
    if rank == 0:
        print("Pre-generation complete.")

if __name__ == "__main__":
    main()





