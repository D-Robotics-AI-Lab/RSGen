import os
import json
import torch
import torch.distributed as dist
torch.backends.cuda.matmul.allow_tf32 = False
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import math
from diffusers import EulerDiscreteScheduler
from migc.migc_utils import seed_everything, load_migc
from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore

# ================= 参数解析 =================

def parse_args():
    parser = argparse.ArgumentParser(description="MIGC Inference Script")
    
    parser.add_argument(
        "--migc_ckpt_path", 
        type=str, 
        default='/data/migc.pth', 
    )
    parser.add_argument(
        "--sd1x_path", 
        type=str, 
        default='/data/sd1-4', 
    )
    parser.add_argument(
        "--base_path", 
        type=str, 
        default="/data/filter_val/images", 
    )
    parser.add_argument(
        "--jsonl_path", 
        type=str, 
        default="/data/filter_val/images/metadata.jsonl", 
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./dota_out", 
    )
    parser.add_argument(
        "--font_path", 
        type=str, 
        default="Rainbow-Party-2.ttf", 
    )

    # 推理参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--migc_steps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=1)
    
    return parser.parse_args()

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Info] Running in single process mode.")
    
    return rank, world_size, device

def load_migc_model(device, sd_path, migc_path, font_path):
    print(f"Loading MIGC model from {sd_path} ...")

    pipe = StableDiffusionMIGCPipeline.from_pretrained(sd_path)
    

    pipe.attention_store = AttentionStore()

    if not os.path.isfile(migc_path):
        raise FileNotFoundError(f"MIGC weights not found at {migc_path}")
        
    load_migc(pipe.unet, pipe.attention_store, migc_path, attn_processor=MIGCProcessor)
    

    pipe = pipe.to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    

    def draw_wrapper(image, bboxes, phrases):
        return draw_box_desc_standalone(image, bboxes, phrases, font_path)
    pipe.draw_box_desc = draw_wrapper
        
    return pipe

def draw_box_desc_standalone(image, bboxes, phrases, font_path="Rainbow-Party-2.ttf"):

    draw = ImageDraw.Draw(image)
    w, h = image.size
    
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:

        font = ImageFont.load_default()

    for box, phrase in zip(bboxes, phrases):

        x0, y0, x1, y1 = box[0]*w, box[1]*h, box[2]*w, box[3]*h
        

        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)


        if hasattr(draw, "textbbox"):
            text_w, text_h = draw.textbbox((0, 0), phrase, font=font)[2:]
        else:
            text_w, text_h = draw.textsize(phrase, font=font)

        draw.rectangle([x0, y0, x0 + text_w, y0 + text_h], fill="red")
        

        draw.text((x0, y0), phrase, fill="white", font=font)
        
    return image


def main():
    args = parse_args()
    

    rank, world_size, device = setup_distributed()


    raw_dir = os.path.join(args.output_dir, "raw")
    concat_dir = os.path.join(args.output_dir, "concat")
    anno_dir = os.path.join(args.output_dir, "anno")
    
    if rank == 0:
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(concat_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)
        print(f"Output directories created at {args.output_dir}")
        print(f"Using arguments: {vars(args)}")
    

    if world_size > 1:
        dist.barrier()


    try:
        pipe = load_migc_model(device, args.sd1x_path, args.migc_ckpt_path, args.font_path)
    except Exception as e:
        print(f"[Rank {rank}] Critical Error loading model: {e}")
        return


    if not os.path.exists(args.jsonl_path):
        if rank == 0: print(f"Error: JSONL file not found at {args.jsonl_path}")
        return

    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    my_lines = all_lines[rank::world_size]
    
    if rank == 0:
        print(f"Total samples: {len(all_lines)}. World Size: {world_size}.")
    print(f"[Rank {rank}] Assigned {len(my_lines)} samples.")


    seed_everything(args.seed) 

    iter_wrapper = tqdm(my_lines, desc=f"Rank {rank}", position=rank) if rank == 0 else my_lines

    for line in iter_wrapper:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        file_name = item.get("file_name")
        if not file_name:
            continue
            
        save_path_raw = os.path.join(raw_dir, file_name)
        if os.path.exists(save_path_raw):
            continue

        raw_captions = item.get("caption", [])
        raw_boxes = item.get("bndboxes", [])

        if not raw_captions or len(raw_captions) < 1:
            continue

        prompt_text = raw_captions[0]
        phrases_list = raw_captions[1:] 
        
        valid_boxes = []
        valid_phrases = []
        min_len = min(len(raw_boxes), len(phrases_list))
        
        for i in range(min_len):
            box = raw_boxes[i]
            phrase = phrases_list[i]
            if all(c == 0 for c in box): continue
            if not phrase or not isinstance(phrase, str) or phrase.strip() == "": continue
            valid_boxes.append(box)
            valid_phrases.append(phrase)

        current_prompt_final = [[prompt_text] + valid_phrases]
        current_bboxes = [valid_boxes]

        try:
            output = pipe(
                prompt=current_prompt_final, 
                bboxes=current_bboxes, 
                num_inference_steps=args.num_inference_steps, 
                guidance_scale=args.guidance_scale, 
                MIGCsteps=args.migc_steps, 
                aug_phase_with_and=False, 
                negative_prompt='worst quality, low quality, bad anatomy, watermark, text, blurry'
            )
            image = output.images[0]
        except Exception as e:
            print(f"[Rank {rank}] Inference failed for {file_name}: {e}")
            continue

        image.save(save_path_raw)

        gt_image_path = os.path.join(args.base_path, file_name)
        
        if os.path.exists(gt_image_path):
            try:
                gt_img = Image.open(gt_image_path).convert("RGB")
                gt_img = gt_img.resize(image.size)
                
                concat_img = Image.new('RGB', (gt_img.width + image.width, image.height))
                concat_img.paste(gt_img, (0, 0))
                concat_img.paste(image, (gt_img.width, 0))
                
                concat_img.save(os.path.join(concat_dir, f'concat_{os.path.basename(file_name)}'))
            except Exception as e:
                pass

        image_anno = image.copy()
        try:
            image_anno = draw_box_desc_standalone(image_anno, valid_boxes, valid_phrases, args.font_path)
            image_anno.save(os.path.join(anno_dir, f'anno_{os.path.basename(file_name)}'))
        except Exception as e:
            pass

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    
    if rank == 0:
        print("All processing done.")

if __name__ == "__main__":
    main()
