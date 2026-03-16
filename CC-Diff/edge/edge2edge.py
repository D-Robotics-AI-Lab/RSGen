import torch
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler
from .pipeline_stable_diffusion_xl_opt import StableDiffusionXLPipeline
from .injection_utils import register_attention_editor_diffusers
from .region_attention import Edge2Edge
from pytorch_lightning import seed_everything

class SDXLEdge2Edge:
    def __init__(self, model_path, lora_path, device, dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        
        print(f"[SDXLEdge2Edge] Loading SDXL from {model_path}...")
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, 
                                  beta_schedule="scaled_linear", clip_sample=False, 
                                  set_alpha_to_one=False)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path, scheduler=scheduler, torch_dtype=dtype
        ).to(device)

        if lora_path:
            print(f"[SDXLEdge2Edge] Loading LoRA from {lora_path}...")
            self.pipe.load_lora_weights(lora_path)
            self.pipe.fuse_lora(lora_scale=1.0)
        
        self.pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.enable_model_cpu_offload() 

    def get_token_indices(self, prompt, class_names):
        tokenizer = self.pipe.tokenizer
        max_len = getattr(tokenizer, "model_max_length", 77)
        token_ids = tokenizer.encode(prompt, truncation=True, max_length=max_len)
        
        indices = []
        current_search_start = 0
        
        for name in class_names:
            name = name.strip()
            if not name:
                indices.append([])
                continue
            candidates = [name, name.lower()]
            target_ids = None
            for cand in candidates:
                ids = tokenizer.encode(cand, add_special_tokens=False)
                if len(ids) > 0:
                    target_ids = ids
                    break

            if target_ids is None or len(target_ids) == 0:
                indices.append([])
                continue

            target_len = len(target_ids)
            found = False
            for i in range(current_search_start, len(token_ids) - target_len + 1):
                if token_ids[i:i+target_len] == target_ids:
                    indices.append(list(range(i, i+target_len)))
                    current_search_start = i + target_len 
                    found = True
                    break
            
            if not found:
                indices.append([])
        return indices

    def refine(self, 
               init_image: Image.Image, 
               hbboxes: list,
               obboxes: list, 
               class_names: list,
               caption: str,
               strength: float = 0.7,
               num_inference_steps: int = 50,
               guidance_scale: float = 7.5,
               seed: int = 42,
               num_guidance_steps: int = 15):
        seed_everything(seed)
        
        scene_description = caption.strip()
        valid_names = [n for n in class_names if n.strip()]
        objects_str = ", ".join(valid_names)
        prompt = f"hed edge map. {objects_str}."
        
        subject_token_indices = self.get_token_indices(prompt, valid_names)
        
        prompts = [prompt]
        
        editor = BoundedAttention(
            hbboxes,
            obboxes,
            prompts,
            subject_token_indices,
            list(range(70, 82)), 
            list(range(70, 82)),
            filter_token_indices=None,
            eos_token_index=None,
            cross_loss_coef=1.5,
            self_loss_coef=0.5,
            max_guidance_iter=num_guidance_steps,
            max_guidance_iter_per_step=5,
            start_step_size=8,
            end_step_size=2,
            loss_stopping_value=0.02,
            min_clustering_step=15,
            num_clusters_per_box=3,
        )

        register_attention_editor_diffusers(self.pipe, editor)
        original_size = init_image.size
        target_size = (1024, 1024)
        init_image_resized = init_image.resize(target_size, Image.BILINEAR)
        # save_path = "debug_init_image_resized.png"
        # init_image_resized.save(save_path)

        images = self.pipe(
            prompts, 
            image=init_image_resized, 
            strength=strength, 
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images
        
        result_image = images[0]
        
        result_image = result_image.resize(original_size, Image.BILINEAR)
        
        return result_image
