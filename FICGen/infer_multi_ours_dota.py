from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline, AutoencoderKL,ControlNetModel, ControlNetXSModel
from diffusers.loaders import AttnProcsLayers
import os, json, random
from PIL import Image, ImageDraw, ImageFont,ImageFilter
import argparse
import torch
import cv2
import functools
import imagesize
import numpy as np
from safetensors.torch import load_file
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor, Attention
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    CLIPImageProcessor, 
    CLIPTextModel, 
    CLIPTokenizer, 
    CLIPVisionModelWithProjection, 
    CLIPTextModelWithProjection, 
    AutoProcessor,
)
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F
import inspect
from src.attention_processor import set_processors
from src.utils import get_classnames, find_nearest, get_similar_examplers, seed_everything
from src.projection import Resampler, SerialSampler
import torch.distributed as dist

logger = logging.get_logger(__name__)
class SafePassthroughAttnProcessor(AttnProcessor):
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask)
    
def set_safe_processors_for_controlnet(controlnet_model):
    safe_processor = SafePassthroughAttnProcessor()
    for module in controlnet_model.modules():
        if isinstance(module, Attention):
            module.set_processor(safe_processor)
    logger.info("Successfully installed SafePassthroughAttnProcessor on all attention modules of ControlNet-XS.")
class StableDiffusionMIPipeline(StableDiffusionPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = True,
    ):
        # Get the parameter signature of the parent class constructor
        parent_init_signature = inspect.signature(super().__init__)
        parent_init_params = parent_init_signature.parameters 
        self.text_projector = CLIPTextModelWithProjection.from_pretrained('/data')
        self.image_proj_model = SerialSampler(
            dim=1280,
            depth=4,
            dim_head=64,
            # heads=20,
            num_queries=[16, 8, 8],
            embedding_dim=1024,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4,
        )
        
        # Dynamically build a parameter dictionary based on the parameters of the parent class constructor
        init_kwargs = {
            # "image_encoder": image_encoder,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            "scheduler": scheduler,
            "safety_checker": safety_checker,
            "feature_extractor": feature_extractor,
            "requires_safety_checker": requires_safety_checker
        }
        if 'image_encoder' in parent_init_params.items():
            init_kwargs['image_encoder'] = image_encoder
        super().__init__(**init_kwargs)
        

    def _encode_prompt(
            self,
            prompts,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        """
        if prompts is not None and isinstance(prompts, str):
            batch_size = 1
        elif prompts is not None and isinstance(prompts, list):
            batch_size = len(prompts)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds_none_flag = (prompt_embeds is None)
        prompt_embeds_list = []
        embeds_pooler_list = []
        text_embeds_list = []
        
        for prompt in prompts:
            if prompt_embeds_none_flag:
                # textual inversion: procecss multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(
                    prompt, padding="longest", return_tensors="pt"
                ).input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[
                    -1
                ] and not torch.equal(text_input_ids, untruncated_ids):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )

                if (
                        hasattr(self.text_encoder.config, "use_attention_mask")
                        and self.text_encoder.config.use_attention_mask
                ):
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                )
                embeds_pooler = prompt_embeds.pooler_output
                prompt_embeds = prompt_embeds[0]
                text_embeds = self.text_projector(text_input_ids.to(device)).text_embeds

            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            embeds_pooler = embeds_pooler.to(dtype=self.text_encoder.dtype, device=device)
            text_embeds = text_embeds.to(dtype=self.text_encoder.dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            embeds_pooler = embeds_pooler.repeat(1, num_images_per_prompt)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )
            embeds_pooler = embeds_pooler.view(
                bs_embed * num_images_per_prompt, -1
            )
            prompt_embeds_list.append(prompt_embeds)
            embeds_pooler_list.append(embeds_pooler)
            text_embeds_list.append(text_embeds)
        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        embeds_pooler = torch.cat(embeds_pooler_list, dim=0)
        text_embeds = torch.cat(text_embeds_list, dim=0)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                negative_prompt = "worst quality, low quality, bad anatomy"
            uncond_tokens = [negative_prompt] * batch_size

            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            final_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return final_prompt_embeds, prompt_embeds, embeds_pooler[:, None, :], text_embeds

    @staticmethod
    def rasterize_union_mask_from_bboxes(bboxes, H, W, device, blur_sigma=2.0):
        B = len(bboxes)
        out = torch.zeros(B, 1, H, W, device=device, dtype=torch.float32)
        for b, boxes in enumerate(bboxes):
            m = np.zeros((H, W), dtype=np.float32)
            for box in boxes:
                if len(box) != 4 or np.count_nonzero(box) == 0: 
                    continue
                x_min, y_min, x_max, y_max = box
                pt1 = (int(x_min * W), int(y_min * H))
                pt2 = (int(x_max * W), int(y_max * H))
                cv2.rectangle(m, pt1, pt2, color=1.0, thickness=-1)

            if blur_sigma and blur_sigma > 0:
                m = cv2.GaussianBlur(m, (0, 0), blur_sigma)

                if m.max() > 0:
                    m /= m.max()
            out[b, 0] = torch.from_numpy(m)
        return out.clamp_(0, 1)
    

    @staticmethod
    def rasterize_union_mask_from_obboxes(obboxes, H, W, device, dilate_px=4, blur_sigma=2.0):
        B = len(obboxes)
        out = torch.zeros(B, 1, H, W, device=device, dtype=torch.float32)
        for b, polys in enumerate(obboxes):
            m = np.zeros((H, W), dtype=np.float32)
            for ob in polys:
                if np.count_nonzero(ob) == 0: 
                    continue
                pts = np.array(ob, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= W; pts[:, 1] *= H
                cv2.fillPoly(m, [pts.astype(np.int32)], 1.0)
            # if dilate_px > 0:
            #     k = 2 * dilate_px + 1
            #     m = cv2.dilate(m, np.ones((k, k), np.uint8))
            if blur_sigma and blur_sigma > 0:
                m = cv2.GaussianBlur(m, (0, 0), blur_sigma); m /= max(m.max(), 1e-6)
            out[b, 0] = torch.from_numpy(m)
        return out.clamp_(0, 1)
    


    
    #obbox
    @staticmethod
    def draw_box_desc(pil_img: Image, bboxes: List[List[float]], prompt: List[str]) -> Image:
        """Utility function to draw bbox on the image"""
        color_list = ['red', 'blue', 'yellow', 'purple', 'green', 'black', 'brown', 'orange', 'white', 'gray']
        width, height = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        font_folder = os.path.dirname(os.path.dirname(__file__))
        font_path = os.path.join('./Rainbow-Party-2.ttf')
        font = ImageFont.truetype(font_path, 30)

        for box_id in range(len(bboxes)):
            obj_box = bboxes[box_id]
            text = prompt[box_id]
            fill = 'black'
            for color in prompt[box_id].split(' '):
                if color in color_list:
                    fill = color
            text = text.split(',')[0]
            
            x_min, y_min = (
                obj_box[0] * width,
                obj_box[1] * height,
            )
            draw.polygon(
                [(x*width, y*height) for x, y in zip(obj_box[::2], obj_box[1::2])],
                outline=fill,
                width=4,
            )
            draw.text((int(x_min), int(y_min)), text, fill=fill, font=font)

        return pil_img


    @torch.no_grad()
    def __call__(
            self,
            prompt: List[List[str]] = None,
            obboxes: List[List[List[float]]] = None,
            bboxes: List[List[List[float]]] = None,
            controlnet: Optional[ControlNetModel] = None, 
            hed_image: Optional[Image.Image] = None,
            img_patch_path: List[List[str]] = None,
            bg_path: List[List[str]] = None,
            data_path: List[List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            GUI_progress=None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        controlnet_conditioning_image=None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_nums = [0] * len(prompt)
        for i, _ in enumerate(prompt):
            prompt_nums[i] = len(_)

        device = self._execution_device
        self.text_projector.to(device)
        self.image_proj_model.to(device)
        
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, cond_prompt_embeds, embeds_pooler, text_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
    
        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        instance_transforms = transforms.Compose(
            [
                transforms.Resize([256, 256], interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5], [0.5]
                ),
            ]
        )
        
        dict_of_images = {}
        list_of_name, data_emb_dict = get_classnames(data_path)

        for name in list_of_name:
            name_of_dir = os.path.join(img_patch_path, name)
            if not os.path.exists(name_of_dir):
                print(f"Warning: Directory {name_of_dir} does not exist. Skipping.")
                continue
            list_of_image = os.listdir(name_of_dir)
            list_of_image = [i for i in list_of_image if i.endswith("jpg")]
            list_of_image = sorted(list_of_image, 
                                   key = lambda img: functools.reduce(lambda x, y: x*y, 
                                                                      imagesize.get(os.path.join(name_of_dir, img))
                                                                    ), reverse=True)
            dict_of_images[name] = {img: functools.reduce(lambda x, y: x/y, imagesize.get(os.path.join(name_of_dir, img))) for img in list_of_image[:200]}

        
        ref_imgs = []
        for index, (caption, bndboxes) in enumerate(zip(prompt, bboxes)):
            categories = caption[1:]
            instances = []
            instance_imgs = []
            for name,bbox in zip(categories, bndboxes):
                if name == '' or name not in dict_of_images: # Handle missing keys
                    instances.append(torch.zeros([3, 256, 256]))
                else:                   
                    try:
                        value = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])    # foreground size based on HBB
                    except:
                        print(bbox)
                        value = 1.0 # default fallback
                    
                    if len(dict_of_images[name]) > 0:
                         chosen_file = list(dict_of_images[name].keys())[find_nearest(list(dict_of_images[name].values()), value)]                    
                         img = Image.open(os.path.join(img_patch_path, name, chosen_file)).convert("RGB")
                         img = instance_transforms(img)
                         instances.append(img)
                    else:
                        instances.append(torch.zeros([3, 256, 256]))

            ref_imgs.append(torch.stack([instance for instance in instances])) 
           
        ref_imgs = torch.stack([img for img in ref_imgs]).view(len(obboxes) * len(obboxes[0]), 3, 256, 256).to(device)
        with torch.no_grad():
            img_features = ref_imgs
            bg_img = get_similar_examplers(data_emb_dict, None, text_embeds[0], topk=1, sim_mode='text2img')
            file_name = os.path.join(bg_path, bg_img[0])
            bg_img = Image.open(file_name).convert('RGB')
            bg_img = instance_transforms(bg_img).to(img_features.device)
            bg_features = bg_img.unsqueeze(0)
        img_features, bg_features = self.image_proj_model(img_features, obboxes, bg_features)
        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        #cnet
        if hed_image is not None:

            edge_maps = hed_image
            edge_maps = TF.to_tensor(hed_image)

            edge_maps = edge_maps.unsqueeze(0)
            if edge_maps.shape[-2] != 512 or edge_maps.shape[-1] != 512:
                edge_maps = F.interpolate(edge_maps, size=(512, 512), mode='bilinear', align_corners=False)

            mask_size = (512, 512) 
            mask_img = Image.new("L", mask_size, 0)
            draw = ImageDraw.Draw(mask_img)
           
            current_obboxes = obboxes[0] if obboxes is not None else [] # bs 1
    
            for ob in current_obboxes:
                pts_norm = np.array(ob).reshape(-1, 2)
                    
                points = []
                for pt in pts_norm:
                    px = pt[0] * mask_size[0]
                    py = pt[1] * mask_size[1]
                    points.append((px, py))
                     
                draw.polygon(points, fill=255)

            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))

            mask_array = np.array(mask_img) / 255.0
            mask_tensor = torch.from_numpy(mask_array).to(device=edge_maps.device, dtype=edge_maps.dtype)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
            
            edge_maps = edge_maps * mask_tensor

            # ###vis
            # edge_maps_np = edge_maps.squeeze().cpu().numpy()
            # if edge_maps_np.shape[0] in [1, 3]:
            #     edge_maps_np = edge_maps_np.transpose(1, 2, 0)
            # if edge_maps_np.ndim == 3 and edge_maps_np.shape[-1] == 1:
            #     edge_maps_np = edge_maps_np.squeeze(-1)
            # edge_maps_np = (edge_maps_np * 255).astype(np.uint8)
            # edge_map_img = Image.fromarray(edge_maps_np)
            # # Save the edge map image
            # edge_map_save_path = os.path.join(args.out_dir, f"edge_map_{file_name}")
            # edge_map_img.save(edge_map_save_path)
            # print(f"Saved masked edge map to {edge_map_save_path}")
            controlnet_conditioning_image = edge_maps.to(device=device, dtype=prompt_embeds.dtype)

                # ###vis
                # edge_maps_np = edge_maps.squeeze().cpu().numpy()
                # if edge_maps_np.shape[0] in [1, 3]:
                #     edge_maps_np = edge_maps_np.transpose(1, 2, 0)
                # if edge_maps_np.ndim == 3 and edge_maps_np.shape[-1] == 1:
                #     edge_maps_np = edge_maps_np.squeeze(-1)
                # edge_maps_np = (edge_maps_np * 255).astype(np.uint8)
                # edge_map_img = Image.fromarray(edge_maps_np)
                # # Save the edge map image
                # edge_map_save_path = os.path.join(args.out_dir, f"edge_map_{file_name}")
                # edge_map_img.save(edge_map_save_path)
                # print(f"Saved masked edge map to {edge_map_save_path}")
            # controlnet_conditioning_image = edge_maps.to(device=device, dtype=prompt_embeds.dtype)


        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if GUI_progress is not None:
                    GUI_progress[0] = int((i + 1) / len(timesteps) * 100)
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                
                cross_attention_kwargs = {'bboxes': bboxes,
                                          'obboxes': obboxes,                                         
                                          'embeds_pooler': embeds_pooler,
                                          'height': height,
                                          'width': width,
                                          'ref_features': (img_features, bg_features),
                                          'do_classifier_free_guidance': do_classifier_free_guidance,
                                          }
                if controlnet is not None and controlnet_conditioning_image is not None:
                    
                    cnet_cond_image = controlnet_conditioning_image

                    mask_512 = self.rasterize_union_mask_from_obboxes(obboxes, H=height, W=width, device=latents.device)  # (B,1,512,512)
                    noise_pred = controlnet(
                        self.unet,
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds[0:2],   #batch size 1 
                        base_encoder_hidden_states=prompt_embeds, 
                        gating_mask=mask_512,
                        controlnet_cond=cnet_cond_image,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )
                else:
                    # Fallback to standard U-Net if ControlNet is not used
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample


                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )

                step_output = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )
                latents = step_output.prev_sample

                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=None
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU Inference Script")
    parser.add_argument("--img_patch_path", type=str, default="/data/img_patch")
    parser.add_argument("--bg_path", type=str, default="/data/filter_train/images")
    parser.add_argument("--edge_path", type=str, default="/data/edge")
    parser.add_argument("--out_dir", type=str, default="/data/out")


    # parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()
    return args

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"Initialized Process Group: Rank {rank}/{world_size}, Local Rank {local_rank}")
        return rank, world_size, local_rank
    else:
        print("Not using distributed mode.")
        return 0, 1, 0

if __name__ == '__main__':
    # 1. 设置分布式环境
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    sd1x_path = '/data/model/sd1-5'
    save_path = '/data/checkpoint' #epoch 300
    data_path = '/data/images'

    pipe = StableDiffusionMIPipeline.from_pretrained(
        sd1x_path)
    
    args = parse_args()
    
    set_processors(pipe.unet)
    custom_layers = AttnProcsLayers(pipe.unet.attn_processors)
    
    state_dict_path = os.path.join(save_path, 'unet/diffusion_pytorch_model.safetensors')
    proj_model_path = os.path.join(save_path, 'ImageProjModel.pth')
    
    state_dict = {k: v for k, v in load_file(state_dict_path).items() if '.processor' in k or '.self_attn' in k}
    custom_layers.load_state_dict(state_dict)
    pipe.image_proj_model.load_state_dict(torch.load(proj_model_path, map_location=device)) 
    controlnet = ControlNetXSModel.init_original(pipe.unet, is_sdxl=False)

    controlnet_weights_path = os.path.join('/data/controlnet/diffusion_pytorch_model.safetensors')
    
    controlnet_state_dict = load_file(controlnet_weights_path)
    controlnet.load_state_dict(controlnet_state_dict)
    
    set_safe_processors_for_controlnet(controlnet)
    controlnet = controlnet.to(f"cuda:{local_rank}")
    pipe = pipe.to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    data = []
    with open(os.path.join(data_path, 'metadata.jsonl'), 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    my_data = data[rank::world_size]
    print(f"Rank {rank} processing {len(my_data)} samples out of {len(data)} total.")

    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    
    seed = 42 
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    raw_dir = os.path.join(args.out_dir, "raw")       
    concat_dir = os.path.join(args.out_dir, "concat")  
    anno_dir = os.path.join(args.out_dir, "annotated") 
    
    if rank == 0:
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(concat_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()

    sample_num = len(my_data)
    
    for i in range(sample_num):
        sample = my_data[i]
        file_name = sample['file_name']
        edge_path = os.path.join(args.edge_path, sample["file_name"])
        hed_source_image = Image.open(edge_path).convert("RGB")
        print(f"Rank {rank} generating: {file_name}")
     
        gt_img_path = os.path.join(data_path, file_name.replace("_gen", ""))
        try:
            gt_img = Image.open(gt_img_path).convert('RGB')
        except FileNotFoundError:
            print(f"GT Image not found: {gt_img_path}, creating dummy.")
            gt_img = Image.new('RGB', (512, 512), (0,0,0))

        prompt = [sample['caption']]
        obboxes = [sample['obboxes']]
        bboxes = [sample['bndboxes']]
        
        image = pipe(
            prompt, obboxes, bboxes, 
            controlnet,
            # None,    ### plug in
            hed_source_image,
            args.img_patch_path, 
            args.bg_path, 
            data_path, 
            num_inference_steps=50, 
            guidance_scale=7.5, 
            negative_prompt=negative_prompt,
            generator=generator 
        ).images[0]
        
        image.save(os.path.join(raw_dir, file_name))
        
        gt_img = gt_img.resize(image.size)
        result = Image.new('RGB', (gt_img.width + image.width, image.height))
        result.paste(gt_img, (0, 0))
        result.paste(image, (image.width, 0))
        result.save(os.path.join(concat_dir, f'output_{file_name}'))

        image_anno = image.copy()
        image_anno = pipe.draw_box_desc(image_anno, obboxes[0], prompt[0][1:])
        image_anno.save(os.path.join(anno_dir, f'anno_output_{file_name}'))

    if world_size > 1:
        dist.destroy_process_group()

