from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline, AutoencoderKL, ControlNetModel, ControlNetXSModel
from diffusers.loaders import AttnProcsLayers
import os, json, random
import argparse
from PIL import Image, ImageDraw, ImageFont,ImageFilter
import numpy as np
import cv2
import torch
from safetensors.torch import load_file
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import torch.distributed as dist


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
import inspect

from ccdiff.attention_processor import set_processors
from ccdiff.dota_utils import dict_of_images, find_nearest, get_similar_examplers, seed_everything
from ccdiff.projection import Resampler, SerialSampler
from ccdiff.modules import FrozenDinoV2Encoder

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
        parent_init_signature = inspect.signature(super().__init__)
        parent_init_params = parent_init_signature.parameters 
        

        init_kwargs = {
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            "scheduler": scheduler,
            "safety_checker": safety_checker,
            "feature_extractor": feature_extractor,
            "requires_safety_checker": requires_safety_checker
        }

        if 'image_encoder' in parent_init_params:
            init_kwargs['image_encoder'] = image_encoder
            
        super().__init__(**init_kwargs)

        self.text_projector = CLIPTextModelWithProjection.from_pretrained('/data')
        self.image_encoder = FrozenDinoV2Encoder() 
        self.image_proj_model = SerialSampler(
            dim=1280,
            depth=4,
            dim_head=64,
            num_queries=[16, 8, 8],
            embedding_dim=self.image_encoder.model.embed_dim,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4,
        )
  

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

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
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
        # negative_prompt_embeds: (prompt_nums[0]+prompt_nums[1]+...prompt_nums[n], token_num, token_channel), <class 'torch.Tensor'>

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                negative_prompt = "worst quality, low quality, bad anatomy"
            uncond_tokens = [negative_prompt] * batch_size

            # textual inversion: procecss multi-vector tokens if necessary
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
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
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
            # negative_prompt_embeds: (len(prompt_nums), token_num, token_channel), <class 'torch.Tensor'>

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            final_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return final_prompt_embeds, prompt_embeds, embeds_pooler[:, None, :], text_embeds

 
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


    @torch.no_grad()
    def __call__(
            self,
            prompt: List[List[str]] = None,
            controlnet: Optional[ControlNetModel] = None, 
            obboxes: List[List[List[float]]] = None,
            hed_image: Optional[Image.Image] = None,
            bboxes: List[List[List[float]]] = None,
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
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            token_indices (Union[List[List[List[int]]], List[List[int]]], optional):
                The list of the indexes in the prompt to layout. Defaults to None.
            bboxes (Union[List[List[List[float]]], List[List[float]]], optional):
                The bounding boxes of the indexes to maintain layout in the image. Defaults to None.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            max_guidance_iter (`int`, *optional*, defaults to `10`):
                The maximum number of iterations for the layout guidance on attention maps in diffusion mode.
            max_guidance_iter_per_step (`int`, *optional*, defaults to `5`):
                The maximum number of iterations to run during each time step for layout guidance.
            scale_factor (`int`, *optional*, defaults to `50`):
                The scale factor used to update the latents during optimization.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        controlnet_conditioning_image = None

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
        ###
        if isinstance(device, str):
            device = torch.device(device)
        self.image_encoder.to(device)
        self.text_projector.to(device)
        self.image_proj_model.to(device)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0


        prompt_embeds, cond_prompt_embeds, embeds_pooler, text_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        # print(prompt_embeds.shape)  3 77 768

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

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        
        
        ref_imgs = []
        for index, (caption, bndboxes) in enumerate(zip(prompt, bboxes)):
            categories = caption[1:]
            instances = []
            instance_imgs = []
            for name,bbox in zip(categories, bndboxes):
                if name == '':
                    instances.append(torch.zeros([3, 224, 224]))
                else:                   
                    value = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) 
                    chosen_file = list(dict_of_images[name].keys())[find_nearest(list(dict_of_images[name].values()), value)]                    
                    img = Image.open(os.path.join('/data/foreground', name, chosen_file))   
                    instance_imgs.append(img)                
                    img = self.feature_extractor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
                    instances.append(img)
            ref_imgs.append(torch.stack([instance for instance in instances])) 
           
        ref_imgs = torch.stack([img for img in ref_imgs]).view(len(obboxes) * len(obboxes[0]), 3, 224, 224).to(device)
        with torch.no_grad():
            img_features = self.image_encoder(ref_imgs)
            bg_img = get_similar_examplers(None, text_embeds[0], topk=1, sim_mode='text2img')
            bg_img = Image.open(os.path.join('/data/filter_train/images', bg_img[0])).convert('RGB')
            bg_features = self.feature_extractor(images=bg_img, return_tensors="pt")['pixel_values'].to(img_features.device)
            bg_features = self.image_encoder(bg_features)
        img_features, bg_features = self.image_proj_model(img_features, obboxes, bg_features)
        edge_maps = None
        if hed_image is not None:
            edge_maps = hed_image
            edge_maps = TF.to_tensor(hed_image)
            edge_maps = edge_maps.unsqueeze(0)
            if edge_maps.shape[-2] != 512 or edge_maps.shape[-1] != 512:
                edge_maps = F.interpolate(edge_maps, size=(512, 512), mode='bilinear', align_corners=False)
            # # Apply mask 
            mask_size = (512, 512)  # img size, specified 512
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

        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if GUI_progress is not None:
                    GUI_progress[0] = int((i + 1) / len(timesteps) * 100)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                
                # predict the noise residual
                # controlnet_cross_attention_kwargs = {
                #     "context": {
                #         "c_crossattn": [cnet_text_embeds]
                #     }
                # }
                cross_attention_kwargs = {'bboxes': bboxes,
                                          'obboxes': obboxes,                                         
                                          'embeds_pooler': embeds_pooler,
                                          'height': height,
                                          'width': width,
                                          'ref_features': (img_features, bg_features),
                                          'do_classifier_free_guidance': do_classifier_free_guidance,
                                          }
                self.unet.eval()
                if controlnet is not None and controlnet_conditioning_image is not None:
                   
                    
                    mask_512 = self.rasterize_union_mask_from_obboxes(obboxes, H=height, W=width, device=latents.device)  # (B,1,512,512)

                    noise_pred = controlnet(
                        self.unet,
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds[0:2],   #batch size 1 
                        base_encoder_hidden_states=prompt_embeds, 
                        gating_mask=mask_512,
                        controlnet_cond=controlnet_conditioning_image,
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

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )

                step_output = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )
                latents = step_output.prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=None
        )


def parse_args():
    #CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 infer_dota.py
    parser = argparse.ArgumentParser(description="Distributed Inference Script")
    parser.add_argument("--sd1x_path", type=str, default="/data/model/sd1-4")
    parser.add_argument("--save_path", type=str, default="/data/checkpoint-dota")
    parser.add_argument("--data_path", type=str, default="/data/images")
    parser.add_argument("--edge_path", type=str, default="/data/edge")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--controlnet_path", type=str, default="/data/controlnet/diffusion_pytorch_model.safetensors")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative_prompt", type=str, default="worst quality, low quality, bad anatomy, watermark, text, blurry")
    return parser.parse_args()

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

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    args = parse_args()

    pipe = StableDiffusionMIPipeline.from_pretrained(args.sd1x_path)
    
    set_processors(pipe.unet)
    custom_layers = AttnProcsLayers(pipe.unet.attn_processors)
    
    state_dict_path = os.path.join(args.save_path, 'unet/diffusion_pytorch_model.safetensors')
    proj_model_path = os.path.join(args.save_path, 'ImageProjModel.pth')
    
    if os.path.exists(state_dict_path):
        state_dict = {k: v for k, v in load_file(state_dict_path).items() if '.processor' in k or '.self_attn' in k}
        custom_layers.load_state_dict(state_dict)
    else:
        print(f"Warning: State dict not found at {state_dict_path}")

    if os.path.exists(proj_model_path):
        pipe.image_proj_model.load_state_dict(torch.load(proj_model_path, map_location="cpu"))
    else:
        print(f"Warning: Projection model not found at {proj_model_path}")

    controlnet = ControlNetXSModel.init_original(pipe.unet, is_sdxl=False)
    
    controlnet_state_dict = load_file(args.controlnet_path)
    controlnet.load_state_dict(controlnet_state_dict)
    set_safe_processors_for_controlnet(controlnet)
    controlnet = controlnet.to(device)
    pipe = pipe.to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    data = []
    metadata_path = os.path.join(args.data_path, 'metadata.jsonl')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        print(f"Error: Metadata file not found at {metadata_path}")
        exit(1)
    
    my_data = data[rank::world_size]
    if rank == 0:
        print(f"Total samples: {len(data)}")
    print(f"Rank {rank} processing {len(my_data)} samples.")

    seed_everything(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed) #42
    
    raw_dir = os.path.join(args.out_dir, "raw")
    concat_dir = os.path.join(args.out_dir, "concat")
    anno_dir = os.path.join(args.out_dir, "annotated")
    
    if rank == 0:
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(concat_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()

    for i in range(len(my_data)):
        sample = my_data[i]
        file_name = sample['file_name']
        edge_file_path = os.path.join(args.edge_path, file_name)
        
        
        hed_source_image = None
        if os.path.exists(edge_file_path):
            hed_source_image = Image.open(edge_file_path).convert("RGB")
        else:
             print(f"Rank {rank}: Edge image not found for {file_name}, proceeding without it.")
        print(f"Rank {rank} generating: {file_name}")
     
        gt_img_path = os.path.join(args.data_path, file_name)

        gt_img = Image.open(gt_img_path).convert('RGB')

        prompt = [sample['caption']]
        obboxes = [sample['obboxes']]
        bboxes = [sample['bndboxes']]
        
        out = pipe(
            prompt=prompt,
            controlnet=controlnet,
            obboxes=obboxes, 
            hed_image=hed_source_image,
            bboxes=bboxes,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            negative_prompt=args.negative_prompt,
            generator=generator
        )
        image = out.images[0]
        
        image.save(os.path.join(raw_dir, file_name))
        
        gt_img = gt_img.resize(image.size)
        result = Image.new('RGB', (gt_img.width + image.width, image.height))
        result.paste(gt_img, (0, 0))
        result.paste(image, (image.width, 0))
        result.save(os.path.join(concat_dir, f'output_{file_name}'))
        
        image_anno = image.copy()
        # bs1 prompt 'global prompt, class, class ...'
        image_anno = pipe.draw_box_desc(image_anno, obboxes[0], prompt[0][1:])
        image_anno.save(os.path.join(anno_dir, f'anno_output_{file_name}'))

    if world_size > 1:
        dist.destroy_process_group()
    
