import inspect
import warnings
import os
import json
from typing import Any, Callable, Dict, List, Optional, Union
import argparse
import cv2
import random
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = False
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from safetensors.torch import load_file

import PIL.Image
from PIL import Image, ImageDraw, ImageFont,ImageFilter
from tqdm import tqdm
from scipy.ndimage import uniform_filter, gaussian_filter
import torch
from transformers import (
    CLIPFeatureExtractor,
    CLIPImageProcessor,
    CLIPProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)

from diffusers import EulerDiscreteScheduler, ControlNetModel, ControlNetXSModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import GatedSelfAttentionDense
from diffusers.models.attention_processor import AttnProcessor, Attention
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import USE_PEFT_BACKEND, logging, replace_example_docstring, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
# from diffusers.pipelines.stable_diffusion.clip_image_project_model import CLIPImageProjection
from migc.migc_pipeline import  MIGCProcessor, AttentionStore
from migc.migc_utils import seed_everything, load_migc




logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



class SafePassthroughAttnProcessor(AttnProcessor):
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask)
    
def set_safe_processors_for_controlnet(controlnet_model):

    safe_processor = SafePassthroughAttnProcessor()
    for module in controlnet_model.modules():
        if isinstance(module, Attention):
            module.set_processor(safe_processor)
    logger.info("Successfully installed SafePassthroughAttnProcessor on all attention modules of ControlNet-XS.")


                

class StableDiffusionMIGCPipeline(StableDiffusionPipeline):
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
        
        # Dynamically build a parameter dictionary based on the parameters of the parent class constructor
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
        if 'image_encoder' in parent_init_params.items():
            init_kwargs['image_encoder'] = image_encoder
        super().__init__(**init_kwargs)
        
        self.instance_set = set()
        self.embedding = {}

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

            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            embeds_pooler = embeds_pooler.to(dtype=self.text_encoder.dtype, device=device)

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
        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        embeds_pooler = torch.cat(embeds_pooler_list, dim=0)
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

        return final_prompt_embeds, prompt_embeds, embeds_pooler[:, None, :]

    def check_inputs(
            self,
            prompt,
            token_indices,
            bboxes,
            height,
            width,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
                callback_steps is not None
                and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
                not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if token_indices is not None:
            if isinstance(token_indices, list):
                if isinstance(token_indices[0], list):
                    if isinstance(token_indices[0][0], list):
                        token_indices_batch_size = len(token_indices)
                    elif isinstance(token_indices[0][0], int):
                        token_indices_batch_size = 1
                    else:
                        raise TypeError(
                            "`token_indices` must be a list of lists of integers or a list of integers."
                        )
                else:
                    raise TypeError(
                        "`token_indices` must be a list of lists of integers or a list of integers."
                    )
            else:
                raise TypeError(
                    "`token_indices` must be a list of lists of integers or a list of integers."
                )

        if bboxes is not None:
            if isinstance(bboxes, list):
                if isinstance(bboxes[0], list):
                    if (
                            isinstance(bboxes[0][0], list)
                            and len(bboxes[0][0]) == 4
                            and all(isinstance(x, float) for x in bboxes[0][0])
                    ):
                        bboxes_batch_size = len(bboxes)
                    elif (
                            isinstance(bboxes[0], list)
                            and len(bboxes[0]) == 4
                            and all(isinstance(x, float) for x in bboxes[0])
                    ):
                        bboxes_batch_size = 1
                    else:
                        print(isinstance(bboxes[0], list), len(bboxes[0]))
                        raise TypeError(
                            "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                        )
                else:
                    print(isinstance(bboxes[0], list), len(bboxes[0]))
                    raise TypeError(
                        "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                    )
            else:
                print(isinstance(bboxes[0], list), len(bboxes[0]))
                raise TypeError(
                    "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                )

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if token_indices_batch_size != prompt_batch_size:
            raise ValueError(
                f"token indices batch size must be same as prompt batch size. token indices batch size: {token_indices_batch_size}, prompt batch size: {prompt_batch_size}"
            )

        if bboxes_batch_size != prompt_batch_size:
            raise ValueError(
                f"bbox batch size must be same as prompt batch size. bbox batch size: {bboxes_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def get_indices(self, prompt: str) -> Dict[str, int]:
        """Utility function to list the indices of the tokens you wish to alte"""
        ids = self.tokenizer(prompt).input_ids
        indices = {
            i: tok
            for tok, i in zip(
                self.tokenizer.convert_ids_to_tokens(ids), range(len(ids))
            )
        }
        return indices

    #hbbox
    @staticmethod
    def draw_box_desc(pil_img: Image, bboxes: List[List[float]], prompt: List[str]) -> Image:
        """Utility function to draw bbox on the image"""
        color_list = ['red', 'blue', 'yellow', 'purple', 'green', 'black', 'brown', 'orange', 'white', 'gray']
        width, height = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        font_folder = os.path.dirname(os.path.dirname(__file__))
        font_path = os.path.join('./Rainbow-Party-2.ttf')
        try:
            font = ImageFont.truetype(font_path, 30)
        except Exception:
            font = ImageFont.load_default()

        for box_id in range(len(bboxes)):
            obj_box = bboxes[box_id]
            text = prompt[box_id]
            fill = 'black'
            for color in prompt[box_id].split(' '):
                if color in color_list:
                    fill = color
            text = text.split(',')[0]

            x_min, y_min, x_max, y_max = (
                obj_box[0] * width,
                obj_box[1] * height,
                obj_box[2] * width,
                obj_box[3] * height,
            )
            draw.rectangle(
                [int(x_min), int(y_min), int(x_max), int(y_max)],
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
     

    @staticmethod
    def rasterize_union_mask_from_bboxes(bboxes, H, W, device, blur_sigma=2.0):
        """
        从 bboxes (轴对齐边界框) 创建一个联合的、平滑的掩码。
        """
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
    def draw_box_desc(pil_img: Image, bboxes: List[List[float]], prompt: List[str]) -> Image:
        """Utility function to draw bbox on the image"""
        color_list = ['red', 'blue', 'yellow', 'purple', 'green', 'black', 'brown', 'orange', 'white', 'gray']
        width, height = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        font_folder = os.path.dirname(os.path.dirname(__file__))
        font_path = os.path.join(font_folder, 'Rainbow-Party-2.ttf')
        font = ImageFont.truetype(font_path, 30)

        for box_id in range(len(bboxes)):
            obj_box = bboxes[box_id]
            text = prompt[box_id]
            fill = 'black'
            for color in prompt[box_id].split(' '):
                if color in color_list:
                    fill = color
            text = text.split(',')[0]
            x_min, y_min, x_max, y_max = (
                obj_box[0] * width,
                obj_box[1] * height,
                obj_box[2] * width,
                obj_box[3] * height,
            )
            draw.rectangle(
                [int(x_min), int(y_min), int(x_max), int(y_max)],
                outline=fill,
                width=4,
            )
            draw.text((int(x_min), int(y_min)), text, fill=fill, font=font)

        return pil_img


    @torch.no_grad()
    def __call__(
            self,
            prompt: List[List[str]] = None,
            controlnet: Optional[ControlNetModel] = None, 
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
            MIGCsteps=20,
            NaiveFuserSteps=-1,
            ca_scale=None,
            ea_scale=None,
            sac_scale=None,
            aug_phase_with_and=False,
            sa_preserve=False,
            use_sa_preserve=False,
            clear_set=False,
            GUI_progress=None
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
        def aug_phase_with_and_function(phase, instance_num):
            instance_num = min(instance_num, 7)
            copy_phase = [phase] * instance_num
            phase = ', and '.join(copy_phase)
            return phase

        if aug_phase_with_and:
            instance_num = len(prompt[0]) - 1
            for i in range(1, len(prompt[0])):
                prompt[0][i] = aug_phase_with_and_function(prompt[0][i],
                                                            instance_num)
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
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, cond_prompt_embeds, embeds_pooler = self._encode_prompt(
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

        #cnet
        edge_maps = None
        if hed_image is not None:
            edge_maps = hed_image
            edge_maps = TF.to_tensor(hed_image)

            edge_maps = edge_maps.unsqueeze(0)
            if edge_maps.shape[-2] != 512 or edge_maps.shape[-1] != 512:
                edge_maps = F.interpolate(edge_maps, size=(512, 512), mode='bilinear', align_corners=False)


            #hbbox
            mask_size = (512, 512) 
            mask_img = Image.new("L", mask_size, 0)
            draw = ImageDraw.Draw(mask_img)
            for bbox in bboxes[0]:  # batch_size=1
                x_min, y_min, x_max, y_max = bbox
                points = [
                    x_min * mask_size[0], 
                    y_min * mask_size[1], 
                    x_max * mask_size[0], 
                    y_max * mask_size[1]
                ]
                draw.rectangle(points, fill=255)

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
        
        if clear_set:
            self.instance_set = set()
            self.embedding = {}

        now_set = set()
        for i in range(len(bboxes[0])):
            now_set.add((tuple(bboxes[0][i]), prompt[0][i + 1]))

        mask_set = (now_set | self.instance_set) - (now_set & self.instance_set)
        self.instance_set = now_set

        guidance_mask = np.full((4, height // 8, width // 8), 1.0)
                
        for bbox, _ in mask_set:
            w_min = max(0, int(width * bbox[0] // 8) - 5)
            w_max = min(width, int(width * bbox[2] // 8) + 5)
            h_min = max(0, int(height * bbox[1] // 8) - 5)
            h_max = min(height, int(height * bbox[3] // 8) + 5)
            guidance_mask[:, h_min:h_max, w_min:w_max] = 0
        
        kernal_size = 5
        guidance_mask = uniform_filter(
            guidance_mask, axes = (1, 2), size = kernal_size
        )
        
        guidance_mask = torch.from_numpy(guidance_mask).to(self.device).unsqueeze(0)

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
                cross_attention_kwargs = {'prompt_nums': prompt_nums,
                                          'bboxes': bboxes,
                                          'ith': i,
                                          'embeds_pooler': embeds_pooler,
                                          'timestep': t,
                                          'height': height,
                                          'width': width,
                                          'MIGCsteps': MIGCsteps,
                                          'NaiveFuserSteps': NaiveFuserSteps,
                                          'ca_scale': ca_scale,
                                          'ea_scale': ea_scale,
                                          'sac_scale': sac_scale,
                                          'sa_preserve': sa_preserve,
                                          'use_sa_preserve': use_sa_preserve}
                
                self.unet.eval()
                mask_512 = self.rasterize_union_mask_from_bboxes(bboxes, H=height, W=width, device=latents.device)
                if controlnet is not None and controlnet_conditioning_image is not None:
                    noise_pred = controlnet(
                        self.unet,
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds[0:2],   #batch size 1 
                        base_encoder_hidden_states=prompt_embeds, # unet
                        gating_mask=mask_512,
                        controlnet_cond=controlnet_conditioning_image,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )
                else:
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

                ori_input = latents.detach().clone()
                if use_sa_preserve and i in self.embedding:
                    latents = (
                            latents * (1.0 - guidance_mask)
                            + torch.from_numpy(self.embedding[i]).to(latents.device) * guidance_mask
                        ).float()
                
                if sa_preserve:
                    self.embedding[i] = ori_input.cpu().numpy()
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd1x_path", type=str, default="/datasd1-4")
    parser.add_argument("--migc_path", type=str, default="/data/checkpoint-1800/migc.pth")

    parser.add_argument("--data_path", type=str, default="/data/val")

    
    parser.add_argument("--edge_path", type=str, default="/data/edge")  
    parser.add_argument("--out_dir", type=str, default="/data/out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    return parser.parse_args()

def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank
    else:
        print("Not using distributed mode.")
        return 0, 1, 0

def main():
    args = parse_args()
    seed_everything(args.seed)
    
    rank, world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    
    OUTPUT_DIR = args.out_dir
    raw_dir = os.path.join(OUTPUT_DIR, "raw")
    concat_dir = os.path.join(OUTPUT_DIR, "concat")
    anno_dir = os.path.join(OUTPUT_DIR, "anno")
    
    if rank == 0:
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(concat_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)
        print(f"Output directory: {OUTPUT_DIR}")
        
    if world_size > 1:
        dist.barrier()
        
    if rank == 0:
        print(f"Loading model from {args.sd1x_path}...")
    
    # Load Pipeline
    pipe = StableDiffusionMIGCPipeline.from_pretrained(
        args.sd1x_path, 
        
    )
    pipe.attention_store = AttentionStore()
    load_migc(pipe.unet, pipe.attention_store, args.migc_path, attn_processor=MIGCProcessor)
    
    # Load ControlNet
    controlnet = ControlNetXSModel.init_original(pipe.unet, is_sdxl=False)
    controlnet_weights_path = '/data/controlnet/diffusion_pytorch_model.safetensors'
    if rank == 0:
        print(f"Loading ControlNet-xs weights from {controlnet_weights_path}...")
    controlnet.load_state_dict(load_file(controlnet_weights_path))
    set_safe_processors_for_controlnet(controlnet)
    
    controlnet = controlnet.to(device)
    pipe = pipe.to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    jsonl_path = os.path.join(args.data_path, "metadata.jsonl")
    all_lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        
    my_lines = all_lines[rank::world_size]
    if rank == 0:
        print(f"Total samples: {len(all_lines)}")
        print(f"[Rank {rank}] Processing {len(my_lines)} samples.")

    for line in tqdm(my_lines, disable=(rank != 0), desc=f"Rank {rank}"):
        item = json.loads(line)
        file_name = item.get("file_name")
        edge_image_path = os.path.join(args.edge_path, file_name)
        hed_source_image = Image.open(edge_image_path).convert("RGB")
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
                controlnet=controlnet,
                hed_image=hed_source_image,
                # hed_image=None,
                bboxes=current_bboxes, 
                num_inference_steps=50, 
                guidance_scale=7.5, 
                MIGCsteps=25, 
                aug_phase_with_and=False, 
                negative_prompt='worst quality, low quality, bad anatomy, watermark, text, blurry'
            )
            image = output.images[0]
        except Exception as e:
            print(f"[Rank {rank}] Inference failed for {file_name}: {e}")
            continue

        image.save(save_path_raw)


        gt_image_path = os.path.join('/data/val', file_name)
        
        if not os.path.exists(gt_image_path): 
             pass

        if os.path.exists(gt_image_path):
            gt_img = Image.open(gt_image_path).convert("RGB")
            # Resize GT to match generated image size (usually 512x512)
            gt_img = gt_img.resize(image.size)
            
            concat_img = Image.new('RGB', (gt_img.width + image.width, image.height))
            concat_img.paste(gt_img, (0, 0))
            concat_img.paste(image, (gt_img.width, 0))
            
            concat_img.save(os.path.join(concat_dir, f'concat_{os.path.basename(file_name)}'))


        image_anno = image.copy()
        image_anno = pipe.draw_box_desc(image_anno, valid_boxes, valid_phrases)
        image_anno.save(os.path.join(anno_dir, f'anno_{os.path.basename(file_name)}'))

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    
    if rank == 0:
        print("All processing done.")

if __name__ == "__main__":
    main()