"""
Inpainting generation pipeline using ComfyUI.
"""

import io
import json
import random
import traceback
from typing import List, Tuple, Optional, Dict, Any, Union

import pandas as pd
from PIL import Image, ImageFile

from ..comfy_api import ComfyUIClientWrapper
from ..comfyui_workflows import (
    load_inpaint_workflow,
    load_inpaint_sdxl_workflow,
    load_inpaint_sdxl_with_refiner_workflow,
    load_inpaint_workflow_clip_skip,
    load_inpaint_sdxl_workflow_clip_skip,
    load_inpaint_sdxl_with_refiner_workflow_clip_skip
)

from src import logger


class InpaintPipeline:
    """Inpainting generation pipeline using ComfyUI."""

    def __init__(
        self,
        model: str,
        model_type: str = "checkpoint",
        refiner: str = None,
        loras: List[str] = None,
        vae: str = None
    ):
        self.client = ComfyUIClientWrapper()
        self.model = model
        self.model_type = model_type
        self.refiner = refiner
        self.loras = loras or []
        self.vae = vae

    def _get_seed(self, seed: int, random_seed: bool) -> int:
        if random_seed:
            return random.randint(0, 9007199254740991)
        return seed

    def _apply_loras(
        self,
        prompt: Dict[str, Any],
        lora_text_weights: List[float],
        lora_unet_weights: List[float],
        base_node: str,
        start_node_id: int
    ) -> Tuple[str, int]:
        current_node_id = start_node_id

        for i, lora in enumerate(self.loras):
            text_weight = lora_text_weights[i] if i < len(lora_text_weights) else 1.0
            unet_weight = lora_unet_weights[i] if i < len(lora_unet_weights) else 1.0
            new_node_id = str(current_node_id)
            prompt[new_node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora,
                    "strength_model": text_weight,
                    "strength_clip": unet_weight,
                    "model": [base_node, 0],
                    "clip": [base_node, 1]
                }
            }
            base_node = new_node_id
            current_node_id += 1

        return base_node, current_node_id

    def _apply_vae(
        self,
        prompt: Dict[str, Any],
        current_node_id: int,
        vae_target_node: str = "8"
    ) -> int:
        if self.vae == "Default":
            return current_node_id

        vae_value = self.vae
        new_node_id = str(current_node_id)
        prompt[new_node_id] = {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_value
            }
        }
        prompt[vae_target_node]["inputs"]["vae"] = [new_node_id, 0]
        return current_node_id + 1

    def _process_results(
        self,
        generated: Dict[str, Any],
        history_data: Dict[str, Any]
    ) -> Tuple[List[Image.Image], pd.DataFrame]:
        output_images = []
        width = history_data.get("Width", 0)
        height = history_data.get("Height", 0)

        for node_id in generated:
            for image_data in generated[node_id]:
                try:
                    image = Image.open(io.BytesIO(image_data))
                    output_images.append(image)
                    width, height = image.size
                except Exception as e:
                    logger.error(f"이미지 로딩 오류: {str(e)}\n\n{traceback.format_exc()}")

        history_data["Width"] = width
        history_data["Height"] = height

        history_df = pd.DataFrame([history_data])
        return output_images, history_df

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        style: str,
        generation_step: int,
        image_input: Union[str, Image.Image, ImageFile.ImageFile],
        denoise_strength: float,
        blur_radius: float,
        blur_expansion_radius: int,
        vae: str,
        clip_skip: int,
        enable_clip_skip: bool,
        clip_g: bool,
        sampler: str,
        scheduler: str,
        batch_count: int,
        cfg_scale: float,
        seed: int,
        random_seed: bool,
        lora_text_weights_json: str,
        lora_unet_weights_json: str
    ) -> Tuple[List[Image.Image], Optional[pd.DataFrame]]:
        """
        Generate images using inpainting workflow.

        Args:
            positive_prompt: Positive prompt text
            negative_prompt: Negative prompt text
            style: Style preset
            generation_step: Number of generation steps
            image_input: Input image path or PIL Image
            denoise_strength: Denoising strength
            blur_radius: Mask blur radius
            blur_expansion_radius: Mask blur expansion radius
            vae: VAE model name or "Default"
            clip_skip: CLIP skip value
            enable_clip_skip: Whether to enable CLIP skip
            clip_g: Whether to use CLIP-G (SDXL)
            sampler: Sampler name
            scheduler: Scheduler name
            batch_count: Number of batches
            cfg_scale: CFG scale value
            seed: Random seed
            random_seed: Whether to use random seed
            lora_text_weights_json: JSON string of LoRA text weights
            lora_unet_weights_json: JSON string of LoRA UNet weights

        Returns:
            Tuple of (generated images list, history DataFrame)
        """
        try:
            seed = self._get_seed(seed, random_seed)

            lora_text_weights = json.loads(lora_text_weights_json)
            lora_unet_weights = json.loads(lora_unet_weights_json)

            if enable_clip_skip:
                clip_skip = clip_skip * (-1)

            # Load appropriate workflow
            if clip_g:
                if enable_clip_skip:
                    prompt = load_inpaint_sdxl_workflow_clip_skip()
                else:
                    prompt = load_inpaint_sdxl_workflow()
            else:
                if enable_clip_skip:
                    prompt = load_inpaint_workflow_clip_skip()
                else:
                    prompt = load_inpaint_workflow()

            # Configure sampler node
            prompt["3"]["inputs"]["cfg"] = cfg_scale
            prompt["3"]["inputs"]["sampler_name"] = sampler
            prompt["3"]["inputs"]["scheduler"] = scheduler
            prompt["3"]["inputs"]["seed"] = seed
            prompt["3"]["inputs"]["steps"] = generation_step
            prompt["3"]["inputs"]["denoise"] = denoise_strength
            prompt["4"]["inputs"]["ckpt_name"] = self.model

            # Configure prompts
            if clip_g:
                prompt["6"]["inputs"]["text_l"] = positive_prompt
                prompt["6"]["inputs"]["text_g"] = positive_prompt
                prompt["7"]["inputs"]["text_l"] = negative_prompt
                prompt["7"]["inputs"]["text_g"] = negative_prompt
            else:
                prompt["6"]["inputs"]["text"] = positive_prompt
                prompt["7"]["inputs"]["text"] = negative_prompt

            # Set input image
            prompt["10"]["inputs"]["image"] = image_input

            # Set blur parameters
            prompt["12"]["inputs"]["blur_radius"] = blur_radius
            prompt["12"]["inputs"]["blur_expansion_radius"] = blur_expansion_radius

            if enable_clip_skip:
                prompt["15"]["inputs"]["stop_at_clip_layer"] = clip_skip

            # Apply LoRAs
            base_node = "4"
            current_node_id = 16 if enable_clip_skip else 15

            base_node, current_node_id = self._apply_loras(
                prompt, lora_text_weights, lora_unet_weights,
                base_node, current_node_id
            )

            prompt["3"]["inputs"]["model"] = [base_node, 0]
            if enable_clip_skip:
                prompt["15"]["inputs"]["clip"] = [base_node, 1]
            else:
                prompt["6"]["inputs"]["clip"] = [base_node, 1]
                prompt["7"]["inputs"]["clip"] = [base_node, 1]

            # Apply VAE
            current_node_id = self._apply_vae(prompt, current_node_id)

            # Generate
            generated = self.client.image2image_generate(prompt)

            history_data = {
                "Positive Prompt": positive_prompt,
                "Negative Prompt": negative_prompt,
                "Generation Steps": generation_step,
                "Model": self.model,
                "Sampler": sampler,
                "Scheduler": scheduler,
                "CFG Scale": cfg_scale,
                "Seed": seed,
                "Width": 0,
                "Height": 0
            }

            return self._process_results(generated, history_data)

        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return [], None

    def generate_with_refiner(
        self,
        positive_prompt: str,
        negative_prompt: str,
        style: str,
        generation_step: int,
        diffusion_img2img_start: int,
        diffusion_refiner_start: int,
        image_input: str,
        denoise_strength: float,
        blur_radius: float,
        blur_expansion_radius: int,
        vae: str,
        clip_skip: int,
        enable_clip_skip: bool,
        sampler: str,
        scheduler: str,
        batch_count: int,
        cfg_scale: float,
        seed: int,
        random_seed: bool,
        lora_text_weights_json: str,
        lora_unet_weights_json: str
    ) -> Tuple[List[Image.Image], Optional[pd.DataFrame]]:
        """
        Generate images using inpainting workflow with refiner.
        """
        try:
            seed = self._get_seed(seed, random_seed)

            lora_text_weights = json.loads(lora_text_weights_json)
            lora_unet_weights = json.loads(lora_unet_weights_json)

            if enable_clip_skip:
                clip_skip = clip_skip * (-1)

            if enable_clip_skip:
                prompt = load_inpaint_sdxl_with_refiner_workflow_clip_skip()
            else:
                prompt = load_inpaint_sdxl_with_refiner_workflow()

            # Configure base sampler
            prompt["3"]["inputs"]["cfg"] = cfg_scale
            prompt["3"]["inputs"]["sampler_name"] = sampler
            prompt["3"]["inputs"]["scheduler"] = scheduler
            prompt["3"]["inputs"]["noise_seed"] = seed
            prompt["3"]["inputs"]["steps"] = generation_step
            prompt["3"]["inputs"]["start_at_step"] = diffusion_img2img_start
            prompt["3"]["inputs"]["end_at_step"] = diffusion_refiner_start
            prompt["4"]["inputs"]["ckpt_name"] = self.model

            # Configure base prompts
            prompt["6"]["inputs"]["text_l"] = positive_prompt
            prompt["6"]["inputs"]["text_g"] = positive_prompt
            prompt["7"]["inputs"]["text_l"] = negative_prompt
            prompt["7"]["inputs"]["text_g"] = negative_prompt

            # Configure refiner sampler
            prompt["10"]["inputs"]["cfg"] = cfg_scale
            prompt["10"]["inputs"]["sampler_name"] = sampler
            prompt["10"]["inputs"]["scheduler"] = scheduler
            prompt["10"]["inputs"]["noise_seed"] = seed
            prompt["10"]["inputs"]["steps"] = generation_step
            prompt["10"]["inputs"]["start_at_step"] = diffusion_refiner_start

            # Configure refiner prompts
            prompt["11"]["inputs"]["text_l"] = positive_prompt
            prompt["11"]["inputs"]["text_g"] = positive_prompt
            prompt["12"]["inputs"]["text_l"] = negative_prompt
            prompt["12"]["inputs"]["text_g"] = negative_prompt
            prompt["13"]["inputs"]["ckpt_name"] = self.refiner

            # Set input image
            prompt["14"]["inputs"]["image"] = image_input

            # Set blur parameters
            prompt["16"]["inputs"]["blur_radius"] = blur_radius
            prompt["16"]["inputs"]["blur_expansion_radius"] = blur_expansion_radius

            if enable_clip_skip:
                prompt["19"]["inputs"]["stop_at_clip_layer"] = clip_skip

            # Apply LoRAs
            base_node = "4"
            current_node_id = 20 if enable_clip_skip else 19

            base_node, current_node_id = self._apply_loras(
                prompt, lora_text_weights, lora_unet_weights,
                base_node, current_node_id
            )

            prompt["3"]["inputs"]["model"] = [base_node, 0]
            if enable_clip_skip:
                prompt["19"]["inputs"]["clip"] = [base_node, 1]
            else:
                prompt["6"]["inputs"]["clip"] = [base_node, 1]
                prompt["7"]["inputs"]["clip"] = [base_node, 1]

            # Apply VAE
            current_node_id = self._apply_vae(prompt, current_node_id)

            # Generate
            generated = self.client.image2image_generate(prompt)

            history_data = {
                "Positive Prompt": positive_prompt,
                "Negative Prompt": negative_prompt,
                "Generation Steps": generation_step,
                "Model": self.model,
                "Sampler": sampler,
                "Scheduler": scheduler,
                "CFG Scale": cfg_scale,
                "Seed": seed,
                "Width": 0,
                "Height": 0
            }

            return self._process_results(generated, history_data)

        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return [], None


# Backward compatibility functions
def generate_images_inpaint(
    positive_prompt: str,
    negative_prompt: str,
    style: str,
    generation_step: int,
    diffusion_model: str,
    diffusion_model_type: str,
    loras: List[str],
    vae: str,
    clip_skip: int,
    enable_clip_skip: bool,
    clip_g: bool,
    sampler: str,
    scheduler: str,
    batch_count: int,
    cfg_scale: float,
    seed: int,
    random_seed: bool,
    image_input: Union[str, Image.Image, ImageFile.ImageFile],
    denoise_strength: float,
    blur_radius: float,
    blur_expansion_radius: int,
    lora_text_weights_json: str,
    lora_unet_weights_json: str
) -> Tuple[List[Image.Image], Optional[pd.DataFrame]]:
    """Backward compatible function for inpainting generation."""
    pipeline = InpaintPipeline(
        model=diffusion_model,
        model_type=diffusion_model_type,
        loras=loras,
        vae=vae
    )
    return pipeline.generate(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        style=style,
        generation_step=generation_step,
        image_input=image_input,
        denoise_strength=denoise_strength,
        blur_radius=blur_radius,
        blur_expansion_radius=blur_expansion_radius,
        clip_skip=clip_skip,
        enable_clip_skip=enable_clip_skip,
        clip_g=clip_g,
        sampler=sampler,
        scheduler=scheduler,
        batch_count=batch_count,
        cfg_scale=cfg_scale,
        seed=seed,
        random_seed=random_seed,
        lora_text_weights_json=lora_text_weights_json,
        lora_unet_weights_json=lora_unet_weights_json
    )


def generate_images_inpaint_with_refiner(
    positive_prompt: str,
    negative_prompt: str,
    style: str,
    generation_step: int,
    diffusion_img2img_start: int,
    diffusion_refiner_start: int,
    diffusion_model: str,
    diffusion_refiner_model: str,
    diffusion_model_type: str,
    loras: List[str],
    vae: str,
    clip_skip: int,
    enable_clip_skip: bool,
    clip_g: bool,
    sampler: str,
    scheduler: str,
    batch_count: int,
    cfg_scale: float,
    seed: int,
    random_seed: bool,
    image_input: str,
    denoise_strength: float,
    blur_radius: float,
    blur_expansion_radius: int,
    lora_text_weights_json: str,
    lora_unet_weights_json: str
) -> Tuple[List[Image.Image], Optional[pd.DataFrame]]:
    """Backward compatible function for inpainting generation with refiner."""
    pipeline = InpaintPipeline(
        model=diffusion_model,
        model_type=diffusion_model_type,
        refiner=diffusion_refiner_model,
        loras=loras,
        vae=vae
    )
    return pipeline.generate_with_refiner(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        style=style,
        generation_step=generation_step,
        diffusion_img2img_start=diffusion_img2img_start,
        diffusion_refiner_start=diffusion_refiner_start,
        image_input=image_input,
        denoise_strength=denoise_strength,
        blur_radius=blur_radius,
        blur_expansion_radius=blur_expansion_radius,
        clip_skip=clip_skip,
        enable_clip_skip=enable_clip_skip,
        sampler=sampler,
        scheduler=scheduler,
        batch_count=batch_count,
        cfg_scale=cfg_scale,
        seed=seed,
        random_seed=random_seed,
        lora_text_weights_json=lora_text_weights_json,
        lora_unet_weights_json=lora_unet_weights_json
    )
