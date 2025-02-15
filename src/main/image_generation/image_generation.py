#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import logging
from typing import List
import pandas as pd
import random
import traceback
import os

from src.api.comfy_api import ComfyUIClient
from src.handlers.txt2img_handlers import generate_images, generate_images_with_refiner
from src.handlers.img2img_handlers import generate_images_to_images, generate_images_to_images_with_refiner

logger = logging.getLogger(__name__)

def generate_images_wrapper(positive_prompt, negative_prompt, style, generation_step, img2img_step_start, diffusion_refiner_start, width, height,
    diffusion_model, diffusion_refiner_model, diffusion_model_type, lora_multiselect, vae, clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
    batch_size, batch_count, cfg_scale, seed, random_seed, image_to_image_mode, image_input=None, image_inpaint_input=None, denoise_strength=1,
    # 이후 20개의 슬라이더 값 (max_diffusion_lora_rows * 2; 예를 들어 10행이면 20개)
    *lora_slider_values):
    n = len(lora_slider_values) // 2
    text_weights = list(lora_slider_values[:n])
    unet_weights = list(lora_slider_values[n:])
    # JSON 문자열로 변환
    text_weights_json = json.dumps(text_weights)
    unet_weights_json = json.dumps(unet_weights)
    if image_to_image_mode == "None":
        if diffusion_refiner_model == "None":
            return generate_images(
                positive_prompt, negative_prompt, style, generation_step, width, height,
                diffusion_model, diffusion_model_type, lora_multiselect, vae, clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
                batch_size, batch_count, cfg_scale, seed, random_seed,
                text_weights_json, unet_weights_json
            )
        else:
            clip_g=True
            return generate_images_with_refiner(
                positive_prompt, negative_prompt, style, generation_step, diffusion_refiner_start, width, height,
                diffusion_model, diffusion_refiner_model, diffusion_model_type, lora_multiselect, vae, clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
                batch_size, batch_count, cfg_scale, seed, random_seed,
                text_weights_json, unet_weights_json
            )
    elif image_to_image_mode == "Image to Image":
        if diffusion_refiner_model == "None":
            return generate_images_to_images(
                positive_prompt, negative_prompt, style, generation_step,
                diffusion_model, diffusion_model_type, lora_multiselect, vae, clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
                batch_count, cfg_scale, seed, random_seed, image_input, denoise_strength,
                text_weights_json, unet_weights_json
            )
        else:
            clip_g=True
            return generate_images_to_images_with_refiner(
                positive_prompt, negative_prompt, style, generation_step, img2img_step_start, diffusion_refiner_start,
                diffusion_model, diffusion_refiner_model, diffusion_model_type, lora_multiselect, vae, clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
                batch_count, cfg_scale, seed, random_seed, image_input, denoise_strength,
                text_weights_json, unet_weights_json
            )