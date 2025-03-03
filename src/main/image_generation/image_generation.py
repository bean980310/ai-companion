#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from typing import List
import pandas as pd
import random
import traceback
import os
import gradio as gr

from src.handlers import generate_images, generate_images_with_refiner, generate_images_to_images, generate_images_to_images_with_refiner, generate_images_inpaint, generate_images_inpaint_with_refiner

from src.common.utils import get_all_diffusion_models
from src.models import diffusion_api_models, diffusers_local, checkpoints_local
from src import logger

def generate_images_wrapper(positive_prompt, negative_prompt, style, generation_step, img2img_step_start, diffusion_refiner_start, width, height,
    diffusion_model, diffusion_refiner_model, diffusion_model_type, lora_multiselect, vae, clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
    batch_size, batch_count, cfg_scale, seed, random_seed, image_to_image_mode, image_input=None, image_inpaint_input=None, denoise_strength=1, blur_radius=5.0, blur_expansion_radius=1,
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
    elif image_to_image_mode == "Inpaint":
        if diffusion_refiner_model == "None":
            return generate_images_inpaint(
                positive_prompt, negative_prompt, style, generation_step,
                diffusion_model, diffusion_model_type, lora_multiselect, vae, clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
                batch_count, cfg_scale, seed, random_seed, image_inpaint_input, denoise_strength, blur_radius, blur_expansion_radius,
                text_weights_json, unet_weights_json
            )
        else:
            clip_g=True
            return generate_images_inpaint_with_refiner(
                positive_prompt, negative_prompt, style, generation_step, img2img_step_start, diffusion_refiner_start,
                diffusion_model, diffusion_refiner_model, diffusion_model_type, lora_multiselect, vae, clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
                batch_count, cfg_scale, seed, random_seed, image_inpaint_input, denoise_strength, blur_radius, blur_expansion_radius,
                text_weights_json, unet_weights_json
            )
            
def update_diffusion_model_list(selected_type):
    diffusion_models_data = get_all_diffusion_models()
    diffusers_local = diffusion_models_data["diffusers"]
    checkpoints_local = diffusion_models_data["checkpoints"]
    
    if selected_type == "all":
        all_models = diffusion_api_models + diffusers_local + checkpoints_local
        # 중복 제거 후 정렬
        all_models = sorted(list(dict.fromkeys(all_models)))
        return gr.update(choices=all_models, value=all_models[0] if all_models else None)
    
    if selected_type == "api":
        updated_list = diffusion_api_models
    elif selected_type == "diffusers":
        updated_list = diffusers_local
    elif selected_type == "checkpoints":
        updated_list = checkpoints_local
    else:
        updated_list == diffusers_local
        
    updated_list = sorted(list(dict.fromkeys(updated_list)))
    return gr.update(choices=updated_list, value=updated_list[0] if updated_list else None)

def toggle_diffusion_api_key_visibility(selected_model):
    api_visible = selected_model in diffusion_api_models
    return gr.update(visible=api_visible)