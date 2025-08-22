#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from typing import List, Any
import pandas as pd
import random
import traceback
import os
import gradio as gr

from PIL import Image, ImageOps, ImageFile

from src.pipelines.diffusion import generate_images, generate_images_with_refiner, generate_images_to_images, generate_images_to_images_with_refiner, generate_images_inpaint, generate_images_inpaint_with_refiner

from src.common.utils import get_all_diffusion_models
from src.models import diffusion_api_models, diffusers_local, checkpoints_local
from src import logger, os_name, arch

def generate_images_wrapper(positive_prompt: str, negative_prompt: str, style: str, generation_step: int, img2img_step_start: int, diffusion_refiner_start: int, width: int, height: int,
    diffusion_model: str, diffusion_refiner_model: str, diffusion_model_type: str, lora_multiselect: List[str], vae: str, clip_skip: int, enable_clip_skip: bool, clip_g: bool, sampler: str, scheduler: str,
    batch_size: int, batch_count: int, cfg_scale: float, seed: int, random_seed: bool, image_to_image_mode: str, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, image_inpaint_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, denoise_strength: float = 1, blur_radius: float = 5.0, blur_expansion_radius: float = 1, api_key: str | None = None,
    # 이후 20개의 슬라이더 값 (max_diffusion_lora_rows * 2; 예를 들어 10행이면 20개)
    *lora_slider_values):
    n = len(lora_slider_values) // 2
    text_weights = list(lora_slider_values[:n])
    unet_weights = list(lora_slider_values[n:])
    # JSON 문자열로 변환
    text_weights_json = json.dumps(text_weights)
    unet_weights_json = json.dumps(unet_weights)
    if diffusion_model_type == "api":
        return api_image_generation(positive_prompt, width, height, diffusion_model, api_key)
    else:
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
        all_models = update_diffusion_allowed_models(os_name, arch)
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

def update_diffusion_allowed_models(os_name, arch):
    if os_name == "Darwin" and arch == "x86_64":
        return diffusion_api_models
    else:
        return diffusion_api_models + diffusers_local + checkpoints_local

def toggle_diffusion_api_key_visibility(selected_model):
    api_visible = selected_model in diffusion_api_models
    return gr.update(visible=api_visible)

def get_allowed_diffusion_models(os_name, arch):
    if os_name == "Darwin" and arch == "x86_64":
        allowed = diffusion_api_models
        allowed_type = ["all", "api"]
    else:
        allowed = diffusion_api_models + diffusers_local + checkpoints_local
        allowed_type = ["all", "api", "diffusers", "checkpoints"]
    
    allowed = list(dict.fromkeys(allowed))
    return sorted(allowed), allowed_type

def api_image_generation(prompt, width, height, model, api_key=None):
    if "dall-e" in model:
        import openai
        if not api_key:
            logger.error("OpenAI API Key가 missing.")
            return [], None 
        openai.api_key = api_key
        
        try:
            response = openai.images.generate(
                model=model,
                prompt=prompt,
                size=f"{width}x{height}",
                quality="standard",
                n=1,
            )
            
            output_images=[]
            
            image = response.data[0].url
            output_images.append(image)
            
            history_entry = {
                "Positive Prompt": prompt,
                "Negative Prompt": "",
                "Generation Steps": "",
                "Model": model,
                "Sampler": "",
                "Scheduler": "",
                "CFG Scale": "",
                "Seed": "",
                "Width": width,
                "Height": height
            }
            history_df = pd.DataFrame([history_entry])

            return output_images, history_df
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return [], None
    elif "gpt-image" in model:
        import base64
        import openai
        if not api_key:
            logger.error("OpenAI API Key가 missing.")
            return [], None 
        openai.api_key = api_key
        try:
            response = openai.images.generate(
                model=model,
                prompt=prompt,
                size=f"{width}x{height}",
                quality="high",
                n=1,
            )
            output_images=[]
            image = response.data[0].url
            output_images.append(image)
            
            history_entry = {
                "Positive Prompt": prompt,
                "Negative Prompt": "",
                "Generation Steps": "",
                "Model": model,
                "Sampler": "",
                "Scheduler": "",
                "CFG Scale": "",
                "Seed": "",
                "Width": width,
                "Height": height
            }
            history_df = pd.DataFrame([history_entry])
            return output_images, history_df
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return [], None 
    else:
        from google import genai
        from google.genai import types
        from PIL import Image
        from io import BytesIO
        if not api_key:
            logger.error("Google API Key가 missing.")
            return [], None
        client = genai.Client(api_key=api_key)
        try:
            response = client.models.generate_images(
                model=model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1
                )
            )
            output_images=[]
            for generated_image in response.generated_images:
                image = Image.open(BytesIO(generated_image.image.image_bytes))
                output_images.append(image)
                
            history_entry = {
                "Positive Prompt": prompt,
                "Negative Prompt": "",
                "Generation Steps": "",
                "Model": model,
                "Sampler": "",
                "Scheduler": "",
                "CFG Scale": "",
                "Seed": "",
                "Width": "",
                "Height": ""
            }
            history_df = pd.DataFrame([history_entry])
            return output_images, history_df
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return [], None 