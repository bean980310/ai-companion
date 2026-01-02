#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from typing import List, Any, Callable
import pandas as pd
import random
import traceback
import os
import gradio as gr

import numpy as np

from PIL import Image, ImageOps, ImageFile

from src.start_app.app_state_manager import app_state

# Import models module for ComfyUI pipeline creation
from src.models.models import create_comfyui_pipeline

from src.common.utils import get_all_diffusion_models, detect_platform, get_diffusion_loras, get_diffusion_vae
from src.models import diffusion_api_models, openai_image_api_models, google_genai_image_models, comfyui_image_models, comfyui_image_loras, comfyui_image_vae, diffusers_local, checkpoints_local
from src import logger, os_name, arch

from .upload import ComfyUIImageUpload

class ImageGeneration:
    def __init__(self):
        self.os_name, self.arch = detect_platform()

    def generate_images_wrapper(self, positive_prompt: str, negative_prompt: str, style: str, generation_step: int, img2img_step_start: int, diffusion_refiner_start: int, width: int, height: int,
        model: str, refiner_model: str, model_provider: str, model_type: str, lora_multiselect: List[str], vae: str, clip_skip: int, enable_clip_skip: bool, clip_g: bool, sampler: str, scheduler: str,
        batch_size: int, batch_count: int, cfg_scale: float, seed: int, random_seed: bool, image_to_image_mode: str, image_input: str | Image.Image | ImageFile.ImageFile | np.ndarray | Callable | Any | None = None, image_inpaint_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, denoise_strength: float = 1, blur_radius: float = 5.0, blur_expansion_radius: float = 1, api_key: str | None = None,
        # 이후 20개의 슬라이더 값 (max_diffusion_lora_rows * 2; 예를 들어 10행이면 20개)
        *lora_slider_values):
        n = len(lora_slider_values) // 2
        text_weights = list(lora_slider_values[:n])
        unet_weights = list(lora_slider_values[n:])
        # JSON 문자열로 변환
        text_weights_json = json.dumps(text_weights)
        unet_weights_json = json.dumps(unet_weights)
        if all(x not in model_provider.lower() for x in ["self-provided", "comfyui", "invokeai", "drawthings", "sd-webui"]):
            return self.api_image_generation(positive_prompt, width, height, model, api_key)
        elif model_provider.lower() == "comfyui":
            if image_to_image_mode == "None":
                # Create pipeline through models.py
                pipeline = create_comfyui_pipeline(
                    image_to_image_mode=image_to_image_mode,
                    model=model,
                    refiner_model=refiner_model,
                    loras=lora_multiselect,
                    vae=vae
                )
                if refiner_model == "None":
                    return pipeline.generate(
                        positive_prompt, negative_prompt, style, generation_step, width, height,
                        clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
                        batch_size, batch_count, cfg_scale, seed, random_seed,
                        text_weights_json, unet_weights_json
                    )
                else:
                    clip_g=True
                    return pipeline.generate_with_refiner(
                        positive_prompt, negative_prompt, style, generation_step, diffusion_refiner_start, width, height,
                        clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
                        batch_size, batch_count, cfg_scale, seed, random_seed,
                        text_weights_json, unet_weights_json
                    )
            elif image_to_image_mode == "Image to Image":
                # Create pipeline through models.py
                pipeline = create_comfyui_pipeline(
                    image_to_image_mode=image_to_image_mode,
                    model=model,
                    refiner_model=refiner_model,
                    loras=lora_multiselect,
                    vae=vae
                )
                if refiner_model == "None":
                    return pipeline.generate(
                        positive_prompt, negative_prompt, style, generation_step,
                        clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
                        batch_count, cfg_scale, seed, random_seed, image_input, denoise_strength,
                        text_weights_json, unet_weights_json
                    )
                else:
                    clip_g=True
                    return pipeline.generate_with_refiner(
                        positive_prompt, negative_prompt, style, generation_step, img2img_step_start, diffusion_refiner_start,
                        clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
                        batch_count, cfg_scale, seed, random_seed, image_input, denoise_strength,
                        text_weights_json, unet_weights_json
                    )
            elif image_to_image_mode == "Inpaint":
                # Create pipeline through models.py
                pipeline = create_comfyui_pipeline(
                    image_to_image_mode=image_to_image_mode,
                    model=model,
                    refiner_model=refiner_model,
                    loras=lora_multiselect,
                    vae=vae
                )
                if refiner_model == "None":
                    return pipeline.generate(
                        positive_prompt, negative_prompt, style, generation_step,
                        clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
                        batch_count, cfg_scale, seed, random_seed, image_inpaint_input, denoise_strength, blur_radius, blur_expansion_radius,
                        text_weights_json, unet_weights_json
                    )
                else:
                    clip_g=True
                    return pipeline.generate_with_refiner(
                        positive_prompt, negative_prompt, style, generation_step, img2img_step_start, diffusion_refiner_start,
                        clip_skip, enable_clip_skip, clip_g, sampler, scheduler,
                        batch_count, cfg_scale, seed, random_seed, image_inpaint_input, denoise_strength, blur_radius, blur_expansion_radius,
                        text_weights_json, unet_weights_json
                    )

        else:
            logger.error("self-provided를 통한 이미지 생성은 현재 지원하지 않습니다. 추후 업데이트에서 지원하도록 하겠습니다.")
            return [], None
                
    def update_diffusion_model_list(self, provider: str, selected_type: str | None = None):
        diffusion_models_data = get_all_diffusion_models()
        diffusers_local = diffusion_models_data["diffusers"]
        checkpoints_local = diffusion_models_data["checkpoints"]

        if provider != "self-provided":
            if provider == "openai":
                updated_list = openai_image_api_models
            elif provider == "google-genai":
                updated_list = google_genai_image_models
            elif provider == "comfyui":
                updated_list = comfyui_image_models

            updated_list = sorted(list(dict.fromkeys(updated_list)))
            app_state.diffusion_choices = updated_list
            return gr.update(visible='hidden'), gr.update(choices=updated_list, value=updated_list[0] if updated_list else None)
        else:
            diffusion_choices, diffusion_type_choices = self.get_allowed_diffusion_models(os_name, arch)
            if selected_type == "all":
                all_models = diffusion_choices
                # 중복 제거 후 정렬
                all_models = sorted(list(dict.fromkeys(all_models)))
                app_state.diffusion_choices = all_models
                app_state.diffusion_type_choices = diffusion_type_choices
                return gr.update(visible=True), gr.update(choices=all_models, value=all_models[0] if all_models else None)
            
            elif selected_type == "diffusers":
                updated_list = diffusers_local
            elif selected_type == "checkpoints":
                updated_list = checkpoints_local
            else:
                updated_list = diffusers_local
                
            updated_list = sorted(list(dict.fromkeys(updated_list)))
            app_state.diffusion_choices = updated_list
            app_state.diffusion_type_choices = selected_type
            return gr.update(visible=True), gr.update(choices=updated_list, value=updated_list[0] if updated_list else None)

    @staticmethod
    def toggle_diffusion_api_key_visibility(provider: str | gr.Dropdown) -> bool:
        api_visible = any(x in provider.lower() for x in ["openai","google-genai", "xai", "hf-inference"])
        return gr.update(visible=api_visible)

    def toggle_diffusion_lora_visible(self, provider: str | gr.Dropdown):
        lora_visible = any(x in provider.lower() for x in ["comfyui", "invokeai", "drawthings", "sd-webui", "self-provided"])
        if not lora_visible:
            lora_visible = "hidden"
        updated_choices = self.get_allowed_diffusion_loras(provider)
        app_state.diffusion_lora_choices = updated_choices
        return gr.update(visible=lora_visible), gr.update(choices=updated_choices)

    def toggle_diffusion_vae_visible(self, provider: str | gr.Dropdown):
        vae_visible = any(x in provider.lower() for x in ["comfyui", "invokeai", "drawthings", "sd-webui", "self-provided"])
        if not vae_visible:
            vae_visible = "hidden"
        updated_choices = self.get_allowed_diffusion_vae(provider)
        app_state.diffusion_vae_choices = updated_choices
        return gr.update(visible=vae_visible), gr.update(choices=updated_choices)

    def toggle_diffusion_refiner_visible(self, provider: str | gr.Dropdown):
        refiner_visible = any(x in provider.lower() for x in ["comfyui", "invokeai", "drawthings", "sd-webui", "self-provided"])
        
        if not refiner_visible:
            refiner_visible = "hidden"
            updated_choices = ["Not Supported"]
        else:
            if provider == "comfyui":
                updated_choices = comfyui_image_models
            elif provider == "self-provided":
                updated_choices = sorted(list(dict.fromkeys(diffusers_local + checkpoints_local)))

            if "None" not in updated_choices:
                updated_choices.insert(0, "None")
            
        app_state.diffusion_refiner_choices = updated_choices
        return gr.update(visible=refiner_visible), gr.update(choices=updated_choices)

    @staticmethod
    def get_allowed_diffusion_models(os_name, arch):
        allowed = diffusers_local + checkpoints_local
        allowed_type = ["all", "diffusers", "checkpoints"]
        
        allowed = list(dict.fromkeys(allowed))
        return sorted(allowed), allowed_type

    @staticmethod
    def process_uploaded_image(image: str | ImageFile.ImageFile | Image.Image | np.ndarray | Callable | Any):
        client = ComfyUIImageUpload()
        print(image)
        image = client.upload_image(image, overwrite=True)
        return image
    
    @staticmethod
    def toggle_refiner_start_step(model):
        slider_visible = model != "None"
        return gr.update(visible=slider_visible)
    
    @staticmethod
    def toggle_denoise_strength_dropdown(mode):
        slider_visible = mode != "None"
        return gr.update(visible=slider_visible)
    
    @staticmethod
    def toggle_blur_radius_slider(mode):
        slider_visible = mode == "Inpaint" or mode == "Inpaint Upload"
        return gr.update(visible=slider_visible), gr.update(visible=slider_visible)
    
    @staticmethod
    def toggle_diffusion_with_refiner_image_to_image_start(model, mode):
        slider_visible = model != "None" and mode != "None"
        return gr.update(visible=slider_visible)

    @staticmethod
    def toggle_image_to_image_input(mode):
        image_visible = mode == "Image to Image"
        return gr.update(visible=image_visible)

    @staticmethod
    def toggle_image_inpaint_input(mode):
        image_visible = mode == "Inpaint"
        return gr.update(visible=image_visible)

    @staticmethod
    def toggle_image_inpaint_mask(mode):
        image_visible = mode == "Inpaint"
        return gr.update(visible=image_visible)

    @staticmethod
    def toggle_image_inpaint_mask_interactive(image: str | Image.Image | Any):
        image_interactive = image is not None
        return gr.update(interactive=image_interactive)
    
    @staticmethod
    def copy_image_for_inpaint(image_input, image) -> gr.update:
        import cv2
        print(type(image_input))
        if isinstance(image_input, Image.Image):
            temp = np.array(image_input)
            temp_cv2 = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR).copy()
            im = temp_cv2
        else:
            im = cv2.imread(image_input)

        height, width, channels = im.shape[:3]
        image['background'] = image_input
        image['layers'][0] = np.zeros((height, width, 4), dtype=np.uint8)

        return gr.update(value=image)

    def get_allowed_diffusion_loras(self, provider: str | gr.Dropdown = "self-provided"):
        if provider == "self-provided":
            diffusion_lora_choices = get_diffusion_loras()
        elif provider == "comfyui":
            diffusion_lora_choices = comfyui_image_loras
        else:
            diffusion_lora_choices = []

        diffusion_lora_choices = list(dict.fromkeys(diffusion_lora_choices))
        diffusion_lora_choices = sorted(diffusion_lora_choices)
        return diffusion_lora_choices

    def get_allowed_diffusion_vae(self, provider: str | gr.Dropdown = "self-provided"):
        if provider == "self-provided":
            vae_choices = get_diffusion_vae()
        elif provider == "comfyui":
            vae_choices = comfyui_image_vae
        else:
            vae_choices = []

        vae_choices = list(dict.fromkeys(vae_choices))
        vae_choices = sorted(vae_choices)
        if "Default" not in vae_choices:
            vae_choices.insert(0, "Default")

        return vae_choices

    @staticmethod
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
        elif all(n in model.lower() for n in ["gemini", "image"]):
            import base64
            import mimetypes
            import datetime
            from google import genai
            from google.genai import types
            from PIL import Image
            from io import BytesIO
            if not api_key:
                logger.error("Google API Key가 missing.")
                return [], None
            client = genai.Client(api_key=api_key)
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=[
                            "IMAGE"
                        ],
                        image_config=types.ImageConfig(
                            aspect_ratio=f"{width}:{height}",
                            image_size='1K'
                        )
                    )
                )
                output_images=[]
                for generated_image in response.parts:
                    if generated_image.inline_data:
                        file_name = f"generated_image{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
                        data = generated_image.inline_data.data
                        ext = mimetypes.guess_extension(generated_image.inline_data.mime_type)
                        with open(f"{file_name}{ext}", "wb") as f:
                            f.write(data)
                        image = Image.open(BytesIO(generated_image.as_image()))
                        output_images.append(image)
                    else:
                        pass
                    
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
                        aspect_ratio=f"{width}:{height}",
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