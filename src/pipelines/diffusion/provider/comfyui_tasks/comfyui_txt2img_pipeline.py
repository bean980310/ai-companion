import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import random
import traceback
import os
import io
from PIL import Image 

from src.api.comfy_api import ComfyUIClient
from workflows.load_workflow import (
    load_txt2img_workflow, 
    load_txt2img_sdxl_workflow, 
    load_txt2img_sdxl_with_refiner_workflow, 
    load_txt2img_workflow_clip_skip, 
    load_txt2img_sdxl_workflow_clip_skip, 
    load_txt2img_sdxl_with_refiner_workflow_clip_skip
)

from src import logger

class Txt2ImgPipeline:
    def __init__(self, model: str, refiner: str = None, loras: List[str] = None, vae: str = None):
        self.client = ComfyUIClient()
        self.model = model
        self.refiner = refiner
        self.loras = loras
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
        base_node: str, 
        current_node_id: int,
        vae_target_node: str = "8"
    ) -> Tuple[str, int]:
        if self.vae == "Default":
            return base_node, current_node_id
            
        new_node_id = str(current_node_id)
        prompt[new_node_id] = {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": self.vae
            }
        }
        base_node = new_node_id
        current_node_id += 1
        prompt[vae_target_node]["inputs"]["vae"] = [base_node, 0]
        
        return base_node, current_node_id

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
        
        # Update width/height in history if changed by image loading
        history_data["Width"] = width
        history_data["Height"] = height
        
        history_df = pd.DataFrame([history_data])
        return output_images, history_df

    def generate_images(
        self,
        positive_prompt: str,
        negative_prompt: str,
        style: str,
        generation_step: int,
        width: int,
        height: int,
        clip_skip: int,
        enable_clip_skip: bool,
        clip_g: bool,
        sampler: str,
        scheduler: str,
        batch_size: int,
        batch_count: int,
        cfg_scale: float,
        seed: int,
        random_seed: bool,
        lora_text_weights_json: str,
        lora_unet_weights_json: str
    ) -> Tuple[List[Image.Image], Optional[pd.DataFrame]]:
        try:
            seed = self._get_seed(seed, random_seed)
            
            lora_text_weights = json.loads(lora_text_weights_json)
            lora_unet_weights = json.loads(lora_unet_weights_json)
            
            if enable_clip_skip:
                clip_skip = clip_skip * (-1)
            
            if clip_g:
                if enable_clip_skip:
                    prompt = load_txt2img_sdxl_workflow_clip_skip()
                else:
                    prompt = load_txt2img_sdxl_workflow()
            else:
                if enable_clip_skip:
                    prompt = load_txt2img_workflow_clip_skip()
                else:
                    prompt = load_txt2img_workflow()
            
            prompt["3"]["inputs"]["cfg"] = cfg_scale
            prompt["3"]["inputs"]["sampler_name"] = sampler
            prompt["3"]["inputs"]["scheduler"] = scheduler
            prompt["3"]["inputs"]["seed"] = seed
            prompt["3"]["inputs"]["steps"] = generation_step
            prompt["4"]["inputs"]["ckpt_name"] = self.model
            prompt["5"]["inputs"]["batch_size"] = batch_size
            prompt["5"]["inputs"]["width"] = width
            prompt["5"]["inputs"]["height"] = height
            
            if clip_g:
                prompt["6"]["inputs"]["text_l"] = positive_prompt
                prompt["6"]["inputs"]["text_g"] = positive_prompt
                prompt["7"]["inputs"]["text_l"] = negative_prompt
                prompt["7"]["inputs"]["text_g"] = negative_prompt
            else:
                prompt["6"]["inputs"]["text"] = positive_prompt
                prompt["7"]["inputs"]["text"] = negative_prompt
                
            if enable_clip_skip:
                prompt["10"]["inputs"]["stop_at_clip_layer"] = clip_skip
            
            base_node = "4"
            if enable_clip_skip:
                current_node_id = 11
            else:
                current_node_id = 10
            
            base_node, current_node_id = self._apply_loras(
                prompt, lora_text_weights, lora_unet_weights, base_node, current_node_id
            )
            
            prompt["3"]["inputs"]["model"] = [base_node, 0]
            if enable_clip_skip:   
                prompt["10"]["inputs"]["clip"] = [base_node, 1]
            else:
                prompt["6"]["inputs"]["clip"] = [base_node, 1]
                prompt["7"]["inputs"]["clip"] = [base_node, 1]
            
            # VAE handling
            # Note: The original code logic for VAE seems to use a new node ID sequence continuing from Lora
            # But wait, in original code:
            # base_node = new_node_id (from lora loop)
            # prompt["3"]["inputs"]["model"] = [base_node, 0]
            # ... clip connections ...
            # THEN vae check.
            # If vae != Default:
            # new_node_id = str(current_node_id) ...
            # base_node = new_node_id
            # prompt["8"]["inputs"]["vae"] = [base_node, 0]
            
            # So the base_node for VAE is the last Lora node (or the checkpoint if no loras).
            # But wait, VAE Loader usually doesn't take model/clip as input, it just loads VAE.
            # Ah, looking at the original code:
            # prompt[new_node_id] = { "class_type": "VAELoader", "inputs": { "vae_name": vae_value } }
            # base_node = new_node_id
            # prompt["8"]["inputs"]["vae"] = [base_node, 0]
            # It seems it just sets base_node to the VAE loader ID, and then connects node 8 (VAE Decode?) to it.
            # Yes, standard VAE Loader.
            
            _, current_node_id = self._apply_vae(prompt, base_node, current_node_id)

            generated = self.client.text2image_generate(prompt)
            
            history_data = {
                "Positive Prompt": positive_prompt,
                "Negative Prompt": negative_prompt,
                "Generation Steps": generation_step,
                "Model": self.model,
                "Sampler": sampler,
                "Scheduler": scheduler,
                "CFG Scale": cfg_scale,
                "Seed": seed,
                "Width": width,
                "Height": height
            }
            
            return self._process_results(generated, history_data)
        
        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return [], None

    def generate_images_with_refiner(
        self,
        positive_prompt: str,
        negative_prompt: str,
        style: str,
        generation_step: int,
        diffusion_refiner_start: int,
        width: int,
        height: int,
        clip_skip: int,
        enable_clip_skip: bool,
        clip_g: bool,
        sampler: str,
        scheduler: str,
        batch_size: int,
        batch_count: int,
        cfg_scale: float,
        seed: int,
        random_seed: bool,
        lora_text_weights_json: str,
        lora_unet_weights_json: str
    ) -> Tuple[List[Image.Image], Optional[pd.DataFrame]]:
        try:
            seed = self._get_seed(seed, random_seed)
            
            lora_text_weights = json.loads(lora_text_weights_json)
            lora_unet_weights = json.loads(lora_unet_weights_json)
            
            if enable_clip_skip:
                clip_skip = clip_skip * (-1)
            
            if enable_clip_skip:
                prompt = load_txt2img_sdxl_with_refiner_workflow_clip_skip()
            else:
                prompt = load_txt2img_sdxl_with_refiner_workflow()
            
            prompt["3"]["inputs"]["cfg"] = cfg_scale
            prompt["3"]["inputs"]["sampler_name"] = sampler
            prompt["3"]["inputs"]["scheduler"] = scheduler
            prompt["3"]["inputs"]["noise_seed"] = seed
            prompt["3"]["inputs"]["steps"] = generation_step
            prompt["3"]["inputs"]["end_at_step"] = diffusion_refiner_start
            prompt["4"]["inputs"]["ckpt_name"] = self.model
            prompt["5"]["inputs"]["batch_size"] = batch_size
            prompt["5"]["inputs"]["width"] = width
            prompt["5"]["inputs"]["height"] = height
            prompt["6"]["inputs"]["text_l"] = positive_prompt
            prompt["6"]["inputs"]["text_g"] = positive_prompt
            prompt["7"]["inputs"]["text_l"] = negative_prompt
            prompt["7"]["inputs"]["text_g"] = negative_prompt
            prompt["10"]["inputs"]["cfg"] = cfg_scale
            prompt["10"]["inputs"]["sampler_name"] = sampler
            prompt["10"]["inputs"]["scheduler"] = scheduler
            prompt["10"]["inputs"]["noise_seed"] = seed
            prompt["10"]["inputs"]["steps"] = generation_step
            prompt["10"]["inputs"]["start_at_step"] = diffusion_refiner_start
            prompt["11"]["inputs"]["text_l"] = positive_prompt
            prompt["11"]["inputs"]["text_g"] = positive_prompt
            prompt["12"]["inputs"]["text_l"] = negative_prompt
            prompt["12"]["inputs"]["text_g"] = negative_prompt
            prompt["13"]["inputs"]["ckpt_name"] = self.refiner
            if enable_clip_skip:
                prompt["14"]["inputs"]["stop_at_clip_layer"] = clip_skip
            
            base_node = "4"
            if enable_clip_skip:
                current_node_id = 15
            else:
                current_node_id = 14
            
            base_node, current_node_id = self._apply_loras(
                prompt, lora_text_weights, lora_unet_weights, base_node, current_node_id
            )
                
            prompt["3"]["inputs"]["model"] = [base_node, 0]
            if enable_clip_skip:
                prompt["14"]["inputs"]["clip"] = [base_node, 1]
            else:
                prompt["6"]["inputs"]["clip"] = [base_node, 1]
                prompt["7"]["inputs"]["clip"] = [base_node, 1]
            
            # Note: The original code had commented out refiner lora logic. I will keep it that way (omitted).
            
            _, current_node_id = self._apply_vae(prompt, base_node, current_node_id)
            
            generated = self.client.text2image_generate(prompt)
            
            history_data = {
                "Positive Prompt": positive_prompt,
                "Negative Prompt": negative_prompt,
                "Generation Steps": generation_step,
                "Model": self.model,
                "Sampler": sampler,
                "Scheduler": scheduler,
                "CFG Scale": cfg_scale,
                "Seed": seed,
                "Width": width,
                "Height": height
            }
            
            return self._process_results(generated, history_data)
        
        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return [], None

# Instantiate the pipeline for backward compatibility if needed, or just leave the class.
# The user asked to "refactor into a class".
# I will expose the class.