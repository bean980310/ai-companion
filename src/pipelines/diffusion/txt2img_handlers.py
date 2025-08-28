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

from src.api.comfy_api import ComfyUIClient
from workflows.load_workflow import load_txt2img_workflow, load_txt2img_sdxl_workflow, load_txt2img_sdxl_with_refiner_workflow, load_txt2img_workflow_clip_skip, load_txt2img_sdxl_workflow_clip_skip, load_txt2img_sdxl_with_refiner_workflow_clip_skip

from src import logger

# class TextToImageHandler:

def generate_images(
    positive_prompt: str,
    negative_prompt: str,
    style: str,
    generation_step: int,
    width: int,
    height: int,
    diffusion_model: str,
    diffusion_model_type: str,
    loras: List[str],
    vae: str,
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
):
    """
    UI에서 전달받은 파라미터를 바탕으로 실제 모델을 로드하고 이미지 생성 작업을 수행.
    - diffusion_model: 선택한 이미지 생성 모델 파일 혹은 모델 ID (예: "model.ckpt" 또는 diffusers model id)
    - diffusion_model_type: "checkpoint" 또는 "diffusers" 값 (UI에서 라디오 버튼 등으로 선택)
    - lora_selection: 선택한 LoRA 모델 리스트 (파일명 혹은 경로)
    - vae_selection: 선택한 VAE 모델 (파일명 혹은 경로)
    """
    try:
        # 모델 로딩 및 설정
        if diffusion_model_type != "diffusers":
            # diffusion_model 예: "checkpoints/sdxl/animagine-xl-4.0.safetensors"
            ckpt_value = diffusion_model.split("/", 1)[-1]  # "sdxl/animagine-xl-4.0.safetensors"
        else:
            ckpt_value = diffusion_model

        processed_loras = []
        for lora in loras:
            if lora.lower() != "none":
                processed_loras.append(lora.split("/", 1)[-1])  # "loras/xxx.safetensors"
                
        if random_seed:
            seed = random.randint(0, 9007199254740991)
            
        lora_text_weights = json.loads(lora_text_weights_json)
        lora_unet_weights = json.loads(lora_unet_weights_json)
        
        if enable_clip_skip:
            clip_skip=clip_skip*(-1)
        
        if clip_g:
            if enable_clip_skip:
                prompt=load_txt2img_sdxl_workflow_clip_skip()
            else:
                prompt=load_txt2img_sdxl_workflow()
        else:
            if enable_clip_skip:
                prompt=load_txt2img_workflow_clip_skip()
            else:
                prompt=load_txt2img_workflow()
        
        prompt["3"]["inputs"]["cfg"] = cfg_scale
        prompt["3"]["inputs"]["sampler_name"] = sampler
        prompt["3"]["inputs"]["scheduler"] = scheduler
        prompt["3"]["inputs"]["seed"] = seed
        prompt["3"]["inputs"]["steps"] = generation_step
        prompt["4"]["inputs"]["ckpt_name"] = ckpt_value
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
        
        for i, lora in enumerate(processed_loras):
            text_weight = lora_text_weights[i] if i < len(lora_text_weights) else 1.0
            unet_weight = lora_unet_weights[i] if i < len(lora_unet_weights) else 1.0
            new_node_id = str(current_node_id)
            prompt[new_node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora,
                    "strength_model": text_weight,
                    "strength_clip": unet_weight,
                    # "model"와 "clip" 입력은 이전 노드의 출력을 참조
                    "model": [base_node, 0],
                    "clip": [base_node, 1]
                }
            }
            # 체인 업데이트: 새로 추가된 노드가 base_node가 됨.
            base_node = new_node_id
            current_node_id += 1
         
        prompt["3"]["inputs"]["model"] = [base_node, 0]
        if enable_clip_skip:   
            prompt["10"]["inputs"]["clip"] = [base_node, 1]
        else:
            prompt["6"]["inputs"]["clip"] = [base_node, 1]
            prompt["7"]["inputs"]["clip"] = [base_node, 1]
        
        if vae == "Default":
            vae_value = None
        else:
            vae_value = vae.split("/", 1)[-1]
            new_node_id = str(current_node_id)
            prompt[new_node_id] = {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": vae_value
                }
            }
            base_node = new_node_id
            current_node_id += 1
            prompt["8"]["inputs"]["vae"] = [base_node, 0]
        
        # 실제 이미지 생성 로직은 각 모델에 맞게 다르게 호출됨.
        # 예시로 Diffusers 파이프라인인 경우:
        if diffusion_model_type == "diffusers":
            # Diffusers 모델의 경우 pipeline 객체에서 generate 호출 (예시)
            # generated = model(prompt=positive_prompt, negative_prompt=negative_prompt, width=width, height=height).images
            generated = ["diffusers_image_example"]  # 실제 생성 결과로 대체
        else:
            # checkpoint 모델일 경우 (예시)
            # generated = model.generate(prompt=positive_prompt, negative_prompt=negative_prompt, width=width, height=height, style=style)
            client=ComfyUIClient()
            generated = client.text2image_generate(prompt)  # 실제 생성 결과로 대체
        
        output_images=[]
        for node_id in generated:
            for image_data in generated[node_id]:
                from PIL import Image
                import io
                try:
                    image = Image.open(io.BytesIO(image_data))
                    output_images.append(image)
                    width, height = image.size
                except Exception as e:
                    logger.error(f"이미지 로딩 오류: {str(e)}\n\n{traceback.format_exc()}")
                    
        # 생성 기록 DataFrame 생성
        history_entry = {
            "Positive Prompt": positive_prompt,
            "Negative Prompt": negative_prompt,
            "Generation Steps": generation_step,
            "Model": diffusion_model,
            "Sampler": sampler,
            "Scheduler": scheduler,
            "CFG Scale": cfg_scale,
            "Seed": seed,
            "Width": width,
            "Height": height
        }
        
        history_df = pd.DataFrame([history_entry])
                
        return output_images, history_df
    
    except Exception as e:
        logger.error(f"이미지 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
        return [], None
    
def generate_images_with_refiner(
    positive_prompt: str,
    negative_prompt: str,
    style: str,
    generation_step: int,
    diffusion_refiner_start: int,
    width: int,
    height: int,
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
    batch_size: int,
    batch_count: int,
    cfg_scale: float,
    seed: int,
    random_seed: bool,
    lora_text_weights_json: str,
    lora_unet_weights_json: str
):
    """
    UI에서 전달받은 파라미터를 바탕으로 실제 모델을 로드하고 이미지 생성 작업을 수행.
    - diffusion_model: 선택한 이미지 생성 모델 파일 혹은 모델 ID (예: "model.ckpt" 또는 diffusers model id)
    - diffusion_model_type: "checkpoint" 또는 "diffusers" 값 (UI에서 라디오 버튼 등으로 선택)
    - lora_selection: 선택한 LoRA 모델 리스트 (파일명 혹은 경로)
    - vae_selection: 선택한 VAE 모델 (파일명 혹은 경로)
    """
    try:
        # 모델 로딩 및 설정
        if diffusion_model_type != "diffusers":
            # diffusion_model 예: "checkpoints/sdxl/animagine-xl-4.0.safetensors"
            ckpt_value = diffusion_model.split("/", 1)[-1]  # "sdxl/animagine-xl-4.0.safetensors"
        else:
            ckpt_value = diffusion_model
            
        refiner_ckpt_value = diffusion_refiner_model.split("/", 1)[-1]

        processed_loras = []
        for lora in loras:
            if lora.lower() != "none":
                processed_loras.append(lora.split("/", 1)[-1])  # "loras/xxx.safetensors"
                
        if random_seed:
            seed = random.randint(0, 9007199254740991)
            
        lora_text_weights = json.loads(lora_text_weights_json)
        lora_unet_weights = json.loads(lora_unet_weights_json)
        
        if enable_clip_skip:
            clip_skip=clip_skip*(-1)
        
        if enable_clip_skip:
            prompt=load_txt2img_sdxl_with_refiner_workflow_clip_skip()
        else:
            prompt=load_txt2img_sdxl_with_refiner_workflow()
        
        prompt["3"]["inputs"]["cfg"] = cfg_scale
        prompt["3"]["inputs"]["sampler_name"] = sampler
        prompt["3"]["inputs"]["scheduler"] = scheduler
        prompt["3"]["inputs"]["noise_seed"] = seed
        prompt["3"]["inputs"]["steps"] = generation_step
        prompt["3"]["inputs"]["end_at_step"] = diffusion_refiner_start
        prompt["4"]["inputs"]["ckpt_name"] = ckpt_value
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
        prompt["13"]["inputs"]["ckpt_name"] = refiner_ckpt_value
        if enable_clip_skip:
            prompt["14"]["inputs"]["stop_at_clip_layer"] = clip_skip
        
        base_node = "4"
        refiner_base_node = "13"
        if enable_clip_skip:
            current_node_id = 15
        else:
            current_node_id = 14
        
        for i, lora in enumerate(processed_loras):
            text_weight = lora_text_weights[i] if i < len(lora_text_weights) else 1.0
            unet_weight = lora_unet_weights[i] if i < len(lora_unet_weights) else 1.0
            new_node_id = str(current_node_id)
            prompt[new_node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora,
                    "strength_model": text_weight,
                    "strength_clip": unet_weight,
                    # "model"와 "clip" 입력은 이전 노드의 출력을 참조
                    "model": [base_node, 0],
                    "clip": [base_node, 1]
                }
            }
            # 체인 업데이트: 새로 추가된 노드가 base_node가 됨.
            base_node = new_node_id
            current_node_id += 1
            
        prompt["3"]["inputs"]["model"] = [base_node, 0]
        if enable_clip_skip:
            prompt["14"]["inputs"]["clip"] = [base_node, 1]
        else:
            prompt["6"]["inputs"]["clip"] = [base_node, 1]
            prompt["7"]["inputs"]["clip"] = [base_node, 1]
        
        # for i, lora in enumerate(processed_loras):
        #     text_weight = lora_text_weights[i] if i < len(lora_text_weights) else 1.0
        #     unet_weight = lora_unet_weights[i] if i < len(lora_unet_weights) else 1.0
        #     new_node_id = str(current_node_id)
        #     prompt[new_node_id] = {
        #         "class_type": "LoraLoader",
        #         "inputs": {
        #             "lora_name": lora,
        #             "strength_model": text_weight,
        #             "strength_clip": unet_weight,
        #             # "model"와 "clip" 입력은 이전 노드의 출력을 참조
        #             "model": [refiner_base_node, 0],
        #             "clip": [refiner_base_node, 1]
        #         }
        #     }
        #     # 체인 업데이트: 새로 추가된 노드가 base_node가 됨.
        #     refiner_base_node = new_node_id
        #     current_node_id += 1
          
        # prompt["10"]["inputs"]["model"] = [refiner_base_node, 0]  
        
        if vae == "Default":
            vae_value = None
        else:
            vae_value = vae.split("/", 1)[-1]
            new_node_id = str(current_node_id)
            prompt[new_node_id] = {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": vae_value
                }
            }
            base_node = new_node_id
            current_node_id += 1
            prompt["8"]["inputs"]["vae"] = [base_node, 0]
        
        # 실제 이미지 생성 로직은 각 모델에 맞게 다르게 호출됨.
        # 예시로 Diffusers 파이프라인인 경우:
        if diffusion_model_type == "diffusers":
            # Diffusers 모델의 경우 pipeline 객체에서 generate 호출 (예시)
            # generated = model(prompt=positive_prompt, negative_prompt=negative_prompt, width=width, height=height).images
            generated = ["diffusers_image_example"]  # 실제 생성 결과로 대체
        else:
            # checkpoint 모델일 경우 (예시)
            # generated = model.generate(prompt=positive_prompt, negative_prompt=negative_prompt, width=width, height=height, style=style)
            client=ComfyUIClient()
            generated = client.text2image_generate(prompt)  # 실제 생성 결과로 대체
        
        output_images=[]
        for node_id in generated:
            for image_data in generated[node_id]:
                from PIL import Image
                import io
                try:
                    image = Image.open(io.BytesIO(image_data))
                    output_images.append(image)
                    width, height = image.size
                except Exception as e:
                    logger.error(f"이미지 로딩 오류: {str(e)}\n\n{traceback.format_exc()}")
                    
        # 생성 기록 DataFrame 생성
        history_entry = {
            "Positive Prompt": positive_prompt,
            "Negative Prompt": negative_prompt,
            "Generation Steps": generation_step,
            "Model": diffusion_model,
            "Sampler": sampler,
            "Scheduler": scheduler,
            "CFG Scale": cfg_scale,
            "Seed": seed,
            "Width": width,
            "Height": height
        }
        
        history_df = pd.DataFrame([history_entry])
                
        return output_images, history_df
    
    except Exception as e:
        logger.error(f"이미지 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
        return [], None