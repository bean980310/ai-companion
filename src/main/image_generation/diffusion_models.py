from typing import List
import pandas as pd
import random
import logging

logger = logging.getLogger(__name__)

def load_ckpt_model(model_path: str):
    # ckpt 파일을 로드하는 예제 함수 (예: torch.load 활용)
    logger.info(f"Loading checkpoint model from {model_path}")
    # 실제 ckpt 로딩 코드를 여기에 구현
    return "ckpt_model_object"

def load_safetensors_model(model_path: str):
    # safetensors 파일을 로드하는 예제 함수 (예: safetensors 라이브러리 활용)
    logger.info(f"Loading safetensors model from {model_path}")
    # 실제 safetensors 로딩 코드를 여기에 구현
    return "safetensors_model_object"

def load_diffusers_pipeline(model_id: str):
    # Diffusers 모델 로드 (예: Huggingface diffusers 라이브러리 활용)
    logger.info(f"Loading diffusers model {model_id}")
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(model_id)
    return pipe

def apply_lora(model, lora_path: str, weight: float = 1.0):
    # LoRA 가중치 적용 예제 함수
    logger.info(f"Applying LoRA from {lora_path} with weight {weight}")
    # 실제 LoRA 적용 로직 구현 (모델의 파라미터에 LoRA를 합치는 코드 등)
    return model

def load_vae(model, vae_path: str, weight: float = 1.0):
    # VAE 모델 로드 및 적용 예제 함수
    logger.info(f"Loading VAE from {vae_path} with weight {weight}")
    # 실제 VAE 로딩 및 모델에 적용하는 코드를 구현
    return model

def load_image_generation_model(model_path: str, model_type: str, lora_paths: List[str], vae_path: str):
    """
    선택된 모델(Checkpoint 또는 Diffusers)을 로드하고,
    선택한 LoRA와 VAE를 적용하여 최종 모델 객체를 반환하는 함수.
    model_type은 "diffusers" 또는 "checkpoint" 값으로 구분.
    """
    # 1. 기본 모델 로드
    if model_type == "diffusers":
        model = load_diffusers_pipeline(model_path)
    else:
        # checkpoint 모델일 경우 파일 확장자에 따라 로딩
        if model_path.endswith(".ckpt"):
            model = load_ckpt_model(model_path)
        elif model_path.endswith(".safetensors"):
            model = load_safetensors_model(model_path)
        else:
            raise ValueError("Unsupported model file format for checkpoint.")
    
    # 2. LoRA 적용 (선택된 여러 LoRA 파일에 대해 반복)
    for lora_path in lora_paths:
        # 여기서 weight는 UI에서 개별 LoRA에 대해 조정한 값(기본 1.0)으로 받을 수 있음
        model = apply_lora(model, lora_path, weight=1.0)
    
    # 3. VAE 적용 (선택한 VAE가 있다면)
    if vae_path and vae_path != "None":
        model = load_vae(model, vae_path, weight=1.0)
    
    return model