# models.py

import random
import platform
import traceback
from typing import Any, List, Literal

import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageFile

import openai
import anthropic
from google import genai
from google.genai import types

from src.common.cache import models_cache

from ai_companion_llm_backend import (
    TransformersCausalModelHandler, 
    TransformersVisionModelHandler, 
    GGUFCausalModelHandler, 
    MlxCausalModelHandler, 
    MlxVisionModelHandler,
)

from ai_companion_llm_backend.provider import (
    AnthropicClientWrapper,
    GoogleAIClientWrapper,
    OpenAIClientWrapper,
    PerplexityClientWrapper,
    XAIClientWrapper,
    OpenRouterClientWrapper,
    HuggingfaceInferenceClientWrapper,
    LMStudioIntegrator
)
from src.common.utils import ensure_model_available, build_model_cache_key, get_all_local_models, convert_folder_to_modelid

from src import logger

LOCAL_MODELS_ROOT = "./models"

def get_default_device():
    """
    Automatically selects the best available device:
    - CUDA if NVIDIA GPU is available.
    - MPS if Apple Silicon (M-Series) is available.
    - CPU otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
# Set default device
default_device = get_default_device()
logger.info(f"Default device set to: {default_device}")

def load_model(selected_model: str, provider: Literal["openai", "anthropic", "google-genai", "perplexity", "xai", 'mistralai', "openrouter", "hf-inference", "ollama", "lmstudio", "self-provided"], model_type: str | None = None, selected_lora: str | None = None, quantization_bit: str = "Q8_0", local_model_path: str | None = None, api_key: str | None = None, device: str = "cpu", lora_path: str | None = None, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
    """
    모델 로드 함수. 특정 모델에 대한 로드 로직을 외부 핸들러로 분리.
    """
    model_id = selected_model
    lora_model_id = selected_lora if selected_lora else None
    vision_model = True if image_input else False
    
    # Pass the device to the handler
    if provider == "self-provided" and model_type != "transformers" and model_type != "gguf" and model_type != "mlx":
        logger.error(f"지원되지 않는 모델 유형: {model_type}")
        return None
    
    if provider not in ["openai", "anthropic", "google-genai", "perplexity", "xai", "mistralai", "openrouter", "hf-inference", "ollama", "lmstudio", "self-provided"]:
        logger.error(f"지원되지 않는 공급자: {provider}")
        return None
    
    # 각자 공급자에 따라 클라이언트 생성.
    if provider == "openai":
        if not api_key:
            logger.error("OpenAI API Key가 missing.")
            return "OpenAI API Key가 필요합니다."
            
        wrapper = OpenAIClientWrapper(selected_model, api_key=api_key)
        return wrapper
    
    elif provider == "anthropic":
        if not api_key:
            logger.error("Anthropic API Key가 missing.")
            return "Anthropic API Key가 필요합니다."
            
        wrapper = AnthropicClientWrapper(selected_model, api_key=api_key)
        return wrapper
    
    elif provider == "google-genai":
        if not api_key:
            logger.error("Google AI API Key가 missing.")
            return "Google AI API Key가 필요합니다."

        wrapper = GoogleAIClientWrapper(selected_model, api_key=api_key)
        return wrapper
    
    elif provider == "perplexity":
        if not api_key:
            logger.error("Perplexity API Key가 missing.")
            return "Perplexity API Key가 필요합니다."
            
        wrapper = PerplexityClientWrapper(selected_model, api_key=api_key)
        return wrapper

    elif provider == "xai":
        if not api_key:
            logger.error("XAI API Key가 missing.")
            return "XAI API Key가 필요합니다."
            
        wrapper = XAIClientWrapper(selected_model, api_key=api_key)
        return wrapper
    
    elif provider == "openrouter":
        if not api_key:
            logger.error("OpenRouter API Key가 missing.")
            return "OpenRouter API Key가 필요합니다."
            
        wrapper = OpenRouterClientWrapper(selected_model, api_key=api_key)
        return wrapper

    elif provider == "hf-inference":
        if not api_key:
            logger.error("Huggingface Token Key가 missing.")
            return "Huggingface Token Key가 필요합니다."
            
        wrapper = HuggingfaceInferenceClientWrapper(selected_model, api_key=api_key)
        return wrapper
    elif provider == "ollama":
        # API 모델은 별도의 로드가 필요 없으므로 핸들러 생성 안함
        return None
    elif provider == "lmstudio":
        # API 모델은 별도의 로드가 필요 없으므로 핸들러 생성 안함
        wrapper = LMStudioIntegrator(selected_model)
        return wrapper
    else:
        # 자체 공급은 클라이언트 대신 핸들러를 생성.
        handler = None
        if model_type == "gguf":
            # GGUF 모델 로딩 로직
            # handler = GGUFModelHandler(
            #     model_id=model_id,
            #     local_model_path=local_model_path,
            #     model_type=model_type
            # )
            handler = GGUFCausalModelHandler(
                model_id=model_id,
                lora_model_id=lora_model_id,
                model_type=model_type,
                device=device, 
                **kwargs
            )
            cache_key = build_model_cache_key(model_id, model_type)
            models_cache[cache_key] = handler
            return handler
        elif model_type == "mlx":
            if vision_model:
                handler = MlxVisionModelHandler(
                    model_id=model_id,
                    lora_model_id=lora_model_id,
                    model_type=model_type,
                    image_input=image_input,
                    **kwargs
                )
                models_cache[build_model_cache_key(model_id, model_type, lora_model_id)] = handler
                return handler
            else:
                handler = MlxCausalModelHandler(
                    model_id=model_id,
                    lora_model_id=lora_model_id,
                    model_type=model_type, 
                    **kwargs
                )
                models_cache[build_model_cache_key(model_id, model_type, lora_model_id)] = handler
                return handler
        else:
            if vision_model:
                handler = TransformersVisionModelHandler(
                    model_id=model_id,
                    lora_model_id=lora_model_id,
                    model_type=model_type,
                    device=device,
                    image_input=image_input, 
                    **kwargs
                )
                models_cache[build_model_cache_key(model_id, model_type, lora_model_id)] = handler
                return handler
            else:
                handler = TransformersCausalModelHandler(
                    model_id=model_id, 
                    lora_model_id=lora_model_id, 
                    model_type=model_type,
                    device=device, 
                    **kwargs
                )
                models_cache[build_model_cache_key(model_id, model_type, lora_model_id)] = handler
                return handler

def generate_answer(history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]], selected_model: str, provider: Literal["openai", "anthropic", "google-genai", "perplexity", "xai", 'mistralai', "openrouter", "hf-inference", "ollama", "lmstudio", "self-provided"], model_type: str | None = None, selected_lora: str | None = None, local_model_path: str | None = None, lora_path: str | None = None, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, api_key: str | None = None, device: str = "cpu", seed: int = 42, max_length: int = -1, temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0, repetition_penalty: float = 1.0, enable_thinking: bool = False, enable_streaming: bool = False, character_language: str = 'ko'):
    """
    사용자 히스토리를 기반으로 답변 생성.
    """
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    else:
        torch.manual_seed(seed)
        
    if not history:
        system_message = {
            "role": "system",
            "content": "당신은 유용한 AI 비서입니다."
        }
        history = [system_message]
    
    cache_key = build_model_cache_key(selected_model, model_type, selected_lora, local_path=local_model_path)
    handler = models_cache.get(cache_key)
    
    last_message = history[-1]
    if last_message["role"] == "assistant":
        last_message = history[-2]

    if not handler:
        logger.info(f"[*] 모델 공급자: {provider}")
        logger.info(f"[*] 모델 로드 중: {selected_model}")
        handler = load_model(selected_model, provider, model_type, selected_lora, local_model_path=local_model_path, device=device, lora_path=lora_path, image_input=image_input, seed=seed, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, enable_thinking=enable_thinking)
        
    if not handler:
        logger.error("모델 핸들러가 로드되지 않았습니다.")
        return "모델 핸들러가 로드되지 않았습니다."
        
    logger.info(f"[*] Generating answer using {handler.__class__.__name__}")
    try:
        answer = handler.generate_answer(history)
        return answer
    except Exception as e:
        logger.error(f"모델 추론 오류: {str(e)}\n\n{traceback.format_exc()}")
        return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
        
def generate_text(history, selected_model, model_type, selected_lora=None, local_model_path=None, lora_path=None, image_input=None, api_key=None, device="cpu", seed=42, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0, character_language='ko'):
    """
    사용자 히스토리를 기반으로 답변 생성.
    """
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    else:
        torch.manual_seed(seed)
        
    if not history:
        system_message = {
            "role": "system",
            "content": "당신은 유용한 AI 비서입니다."
        }
        history = [system_message]
    
    cache_key = build_model_cache_key(selected_model, model_type, selected_lora, local_path=local_model_path)
    handler = models_cache.get(cache_key)
    
    last_message = history[-1]
    if last_message["role"] == "assistant":
        last_message = history[-2]
    
    # if character_language == "ko":
    #     # 한국어 모델 사용
    #     model_id = "klue/bert-base"  # 예시
    # elif character_language == "en":
    #     # 영어 모델 사용
    #     model_id = "bert-base-uncased"  # 예시
    # elif character_language == "ja":
    #     # 일본어 모델 사용
    #     model_id = "tohoku-nlp/bert-base-japanese"  # 예시"
    # else:
    #     # 기본 모델 사용
    #     model_id = "gpt-3.5-turbo"
    
    if model_type == "api":
        if "claude" in selected_model:
            if not api_key:
                logger.error("Anthropic API Key가 missing.")
                return "Anthropic API Key가 필요합니다."
            
            client = anthropic.Client(api_key=api_key)
            # Anthropic 메시지 형식으로 변환
            messages = []
            for msg in history:
                if msg["role"] == "system":
                    continue  # Claude API는 시스템 메시지를 별도로 처리하지 않음
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            logger.info(f"[*] Anthropic API 요청: {messages}")
            
            try:
                response = client.messages.create(
                    model=selected_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024
                )
                answer = response.content[0].text
                logger.info(f"[*] Anthropic 응답: {answer}")
                return answer
            except Exception as e:
                logger.error(f"Anthropic API 오류: {str(e)}\n\n{traceback.format_exc()}")
                return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
        elif "gemini" in selected_model:
            if not api_key:
                logger.error("Google API Key가 missing.")
                return "Google API Key가 필요합니다."

            client = genai.Client(api_key=api_key)
            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            config = types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.7,
                top_p=0.9
            )
            logger.info(f"[*] Google API 요청: {messages}")
            try: 
                response = client.models.generate_content(
                    model=selected_model,
                    contents=messages,
                    config=config
                )
                answer = response.text
                logger.info(f"[*] Google 응답: {answer}")
                return answer
            except Exception as e:
                logger.error(f"Google API 오류: {str(e)}\n\n{traceback.format_exc()}")
                return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
        else:
            if not api_key:
                logger.error("OpenAI API Key가 missing.")
                return "OpenAI API Key가 필요합니다."
            openai.api_key = api_key
            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] OpenAI API 요청: {messages}")
            
            try:
                response = openai.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.9,
                )
                answer = response.choices[0].message["content"]
                logger.info(f"[*] OpenAI 응답: {answer}")
                return answer
            except Exception as e:
                logger.error(f"OpenAI API 오류: {str(e)}\n\n{traceback.format_exc()}")
                return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
    
    else:
        if not handler:
            logger.info(f"[*] 모델 로드 중: {selected_model}")
            handler = load_model(selected_model, model_type, selected_lora, local_model_path=local_model_path, device=device, lora_path=lora_path)
        
        if not handler:
            logger.error("모델 핸들러가 로드되지 않았습니다.")
            return "모델 핸들러가 로드되지 않았습니다."
        
        logger.info(f"[*] Generating answer using {handler.__class__.__name__}")
        try:
            if isinstance(handler, TransformersVisionModelHandler) or isinstance(handler, MlxVisionModelHandler):
                answer = handler.generate_answer(history, image_input, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
            else:
                answer = handler.generate_answer(history, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
            return answer
        except Exception as e:
            logger.error(f"모델 추론 오류: {str(e)}\n\n{traceback.format_exc()}")
            return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
        
def generate_chat_title(first_message, selected_model, model_type, selected_lora=None, local_model_path=None, lora_path=None, device="cpu", image_input=None):
    """
    첫 번째 메시지를 기반으로 채팅 제목을 생성하는 함수.
    모델 핸들러에 generate_chat_title 메서드가 구현되어 있어야 함.
    """
    cache_key = build_model_cache_key(selected_model, model_type, selected_lora, local_path=local_model_path)
    handler = models_cache.get(cache_key)
    
    if not handler:
        logger.info(f"[*] 모델 로드 중: {selected_model}")
        handler = load_model(selected_model, model_type, selected_lora, local_model_path=local_model_path, device=device, lora_path=lora_path)
    
    if not handler:
        logger.error("모델 핸들러가 로드되지 않았습니다.")
        return "모델 핸들러가 로드되지 않았습니다."
    
    logger.info(f"[*] Generating chat title using {handler.__class__.__name__}")
    try:
        if hasattr(handler, "generate_chat_title"):
            if image_input:
                title = handler.generate_chat_title(first_message, image_input)
            else:
                title = handler.generate_chat_title(first_message)
            return title
        else:
            logger.error("모델 핸들러에 채팅 제목 생성 기능이 없습니다.")
            return "채팅 제목 생성 기능이 없습니다."
    except Exception as e:
        logger.error(f"채팅 제목 생성 오류: {str(e)}\n\n{traceback.format_exc()}")
        return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
    
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
    image_input: str | Image.Image | ImageFile.ImageFile | Any,
    denoise_strength: float,
    blur_radius: float,
    blur_expansion_radius: int,
    lora_text_weights_json: str,
    lora_unet_weights_json: str,
):
    use_comfyui = True if diffusion_model_type == "checkpoints" else False