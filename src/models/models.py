# models.py
import random
import platform
import traceback
from typing import Any, List, Literal, Optional

import numpy as np
import torch
from PIL import Image, ImageFile

import openai
import anthropic
from google import genai
from google.genai import types

from src.common.cache import models_cache

from ai_companion_llm_backend import TransformersVisionModelHandler, TransformersUnifiedModelHandler, GGUFCausalModelHandler, MlxVisionModelHandler, MlxUnifiedModelHandler

import ai_companion_gateway as gateway

from src.common.utils import build_model_cache_key
from src.common.custom_providers import CUSTOM_PREFIX, get_custom_provider, get_custom_provider_names
from src.models.custom_provider_handler import CustomOpenAIClientWrapper

from ai_companion_image_backend.provider.comfyui import ComfyUIProvider, Txt2ImgPipeline, Img2ImgPipeline, InpaintPipeline

from ai_companion_core import logger

from ..common.utils import get_all_local_models

LOCAL_MODELS_ROOT = "./models"

# 로컬 추론(transformers/gguf/mlx 핸들러)을 사용하는 프로바이더.
# 그 외 프로바이더는 전부 ai-companion-gateway의 클라이언트 팩토리로 위임된다.
LOCAL_MODEL_PROVIDERS = ("self-provided",)


def _register_custom_gateway_providers():
    """~/.ai-companion/custom_providers.yaml 의 프로필을 gateway에 등록한다.

    이미 등록된 프로필은 건너뛴다. 프로필을 추가/수정하면
    gateway.unregister_client_builder() 후 다시 호출해 갱신할 수 있다.
    """
    for profile_name in get_custom_provider_names():
        provider_id = f"{CUSTOM_PREFIX}{profile_name}"
        if provider_id in gateway.supported_providers():
            continue
        profile = get_custom_provider(profile_name)
        if not profile:
            continue

        def _build(selected_model, api_key, _profile=profile, **kwargs):
            return CustomOpenAIClientWrapper(
                selected_model,
                api_key=api_key or _profile.get("api_key") or "not-needed",
                base_url=_profile.get("base_url"),
                **kwargs,
            )

        gateway.register_client_builder(provider_id, _build, requires_api_key=False)


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


def load_model(
    selected_model: str,
    provider: Literal["openai", "anthropic", "google-genai", "perplexity", "xai", "mistralai", "openrouter", "hf-inference", "ollama", "lmstudio", "vllm", "oobabooga", "self-provided"],
    model_type: str | None = None,
    selected_lora: str | None = None,
    quantization_bit: str = "Q8_0",
    local_model_path: str | None = None,
    api_key: str | None = None,
    device: str = "cpu",
    lora_path: str | None = None,
    image_input: str | List[str] | Image.Image | List[Image.Image] | ImageFile.ImageFile | List[ImageFile.ImageFile] | Any | None = None,
    audio_input: str | List[str] | Any | None = None,
    video_input: str | List[str] | Any | None = None,
    **kwargs,
):
    """
    모델 로드 함수.

    - API/서버 프로바이더: ai-companion-gateway의 클라이언트 팩토리로 위임.
      (프로바이더별 if/elif 분기 대신 gateway.create_client()가 단일 진입점)
    - self-provided(로컬 추론): transformers / gguf / mlx 핸들러를 직접 생성.
    """
    model_id = selected_model
    lora_model_id = selected_lora if selected_lora else None

    # 커스텀 OpenAI 호환 프로필(custom:<name>)은 gateway에 빌더로 등록해 사용한다.
    if provider.startswith(CUSTOM_PREFIX):
        _register_custom_gateway_providers()
        profile_name = provider[len(CUSTOM_PREFIX):]
        if not get_custom_provider(profile_name):
            logger.error(f"사용자 정의 provider를 찾을 수 없습니다: {profile_name}")
            return "사용자 정의 provider를 찾을 수 없습니다."

    if provider == "oobabooga":
        # API 모델은 별도의 로드가 필요 없으므로 핸들러 생성 안함
        return None

    if provider not in LOCAL_MODEL_PROVIDERS:
        # gateway가 처리하는 프로바이더 (openai/anthropic/ollama/lmstudio/vllm/custom:* 등)
        try:
            return gateway.create_client(selected_model, provider, api_key=api_key, **kwargs)
        except ValueError as e:
            logger.error(str(e))
            return str(e)

    # 이하: self-provided 로컬 추론 핸들러
    if model_type not in ("transformers", "gguf", "mlx"):
        logger.error(f"지원되지 않는 모델 유형: {model_type}")
        return None

    vision_model = True if image_input else False

    # Pass the device to the handler
    # 자체 공급은 클라이언트 대신 핸들러를 생성.
    handler = None
    if model_type == "gguf":
        # GGUF 모델 로딩 로직
        # handler = GGUFModelHandler(
        #     model_id=model_id,
        #     local_model_path=local_model_path,
        #     model_type=model_type
        # )
        handler = GGUFCausalModelHandler(model_id=model_id, lora_model_id=lora_model_id, model_type=model_type, device=device, **kwargs)
        cache_key = build_model_cache_key(model_id, model_type, provider)
        models_cache[cache_key] = handler
        return handler
    elif model_type == "mlx":
        handler = MlxUnifiedModelHandler(model_id=model_id, lora_model_id=lora_model_id, model_type=model_type, image_input=image_input, audio_input=audio_input, video_input=video_input, **kwargs)
        models_cache[build_model_cache_key(model_id, model_type, provider, lora_model_id)] = handler
        return handler
        # if vision_model:
        #     handler = MlxVisionModelHandler(
        #         model_id=model_id,
        #         lora_model_id=lora_model_id,
        #         model_type=model_type,
        #         image_input=image_input,
        #         **kwargs
        #     )
        #     models_cache[build_model_cache_key(model_id, model_type, lora_model_id)] = handler
        #     return handler
        # else:
        #     handler = MlxCausalModelHandler(
        #         model_id=model_id,
        #         lora_model_id=lora_model_id,
        #         model_type=model_type,
        #         **kwargs
        #     )
        #     models_cache[build_model_cache_key(model_id, model_type, lora_model_id)] = handler
        #     return handler
    else:
        handler = TransformersUnifiedModelHandler(model_id=model_id, lora_model_id=lora_model_id, model_type=model_type, device=device, image_input=image_input, audio_input=audio_input, video_input=video_input, **kwargs)
        models_cache[build_model_cache_key(model_id, model_type, provider, lora_model_id)] = handler
        return handler
        # if vision_model:
        #     handler = TransformersVisionModelHandler(
        #         model_id=model_id   ,
        #         lora_model_id=lora_model_id,
        #         model_type=model_type,
        #         device=device,
        #         image_input=image_input,
        #         **kwargs
        #     )
        #     models_cache[build_model_cache_key(model_id, model_type, lora_model_id)] = handler
        #     return handler
        # else:
        #     handler = TransformersCausalModelHandler(
        #         model_id=model_id,
        #         lora_model_id=lora_model_id,
        #         model_type=model_type,
        #         device=device,
        #         **kwargs
        #     )
        #     models_cache[build_model_cache_key(model_id, model_type, lora_model_id)] = handler
        #     return handler


def generate_answer(
    history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]],
    selected_model: str,
    provider: Literal["openai", "anthropic", "google-genai", "perplexity", "xai", "mistralai", "openrouter", "hf-inference", "ollama", "lmstudio", "oobabooga", "self-provided"],
    model_type: str | None = None,
    selected_lora: str | None = None,
    local_model_path: str | None = None,
    lora_path: str | None = None,
    image_input: str | list[str] | Image.Image | ImageFile.ImageFile | Any | None = None,
    audio_input: str | list[str] | Any | None = None,
    video_input: str | list[str] | Any | None = None,
    api_key: str | None = None,
    device: str = "cpu",
    seed: int = 42,
    max_length: int = -1,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    enable_thinking: bool = False,
    enable_streaming: bool = False,
    character_language: str = "ko",
):
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
        system_message = {"role": "system", "content": "당신은 유용한 AI 비서입니다."}
        history = [system_message]

    cache_key = build_model_cache_key(selected_model, model_type, provider, selected_lora, local_path=local_model_path)
    handler = models_cache.get(cache_key)

    last_message = history[-1]
    if last_message["role"] == "assistant":
        last_message = history[-2]

    if not handler:
        logger.info(f"[*] 모델 공급자: {provider}")
        logger.info(f"[*] 모델 로드 중: {selected_model}")
        handler = load_model(
            selected_model,
            provider,
            model_type,
            selected_lora,
            local_model_path=local_model_path,
            device=device,
            lora_path=lora_path,
            image_input=image_input,
            audio_input=audio_input,
            video_input=video_input,
            seed=seed,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
        )

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


def generate_text(history, selected_model, model_type, selected_lora=None, local_model_path=None, lora_path=None, image_input=None, api_key=None, device="cpu", seed=42, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0, character_language="ko"):
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
        system_message = {"role": "system", "content": "당신은 유용한 AI 비서입니다."}
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
                messages.append({"role": msg["role"], "content": msg["content"]})

            logger.info(f"[*] Anthropic API 요청: {messages}")

            try:
                response = client.messages.create(model=selected_model, messages=messages, temperature=0.7, max_tokens=1024)
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
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
            config = types.GenerateContentConfig(max_output_tokens=1024, temperature=0.7, top_p=0.9)
            logger.info(f"[*] Google API 요청: {messages}")
            try:
                response = client.models.generate_content(model=selected_model, contents=messages, config=config)
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
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
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


def generate_chat_title(first_message, selected_model, model_type, provider, selected_lora=None, local_model_path=None, lora_path=None, device="cpu", image_input=None, **kwargs):
    """
    첫 번째 메시지를 기반으로 채팅 제목을 생성하는 함수.
    모델 핸들러에 generate_chat_title 메서드가 구현되어 있어야 함.
    """
    cache_key = build_model_cache_key(selected_model, model_type, provider, selected_lora, local_path=local_model_path)
    handler = models_cache.get(cache_key)

    if not handler:
        logger.info(f"[*] 모델 로드 중: {selected_model}")
        handler = load_model(selected_model, provider, model_type, selected_lora, local_model_path=local_model_path, device=device, lora_path=lora_path, image_input=image_input)

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


def create_comfyui_pipeline(image_to_image_mode: str, model: str, refiner_model: str = "None", loras: Optional[List[str]] = None, vae: str = "Default", host: str = "127.0.0.1", port: int = 8188) -> Txt2ImgPipeline | Img2ImgPipeline | InpaintPipeline:
    """
    Create a ComfyUI pipeline based on the generation mode.

    Args:
        image_to_image_mode: Generation mode ("None" for txt2img, "Image to Image", or "Inpaint")
        model: Main checkpoint model name
        refiner_model: Refiner model name (for SDXL)
        loras: List of LoRA model names
        vae: VAE model name or 'Default'
        host: ComfyUI server host
        port: ComfyUI server port

    Returns:
        The appropriate pipeline instance (Txt2ImgPipeline, Img2ImgPipeline, or InpaintPipeline)
    """
    if loras is None:
        loras = []

    # Normalize refiner_model value
    refiner = None if refiner_model == "None" else refiner_model

    # Create the appropriate pipeline based on mode
    if image_to_image_mode == "None":
        logger.info(f"Creating Txt2ImgPipeline with model={model}, refiner={refiner}")
        pipeline = Txt2ImgPipeline(model=model, refiner=refiner, loras=loras, vae=vae)
    elif image_to_image_mode == "Image to Image":
        logger.info(f"Creating Img2ImgPipeline with model={model}, refiner={refiner}")
        pipeline = Img2ImgPipeline(model=model, refiner=refiner, loras=loras, vae=vae)
    elif image_to_image_mode == "Inpaint":
        logger.info(f"Creating InpaintPipeline with model={model}, refiner={refiner}")
        pipeline = InpaintPipeline(model=model, refiner=refiner, loras=loras, vae=vae)
    else:
        raise ValueError(f"Unknown image_to_image_mode: {image_to_image_mode}")

    return pipeline


def create_comfyui_provider(model: str, refiner_model: str = "None", loras: Optional[List[str]] = None, vae: str = "Default", host: str = "127.0.0.1", port: int = 8188) -> ComfyUIProvider:
    """
    Create a ComfyUI provider instance.

    Args:
        model: Main checkpoint model name
        refiner_model: Refiner model name (for SDXL)
        loras: List of LoRA model names
        vae: VAE model name or 'Default'
        host: ComfyUI server host
        port: ComfyUI server port

    Returns:
        Configured ComfyUIProvider instance
    """
    if loras is None:
        loras = []

    # Normalize refiner_model value
    refiner = None if refiner_model == "None" else refiner_model

    logger.info(f"Creating ComfyUIProvider with model={model}, refiner={refiner}")

    provider = ComfyUIProvider(model=model, model_type="checkpoint", refiner=refiner, loras=loras, vae=vae, host=host, port=port)

    return provider


__all__ = ["get_all_local_models"]
