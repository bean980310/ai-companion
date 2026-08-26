from typing import Optional


from ai_companion_core import logger
from ai_companion_core.environ_manager import load_env_variables


class LocalModelNotFound(Exception):
    pass


class ServerNotRunning(Exception):
    pass


def get_comfyui_image_models(url: str = "localhost:8188", folder: str = "checkpoints"):
    from comfy_client import ComfyUI

    model_list = []
    client = ComfyUI(server_url=url)

    try:
        model = client.models.list(folder=folder)

        if len(model) == 0:
            raise LocalModelNotFound(f"{folder} 모델이 존재하지 않습니다.")

        for m in model:
            model_list.append(m)

        logger.info(f"ComfyUI {folder} 모델 목록: {model_list}")

        return model_list
    except LocalModelNotFound:
        logger.error(f"{folder} 모델이 존재하지 않습니다.")
        return [f"{folder} 모델이 존재하지 않습니다."]
    except ServerNotRunning:
        logger.error("ComfyUI를 설치하고 서버를 실행해주세요.")
        return ["ComfyUI를 설치하고 서버를 실행해주세요."]


def get_sglang_image_models(api_host: str = "http://localhost:30001/v1"):
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        logger.error("openai가 설치되지 않았습니다.")
        return ["openai가 설치되지 않았습니다."]

    model_list = []
    client = OpenAI(api_key="not-needed", base_url=api_host)

    try:
        model = client.models.list()

        if len(model.data) == 0:
            raise LocalModelNotFound("모델이 존재하지 않습니다.")

        for m in model.data:
            model_list.append(m.id)

        logger.info(f"sglang 이미지 모델 목록: {model_list}")

        return model_list
    except LocalModelNotFound:
        logger.error("모델이 존재하지 않습니다.")
        return ["모델이 존재하지 않습니다."]
    except openai.PermissionDeniedError:
        logger.error("sglang을 설치하고 서버를 실행해주세요.")
        return ["sglang을 설치하고 서버를 실행해주세요."]
    except openai.APIConnectionError:
        logger.error("sglang을 설치하고 서버를 실행해주세요.")
        return ["sglang을 설치하고 서버를 실행해주세요."]
    except ServerNotRunning:
        logger.error("sglang을 설치하고 서버를 실행해주세요.")
        return ["sglang을 설치하고 서버를 실행해주세요."]


def get_openai_image_models(api_key: Optional[str] = None):
    import openai
    from openai import OpenAI

    model_list = []

    if not api_key:
        model_list.append("OpenAI API Key가 필요합니다.")
        return model_list

    client = OpenAI(api_key=api_key)

    try:
        model = client.models.list()

        for m in model.data:
            model_id = m.id

            if "image" in model_id.lower():
                model_list.append(model_id)

        logger.info(f"openai 이미지 모델 목록: {model_list}")

        return model_list

    except openai.AuthenticationError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        return model_list


# def get_openai_video_models(api_key: Optional[str] = None):
#     import openai
#     from openai import OpenAI

#     model_list = []

#     if not api_key:
#         model_list.append("OpenAI API Key가 필요합니다.")
#         return model_list

#     client = OpenAI(api_key=api_key)

#     try:
#         model = client.models.list()

#         for m in model.data:
#             model_id = m.id

#             if "sora" in model_id.lower():
#                 model_list.append(model_id)

#         return model_list

#     except openai.AuthenticationError as e:
#         model_list.append(f"OpenAI API 오류 발생: {e}")
#         return model_list


def get_google_genai_image_models(api_key: Optional[str] = None):
    from google import genai
    from google.genai import errors

    model_list = []

    if not api_key:
        model_list.append("Google AI API Key가 필요합니다.")
        return model_list

    client = genai.Client(api_key=api_key)

    try:
        model = client.models.list()

        for m in model.page:
            include = any(k in m.name.lower() for k in ["imagen", "image"])

            if include:
                model_list.append(m.name)

        logger.info(f"google genai 이미지 모델 목록: {model_list}")

        return model_list

    except errors.ClientError as e:
        model_list.append(f"Google AI API 오류 발생: {e}")
        logger.error(f"Google AI API 오류 발생 (클라이언트 오류): {e}")
        return model_list
    except errors.ServerError as e:
        model_list.append(f"Google AI API 오류 발생: {e}")
        logger.error(f"Google AI API 오류 발생 (서버 오류): {e}")
        return model_list
    except errors.APIError as e:
        model_list.append(f"Google AI API 오류 발생: {e}")
        logger.error(f"Google AI API 오류 발생 (API 오류): {e}")
        return model_list
    except Exception as e:
        model_list.append(f"Google AI API 오류 발생: {e}")
        logger.exception(f"Google AI API 오류 발생 (예기치 못한 오류): {e}")
        return model_list


def get_google_genai_video_models(api_key: Optional[str] = None):
    from google import genai
    from google.genai import errors

    model_list = []

    if not api_key:
        model_list.append("Google AI API Key가 필요합니다.")
        return model_list

    client = genai.Client(api_key=api_key)

    try:
        model = client.models.list()

        for m in model.page:
            include = any(k in m.name.lower() for k in ["veo"])

            if include:
                model_list.append(m.name)

        logger.info(f"google genai 비디오 모델 목록: {model_list}")

        return model_list

    except errors.ClientError as e:
        model_list.append(f"Google AI API 오류 발생: {e}")
        logger.error(f"Google AI API 오류 발생 (클라이언트 오류): {e}")
        return model_list
    except errors.ServerError as e:
        model_list.append(f"Google AI API 오류 발생: {e}")
        logger.error(f"Google AI API 오류 발생 (서버 오류): {e}")
        return model_list
    except errors.APIError as e:
        model_list.append(f"Google AI API 오류 발생: {e}")
        logger.error(f"Google AI API 오류 발생 (API 오류): {e}")
        return model_list
    except Exception as e:
        model_list.append(f"Google AI API 오류 발생: {e}")
        logger.exception(f"Google AI API 오류 발생 (예기치 못한 오류): {e}")
        return model_list


image_api_models = []
video_api_models = []

comfyui_models = []
comfyui_loras = []
comfyui_vae = []
comfyui_controlnet = []
comfyui_clip = []
comfyui_clip_vision = []
comfyui_text_encoders = []
comfyui_embeddings = []
comfyui_diffusion_models = []
comfyui_pretrained_models = []
comfyui_inpaint_models = []
comfyui_ipadapter = []
comfyui_unet = []

openai_image_api_models = []
# openai_video_api_models = []

google_genai_image_api_models = []
google_genai_video_api_models = []

huggingface_inference_image_api_models = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-3.5-large",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large-turbo",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
    "Qwen/Qwen-Image",
    "Qwen/Qwen-Image-2512",
    "Tongyi-MAI/Z-Image-Turbo",
    "Tongyi-MAI/Z-Image",
    "Alpha-VLLM/Lumina-Image-2.0",
    "fal/AuraFlow-v0.3",
    "zai-org/CogView4-6B",
    "zai-org/GLM-Image",
    "HiDream-ai/HiDream-I1-Fast",
    "HiDream-ai/HiDream-I1-Dev",
    "HiDream-ai/HiDream-I1-Full",
]

huggingface_inference_image_edit_api_models = [
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-dev",
    "Qwen/Qwen-Image-Edit",
    "Qwen/Qwen-Image-Edit-2509",
    "Qwen/Qwen-Image-Edit-2511",
]

huggingface_inference_video_api_models = [
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "zai-org/CogVideoX-5b",
]

huggingface_inference_image_to_video_api_models = [
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "Lightricks/LTX-2",
]

# --- Lazy Loading ---

_initialized_image_providers: set = set()

# ComfyUI 관련 폴더 목록
_COMFYUI_FOLDERS = {
    "comfyui_models": "checkpoints",
    "comfyui_loras": "loras",
    "comfyui_vae": "vae",
    "comfyui_controlnet": "controlnet",
    "comfyui_clip": "clip_gguf",
    "comfyui_clip_vision": "clip_vision",
    "comfyui_text_encoders": "text_encoders",
    "comfyui_embeddings": "embeddings",
    "comfyui_diffusion_models": "diffusion_models",
    "comfyui_pretrained_models": "diffusers",
    "comfyui_inpaint_models": "inpaint",
    "comfyui_ipadapter": "ipadapter",
    "comfyui_unet": "unet_gguf",
}

# 정적 리스트 provider (네트워크 호출 불필요)
_STATIC_IMAGE_PROVIDERS = {"hf-inference", "self-provided"}


def initialize_image_provider(provider: str) -> None:
    """
    특정 provider의 이미지 모델 목록을 로딩합니다.
    이미 초기화된 provider는 스킵합니다.

    Args:
        provider: 초기화할 provider 이름
    """
    global comfyui_models, comfyui_loras, comfyui_vae, comfyui_controlnet
    global comfyui_clip, comfyui_clip_vision, comfyui_text_encoders
    global comfyui_embeddings, comfyui_diffusion_models, comfyui_pretrained_models
    global comfyui_inpaint_models, comfyui_ipadapter, comfyui_unet
    global openai_image_api_models, google_genai_image_api_models
    global google_genai_video_api_models

    if provider in _initialized_image_providers:
        return

    if provider in _STATIC_IMAGE_PROVIDERS:
        _initialized_image_providers.add(provider)
        return

    if provider == "comfyui":
        for var_name, folder in _COMFYUI_FOLDERS.items():
            result = get_comfyui_image_models(folder=folder)
            globals()[var_name] = result
        logger.info(f"Image provider 'comfyui' 모델 목록 로딩 완료")
    elif provider == "openai":
        openai_image_api_models = get_openai_image_models(load_env_variables("OPENAI_API_KEY"))
        globals()["openai_image_api_models"] = openai_image_api_models
        logger.info(f"Image provider 'openai' 모델 목록 로딩 완료: {len(openai_image_api_models)}개")
    elif provider == "google-genai":
        google_genai_image_api_models = get_google_genai_image_models(load_env_variables("GEMINI_API_KEY"))
        google_genai_video_api_models = get_google_genai_video_models(load_env_variables("GEMINI_API_KEY"))
        globals()["google_genai_image_api_models"] = google_genai_image_api_models
        globals()["google_genai_video_api_models"] = google_genai_video_api_models
        logger.info(f"Image provider 'google-genai' 모델 목록 로딩 완료")
    else:
        logger.warning(f"알 수 없는 Image provider: {provider}")
        return

    _initialized_image_providers.add(provider)


def refresh_image_provider(provider: str) -> None:
    """
    이미 초기화된 provider의 이미지 모델 목록을 강제로 갱신합니다.

    Args:
        provider: 갱신할 provider 이름
    """
    _initialized_image_providers.discard(provider)
    initialize_image_provider(provider)


def is_image_provider_initialized(provider: str) -> bool:
    """provider가 이미 초기화되었는지 확인합니다."""
    return provider in _initialized_image_providers


def rebuild_image_api_models() -> None:
    """초기화된 모든 provider의 모델을 image_api_models에 집계합니다."""
    global image_api_models, video_api_models
    image_api_models = []
    video_api_models = []
    image_api_models.extend(globals().get("openai_image_api_models", []))
    image_api_models.extend(globals().get("google_genai_image_api_models", []))
    image_api_models.extend(globals().get("comfyui_models", []))
    # video_api_models.extend(globals().get("openai_video_api_models", []))
    video_api_models.extend(globals().get("google_genai_video_api_models", []))
