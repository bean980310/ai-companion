from typing import Optional

from ai_companion_core import logger
from ai_companion_core.environ_manager import load_env_variables


class LocalModelNotFound(Exception):
    pass


class ServerNotRunning(Exception):
    pass


def get_lmstudio_models(api_host: str = "localhost:1234", api_key: str = "lmstudio"):
    try:
        import lmstudio as lms
    except ImportError:
        logger.error("lmstudio가 설치되지 않았습니다.")
        return ["lmstudio가 설치되지 않았습니다."]

    llm = []
    client = lms.Client(api_host=api_host)
    try:
        downloaded_llm = client.list_downloaded_models("llm")

        if len(downloaded_llm) == 0:
            raise LocalModelNotFound("모델이 존재하지 않습니다.")

        for m in downloaded_llm:
            llm.append(m.model_key)

        logger.info(f"lmstudio 모델 목록: {llm}")

        return llm
    except LocalModelNotFound:
        logger.error("모델이 존재하지 않습니다.")
        return ["모델이 존재하지 않습니다."]
    except ServerNotRunning:
        logger.error("LM Studio를 설치하고 서버를 실행해주세요.")
        return ["LM Studio를 설치하고 서버를 실행해주세요."]


def get_lmstudio_embedding_models(api_host: str = "localhost:1234", api_key: str = "lmstudio"):
    try:
        import lmstudio as lms
    except ImportError:
        logger.error("lmstudio가 설치되지 않았습니다.")
        return ["lmstudio가 설치되지 않았습니다."]

    embedding = []
    client = lms.Client(api_host=api_host)
    try:
        downloaded_embedding = client.list_downloaded_models("embedding")

        if len(downloaded_embedding) == 0:
            raise LocalModelNotFound("모델이 존재하지 않습니다.")

        for m in downloaded_embedding:
            embedding.append(m.model_key)

        logger.info(f"lmstudio embedding 모델 목록: {embedding}")

        return embedding
    except LocalModelNotFound:
        logger.error("모델이 존재하지 않습니다.")
        return ["모델이 존재하지 않습니다."]
    except ServerNotRunning:
        logger.error("LM Studio를 설치하고 서버를 실행해주세요.")
        return ["LM Studio를 설치하고 서버를 실행해주세요."]


def get_ollama_models(host: str = "http://localhost:11434", api_key: str = "ollama"):
    try:
        import ollama
    except ImportError:
        logger.error("ollama가 설치되지 않았습니다.")
        return ["ollama가 설치되지 않았습니다."]

    llm = []
    try:
        client = ollama.Client(host=host)
        models = client.list().models

        if len(models) == 0:
            raise LocalModelNotFound("모델이 존재하지 않습니다.")

        for m in models:
            llm.append(m.model)

        logger.info(f"ollama 모델 목록: {llm}")

        return llm
    except LocalModelNotFound:
        logger.error("모델이 존재하지 않습니다.")
        return ["모델이 존재하지 않습니다."]
    except ServerNotRunning:
        logger.error("Ollama를 설치하고 서버를 실행해주세요.")
        return ["Ollama를 설치하고 서버를 실행해주세요."]


def get_omlx_models(host: str = "http://localhost:8001/v1", api_key: str = "omlx"):
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        logger.error("openai가 설치되지 않았습니다.")
        return ["openai가 설치되지 않았습니다."]

    llm = []
    client = OpenAI(api_key=api_key, base_url=host)

    try:
        model = client.models.list()

        if len(model.data) == 0:
            raise LocalModelNotFound("모델이 존재하지 않습니다.")

        for m in model.data:
            llm.append(m.id)

        logger.info(f"omlx 모델 목록: {llm}")

        return llm
    except LocalModelNotFound:
        logger.error("모델이 존재하지 않습니다.")
        return ["모델이 존재하지 않습니다."]
    except openai.PermissionDeniedError:
        logger.error("omlx를 설치하고 서버를 실행해주세요.")
        return ["omlx를 설치하고 서버를 실행해주세요."]
    except openai.APIConnectionError:
        logger.error("omlx를 설치하고 서버를 실행해주세요.")
        return ["omlx를 설치하고 서버를 실행해주세요."]
    except ServerNotRunning:
        logger.error("omlx를 설치하고 서버를 실행해주세요.")
        return ["omlx를 설치하고 서버를 실행해주세요."]


def get_oobabooga_models(host: str = "http://localhost:5000/v1", api_key: str = "oobabooga"):
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        logger.error("openai가 설치되지 않았습니다.")
        return ["openai가 설치되지 않았습니다."]

    llm = []
    client = OpenAI(api_key="not-needed", base_url=host)

    try:
        model = client.models.list()

        if len(model.data) == 0:
            raise LocalModelNotFound("모델이 존재하지 않습니다.")

        for m in model.data:
            llm.append(m.id)

        logger.info(f"oobabooga 모델 목록: {llm}")

        return llm
    except LocalModelNotFound:
        logger.error("모델이 존재하지 않습니다.")
        return ["모델이 존재하지 않습니다."]
    except openai.PermissionDeniedError:
        logger.error("Oobabooga를 설치하고 서버를 실행해주세요.")
        return ["Oobabooga를 설치하고 서버를 실행해주세요."]
    except openai.APIConnectionError:
        logger.error("Oobabooga를 설치하고 서버를 실행해주세요.")
        return ["Oobabooga를 설치하고 서버를 실행해주세요."]
    except ServerNotRunning:
        logger.error("Oobabooga를 설치하고 서버를 실행해주세요.")
        return ["Oobabooga를 설치하고 서버를 실행해주세요."]


def get_vllm_models(host: str = "http://localhost:8000/v1", api_key: str = "vllm"):
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        logger.error("openai가 설치되지 않았습니다.")
        return ["openai가 설치되지 않았습니다."]

    llm = []
    client = OpenAI(api_key="not-needed", base_url=host)

    try:
        model = client.models.list()

        if len(model.data) == 0:
            raise LocalModelNotFound("모델이 존재하지 않습니다.")

        for m in model.data:
            llm.append(m.id)

        logger.info(f"vllm 모델 목록: {llm}")

        return llm
    except LocalModelNotFound:
        logger.error("모델이 존재하지 않습니다.")
        return ["모델이 존재하지 않습니다."]
    except openai.PermissionDeniedError:
        logger.error("vllm을 설치하고 서버를 실행해주세요.")
        return ["vllm을 설치하고 서버를 실행해주세요."]
    except openai.APIConnectionError:
        logger.error("vllm을 설치하고 서버를 실행해주세요.")
        return ["vllm을 설치하고 서버를 실행해주세요."]
    except ServerNotRunning:
        logger.error("vllm을 설치하고 서버를 실행해주세요.")
        return ["vllm을 설치하고 서버를 실행해주세요."]


def get_sglang_llm_models(host: str = "http://localhost:30001/v1", api_key: str = "sglang"):
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        logger.error("openai가 설치되지 않았습니다.")
        return ["openai가 설치되지 않았습니다."]

    llm = []
    client = OpenAI(api_key="not-needed", base_url=host)

    try:
        model = client.models.list()

        if len(model.data) == 0:
            raise LocalModelNotFound("모델이 존재하지 않습니다.")

        for m in model.data:
            llm.append(m.id)

        logger.info(f"sglang 모델 목록: {llm}")

        return llm
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


def get_openai_llm_models(api_key: Optional[str] = None):
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        logger.error("openai가 설치되지 않았습니다.")
        return ["openai가 설치되지 않았습니다."]
    except Exception as e:
        logger.exception(f"OpenAI API 오류 발생 (예기치 못한 오류): {e}")
        return [f"OpenAI API 오류 발생: {e}"]

    model_list = []

    if not api_key:
        model_list.append("OpenAI API Key가 필요합니다.")
        return model_list

    client = OpenAI(api_key=api_key)

    try:
        model = client.models.list()

        gpt_pattern = ["gpt-4o", "gpt-4.1", "gpt-5", "gpt-oss"]

        for m in model.data:
            model_id = m.id

            include = any(k in model_id.lower() for k in gpt_pattern)
            exclude_type = all(
                k not in model_id.lower()
                for k in [
                    "image",
                    "realtime",
                    "tts",
                    "audio",
                    "transcribe",
                    "codex",
                    "search",
                    "preview",
                ]
            )
            exclude_model = all(
                k not in model_id.lower()
                for k in [
                    "gpt-4.1-mini",
                    "gpt-4.1-nano",
                    "gpt-4o-mini",
                    "chatgpt-4o-latest",
                ]
            )
            if include and exclude_type and exclude_model:
                model_list.append(model_id)

        logger.info(f"openai 모델 목록: {model_list}")

        return model_list

    except openai.BadRequestError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        logger.error(f"OpenAI API 오류 발생 (잘못된 요청): {e}")
        return model_list
    except openai.AuthenticationError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        logger.error(f"OpenAI API 오류 발생 (인증 오류): {e}")
        return model_list
    except openai.PermissionDeniedError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        logger.error(f"OpenAI API 오류 발생 (권한 오류): {e}")
        return model_list
    except openai.APITimeoutError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        logger.error(f"OpenAI API 오류 발생 (시간 초과): {e}")
        return model_list
    except openai.APIConnectionError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        logger.error(f"OpenAI API 오류 발생 (연결 오류): {e}")
        return model_list
    except openai.RateLimitError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        logger.error(f"OpenAI API 오류 발생 (한도 초과): {e}")
        return model_list
    except openai.InternalServerError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        logger.error(f"OpenAI API 오류 발생 (서버 오류): {e}")
        return model_list
    except openai.APIError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        logger.error(f"OpenAI API 오류 발생 (API 오류): {e}")
        return model_list
    except Exception as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        logger.exception(f"OpenAI API 오류 발생 (예기치 못한 오류): {e}")
        return model_list


def get_anthropic_llm_models(api_key: Optional[str] = None):
    try:
        import anthropic
        from anthropic import Anthropic
    except ImportError:
        logger.error("anthropic가 설치되지 않았습니다.")
        return ["anthropic가 설치되지 않았습니다."]
    except Exception as e:
        logger.exception(f"Anthropic API 오류 발생 (예기치 못한 오류): {e}")
        return [f"Anthropic API 오류 발생: {e}"]

    model_list = []

    if not api_key:
        model_list.append("Anthropic API Key가 필요합니다.")
        return model_list

    client = Anthropic(api_key=api_key)

    try:
        model = client.beta.models.list()

        for m in model.data:
            model_id = m.id
            model_list.append(model_id)

        logger.info(f"anthropic 모델 목록: {model_list}")

        return model_list

    except anthropic.BadRequestError as e:
        model_list.append(f"Anthropic API 오류 발생: {e}")
        logger.error(f"Anthropic API 오류 발생 (잘못된 요청): {e}")
        return model_list
    except anthropic.AuthenticationError as e:
        model_list.append(f"Anthropic API 오류 발생: {e}")
        logger.error(f"Anthropic API 오류 발생 (인증 오류): {e}")
        return model_list
    except anthropic.PermissionDeniedError as e:
        model_list.append(f"Anthropic API 오류 발생: {e}")
        logger.error(f"Anthropic API 오류 발생 (권한 오류): {e}")
        return model_list
    except anthropic.APITimeoutError as e:
        model_list.append(f"Anthropic API 오류 발생: {e}")
        logger.error(f"Anthropic API 오류 발생 (시간 초과): {e}")
        return model_list
    except anthropic.APIConnectionError as e:
        model_list.append(f"Anthropic API 오류 발생: {e}")
        logger.error(f"Anthropic API 오류 발생 (연결 오류): {e}")
        return model_list
    except anthropic.InternalServerError as e:
        model_list.append(f"Anthropic API 오류 발생: {e}")
        logger.error(f"Anthropic API 오류 발생 (서버 오류): {e}")
        return model_list
    except anthropic.RateLimitError as e:
        model_list.append(f"Anthropic API 오류 발생: {e}")
        logger.error(f"Anthropic API 오류 발생 (한도 초과): {e}")
        return model_list
    except anthropic.APIError as e:
        model_list.append(f"Anthropic API 오류 발생: {e}")
        logger.error(f"Anthropic API 오류 발생 (API 오류): {e}")
        return model_list
    except Exception as e:
        model_list.append(f"Anthropic API 오류 발생: {e}")
        logger.exception(f"Anthropic API 오류 발생 (예기치 못한 오류): {e}")
        return model_list


def get_google_genai_llm_models(api_key: Optional[str] = None):
    try:
        from google import genai
        from google.genai import errors
        from google.api_core import exceptions
    except ImportError:
        logger.error("google-genai가 설치되지 않았습니다.")
        return ["google-genai가 설치되지 않았습니다."]
    except Exception as e:
        logger.exception(f"Google AI API 오류 발생 (예기치 못한 오류): {e}")
        return [f"Google AI API 오류 발생: {e}"]

    model_list = []

    if not api_key:
        model_list.append("Google AI API Key가 필요합니다.")
        return model_list

    client = genai.Client(api_key=api_key)

    try:
        model = client.models.list()

        for m in model.page:
            include = any(k in m.name.lower() for k in ["gemini", "gemma"])
            exclude_type = all(k not in m.name.lower() for k in ["embedding", "tts", "exp"])
            exclude_model = all(k not in m.name.lower() for k in ["gemini-2.0"])

            if "generateContent" in m.supported_actions and include and exclude_type and exclude_model:
                model_list.append(m.name)

        logger.info(f"google genai 모델 목록: {model_list}")

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


def get_perplexity_llm_models(api_key: Optional[str] = None):
    try:
        import perplexity
        from perplexity import Perplexity
    except ImportError:
        logger.error("perplexity가 설치되지 않았습니다.")
        return ["perplexity가 설치되지 않았습니다."]
    except Exception as e:
        logger.exception(f"Perplexity API 오류 발생 (예기치 못한 오류): {e}")
        return [f"Perplexity API 오류 발생: {e}"]

    model_list = []

    # api_models = [
    #     "sonar",
    #     "sonar-pro",
    #     # "sonar-reasoning",
    #     "sonar-reasoning-pro",
    #     "sonar-deep-research",
    # ]

    if not api_key:
        model_list.append("Perplexity API Key가 필요합니다.")
        return model_list

    # unused
    # client = Perplexity(api_key=api_key)

    try:
        import requests

        url = "https://api.perplexity.ai/v1/models"
        model = requests.get(url).json()

        for m in model["data"]:
            model_list.append(m["id"])

        logger.info(f"perplexity 모델 목록: {model_list}")

        return model_list

    except perplexity.AuthenticationError as e:
        model_list.append(f"Perplexity API 오류 발생: {e}")
        logger.error(f"Perplexity API 오류 발생 (인증 오류): {e}")
        return model_list
    except perplexity.APIError as e:
        model_list.append(f"Perplexity API 오류 발생: {e}")
        logger.error(f"Perplexity API 오류 발생 (API 오류): {e}")
        return model_list
    except Exception as e:
        model_list.append(f"Perplexity API 오류 발생: {e}")
        logger.exception(f"Perplexity API 오류 발생 (예기치 못한 오류): {e}")
        return model_list


def get_xai_llm_models(api_key: Optional[str] = None):
    try:
        import xai_sdk
    except ImportError:
        logger.error("xai_sdk가 설치되지 않았습니다.")
        return ["xai_sdk가 설치되지 않았습니다."]
    except Exception as e:
        logger.exception(f"XAI API 오류 발생 (예기치 못한 오류): {e}")
        return [f"XAI API 오류 발생: {e}"]

    model_list = []

    if not api_key:
        model_list.append("XAI API Key가 필요합니다.")
        return model_list

    client = xai_sdk.Client(api_key=api_key)

    try:
        model = client.models.list_language_models()
        for m in model:
            model_list.append(m.name)

        logger.info(f"XAI 모델 목록: {model_list}")

        return model_list

    except Exception as e:
        model_list.append(f"XAI API 오류 발생: {e}")
        logger.error(f"XAI API 오류 발생: {e}")
        return model_list


def get_mistralai_llm_models(api_key: Optional[str] = None):
    try:
        import mistralai
        from mistralai.client import Mistral
    except ImportError:
        logger.error("mistralai가 설치되지 않았습니다.")
        return ["mistralai가 설치되지 않았습니다."]
    except Exception as e:
        logger.exception(f"Mistral AI API 오류 발생 (예기치 못한 오류): {e}")
        return [f"Mistral AI API 오류 발생: {e}"]

    model_list = []

    LLM_ALIASES = [
        "mistral-large-pixtral-2411",
        "mistral-medium",
        "mistral-tiny",
        "mistral-tiny-2312",
        "mistral-tiny-2407",
        "open-mistral-7b",
        "open-mistral-nemo",
        "voxtral-mini-transcribe",
        "latest",
    ]

    if not api_key:
        model_list.append("Mistral AI API Key가 필요합니다.")
        return model_list

    client = Mistral(api_key=api_key)

    try:
        model = client.models.list()
        for m in model.data:
            if m.capabilities.completion_chat and not m.deprecation and all(x not in m.id for x in LLM_ALIASES):
                model_list.append(m.id)

        logger.info(f"Mistral AI 모델 목록: {model_list}")

        return model_list

    except Exception as e:
        model_list.append(f"Mistral AI API 오류 발생: {e}")
        logger.error(f"Mistral AI API 오류 발생: {e}")
        return model_list


# def get_huggingface_hub_models(api_key: Optional[str] = None):
#     try:
#         import huggingface_hub
#         from huggingface_hub import HfApi
#     except ImportError:
#         logger.error("huggingface_hub가 설치되지 않았습니다.")
#         return ["huggingface_hub가 설치되지 않았습니다."]
#     except Exception as e:
#         logger.exception(f"HuggingFace API 오류 발생 (예기치 못한 오류): {e}")
#         return [f"HuggingFace API 오류 발생: {e}"]

#     tags = ["image-text-to-text", "text-generation"]

#     model_list = []

#     if not api_key:
#         model_list.append("HuggingFace API Key가 필요합니다.")
#         return model_list

#     client = HfApi(token=api_key, library_name="transformers")

#     try:
#         for t in tags:
#             models = client.list_models(filter=[t, "transformers"], inference="warm", sort="trending_score", expand=["inference"])
#             for m in models:
#                 model_list.append(m.id)

#         logger.info(f"HuggingFace 모델 목록: {model_list}")

#         return model_list

#     except Exception as e:
#         model_list.append(f"HuggingFace API 오류 발생: {e}")
#         logger.error(f"HuggingFace API 오류 발생: {e}")
#         return model_list


llm_api_models = []
lmstudio_models = []
ollama_models = []
oobabooga_models = []
omlx_models = []
vllm_api_models = []
sglang_llm_models = []
openai_api_models = []
anthropic_api_models = []
google_genai_api_models = []
perplexity_api_models = []
xai_api_models = []
mistralai_api_models = []
# huggingface_hub_models = []

openrouter_api_models = [
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-4-maverick",
    "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen/qwen3-vl-30b-a3b-thinking",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "qwen/qwen3-max",
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3.5-plus-02-15",
    "qwen/qwen3.5-122b-a10b",
    "qwen/qwen3.5-35b-a3b",
    "mistralai/mistral-small-3.2-24b-instruct",
    "mistralai/mistral-medium-3.1",
    "mistralai/mistral-large-2512",
    "moonshotai/kimi-k2",
    "moonshotai/kimi-k2-0905",
    "moonshotai/kimi-k2.5",
    "z-ai/glm-4.6v",
    "z-ai/glm-4.7",
    "z-ai/glm-5",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-chat-v3.1",
    "minimax/minimax-01",
    "minimax/minimax-m1",
]

huggingface_inference_api_models = [
    "meta-llama/Llama-4-Scout-17B-16E-Instruct:fastest",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct:cheapest",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:fastest",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:cheapest",
    "Qwen/Qwen3.5-397B-A17B:fastest",
    "Qwen/Qwen3.5-397B-A17B:cheapest",
    "Qwen/Qwen3.5-122B-A10B:fastest",
    "Qwen/Qwen3.5-122B-A10B:cheapest",
    "Qwen/Qwen3.6-35B-A3B:fastest",
    "Qwen/Qwen3.6-35B-A3B:cheapest",
    "Qwen/Qwen3.6-27B:fastest",
    "Qwen/Qwen3.6-27B:cheapest",
    "moonshotai/Kimi-K2.5:fastest",
    "moonshotai/Kimi-K2.5:cheapest",
    "moonshotai/Kimi-K2.6:fastest",
    "moonshotai/Kimi-K2.6:cheapest",
    "moonshotai/Kimi-K2.7-Code:fastest",
    "moonshotai/Kimi-K2.7-Code:cheapest",
    "google/gemma-4-31B-it:fastest",
    "google/gemma-4-31B-it:cheapest",
    "google/gemma-4-26B-A4B-it:fastest",
    "google/gemma-4-26B-A4B-it:cheapest",
    "zai-org/GLM-4.6V:fastest",
    "zai-org/GLM-4.6V:cheapest",
    "zai-org/GLM-4.6V:zai-org",
    "zai-org/GLM-4.6V-Flash:fastest",
    "zai-org/GLM-4.6V-Flash:cheapest",
    "zai-org/GLM-4.6V-Flash:zai-org",
    "zai-org/GLM-4.7:fastest",
    "zai-org/GLM-4.7:cheapest",
    "zai-org/GLM-4.7:zai-org",
    "zai-org/GLM-4.7-Flash:fastest",
    "zai-org/GLM-4.7-Flash:cheapest",
    "zai-org/GLM-4.7-Flash:zai-org",
    "zai-org/GLM-5:fastest",
    "zai-org/GLM-5:cheapest",
    "zai-org/GLM-5:zai-org",
    "zai-org/GLM-5.1:fastest",
    "zai-org/GLM-5.1:cheapest",
    "zai-org/GLM-5.1:zai-org",
    "zai-org/GLM-5.2:fastest",
    "zai-org/GLM-5.2:cheapest",
    "zai-org/GLM-5.2:zai-org",
    "deepseek-ai/DeepSeek-V4-Pro:fastest",
    "deepseek-ai/DeepSeek-V4-Pro:cheapest",
    "deepseek-ai/DeepSeek-V4-Flash:fastest",
    "deepseek-ai/DeepSeek-V4-Flash:cheapest",
    "MiniMaxAI/MiniMax-M2.5:fastest",
    "MiniMaxAI/MiniMax-M2.5:cheapest",
    "MiniMaxAI/MiniMax-M2.7:fastest",
    "MiniMaxAI/MiniMax-M2.7:cheapest",
    "MiniMaxAI/MiniMax-M3:fastest",
    "MiniMaxAI/MiniMax-M3:cheapest",
    "CohereLabs/command-a-plus-05-2026-bf16:cohere",
    "CohereLabs/command-a-plus-05-2026-fp8:cohere",
    "CohereLabs/command-a-plus-05-2026-w4a4:cohere",
    "openai/gpt-oss-20b:fastest",
    "openai/gpt-oss-20b:cheapest",
    "openai/gpt-oss-120b:fastest",
    "openai/gpt-oss-120b:cheapest",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16:cheapest",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16:fastest",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:cheapest",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:fastest",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:cheapest",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:fastest",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4:cheapest",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4:fastest",
]

# --- Lazy Loading ---

_initialized_llm_providers: set = set()

# Provider name → (global variable name, loader function, env key)
_LLM_PROVIDER_LOADERS = {
    "lmstudio": ("lmstudio_models", get_lmstudio_models, "LM_API_KEY"),
    "ollama": ("ollama_models", get_ollama_models, "OLLAMA_API_KEY"),
    "oobabooga": ("oobabooga_models", get_oobabooga_models, "OOGA_API_KEY"),
    "omlx": ("omlx_models", get_omlx_models, "OMLX_API_KEY"),
    "vllm-api": ("vllm_api_models", get_vllm_models, "VLLM_API_KEY"),
    "sglang": ("sglang_llm_models", get_sglang_llm_models, "SGLANG_API_KEY"),
    "openai": ("openai_api_models", get_openai_llm_models, "OPENAI_API_KEY"),
    "anthropic": ("anthropic_api_models", get_anthropic_llm_models, "ANTHROPIC_API_KEY"),
    "google-genai": ("google_genai_api_models", get_google_genai_llm_models, "GEMINI_API_KEY"),
    "perplexity": ("perplexity_api_models", get_perplexity_llm_models, "PERPLEXITY_API_KEY"),
    "xai": ("xai_api_models", get_xai_llm_models, "XAI_API_KEY"),
    "mistralai": ("mistralai_api_models", get_mistralai_llm_models, "MISTRAL_API_KEY"),
}

# 정적 리스트 provider (네트워크 호출 불필요)
_STATIC_LLM_PROVIDERS = {"openrouter", "hf-inference", "self-provided"}


def initialize_llm_provider(provider: str) -> None:
    """
    특정 provider의 LLM 모델 목록을 로딩합니다.
    이미 초기화된 provider는 스킵합니다.

    Args:
        provider: 초기화할 provider 이름
    """
    global lmstudio_models, ollama_models, oobabooga_models, omlx_models
    global vllm_api_models, sglang_llm_models
    global openai_api_models, anthropic_api_models, google_genai_api_models
    global perplexity_api_models, xai_api_models, mistralai_api_models

    if provider in _initialized_llm_providers:
        return

    if provider in _STATIC_LLM_PROVIDERS:
        _initialized_llm_providers.add(provider)
        return

    if provider in _LLM_PROVIDER_LOADERS:
        var_name, loader_fn, env_key = _LLM_PROVIDER_LOADERS[provider]
        api_key = load_env_variables(env_key)
        result = loader_fn(api_key=api_key)

        # Update the module-level variable
        globals()[var_name] = result

        logger.info(f"LLM provider '{provider}' 모델 목록 로딩 완료: {len(result)}개")
        _initialized_llm_providers.add(provider)
    else:
        logger.warning(f"알 수 없는 LLM provider: {provider}")


def refresh_llm_provider(provider: str) -> None:
    """
    이미 초기화된 provider의 모델 목록을 강제로 갱신합니다.

    Args:
        provider: 갱신할 provider 이름
    """
    _initialized_llm_providers.discard(provider)
    initialize_llm_provider(provider)


def is_llm_provider_initialized(provider: str) -> bool:
    """provider가 이미 초기화되었는지 확인합니다."""
    return provider in _initialized_llm_providers


def rebuild_llm_api_models() -> None:
    """초기화된 모든 provider의 모델을 llm_api_models에 집계합니다."""
    global llm_api_models
    llm_api_models = []
    llm_api_models.extend(globals().get("lmstudio_models", []))
    llm_api_models.extend(globals().get("ollama_models", []))
    llm_api_models.extend(globals().get("openai_api_models", []))
    llm_api_models.extend(globals().get("anthropic_api_models", []))
    llm_api_models.extend(globals().get("google_genai_api_models", []))
    llm_api_models.extend(globals().get("perplexity_api_models", []))
    llm_api_models.extend(globals().get("xai_api_models", []))
    llm_api_models.extend(globals().get("mistralai_api_models", []))
    llm_api_models.extend(openrouter_api_models)

