import os
from typing import List
import re
from ..common.environ_manager import load_env_variables

def get_lmstudio_models(api_host: str = "localhost:1234"):
    import lmstudio as lms

    llm = []
    client = lms.Client(api_host=api_host)
    try:
        downloaded_llm = client.list_downloaded_models("llm")

        if len(downloaded_llm) == 0:
            return ["모델이 존재하지 않습니다."]
        
        for m in downloaded_llm:
            llm.append(m.model_key)

        return llm
    except:
        return ["LM Studio를 설치하고 서버를 실행해주세요."]

def get_lmstudio_embedding_models(api_host: str = "localhost:1234"):
    import lmstudio as lms

    embedding = []
    client = lms.Client(api_host=api_host)
    try:
        downloaded_embedding = client.list_downloaded_models("embedding")

        if len(downloaded_embedding) == 0:
            return ["모델이 존재하지 않습니다."]
        
        for m in downloaded_embedding:
            embedding.append(m.model_key)

        return embedding
    except:
        return ["LM Studio를 설치하고 서버를 실행해주세요."]

def get_ollama_models(host: str = "http://localhost:11434"):
    import ollama

    llm = []
    try:
        client = ollama.Client(host=host)
        models = client.list().models

        if len(models) == 0:
            return ["모델이 존재하지 않습니다."]

        for m in models:
            llm.append(m.model)

        return llm
    except:
        return ["Ollama를 설치하고 서버를 실행해주세요."]

def get_oobabooga_models(host: str = "http://localhost:5000/v1"):
    import openai
    from openai import OpenAI

    llm = []
    client = OpenAI(api_key="not-needed", base_url=host)

    try:
        model = client.models.list()

        if len(model.data) == 0:
            return ["모델이 존재하지 않습니다."]

        for m in model.data:
            llm.append(m.id)

        return llm
    except:
        return ["Oobabooga를 설치하고 서버를 실행해주세요."]

def get_openai_llm_models(api_key: str = None):
    import openai
    from openai import OpenAI

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

            date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

            include = any(k in model_id.lower() for k in gpt_pattern)
            exclude_type = all(k not in model_id.lower() for k in ["image", "realtime", "tts", "audio", "transcribe", "codex", "search", "preview"])
            exclude_mini = all(k not in model_id.lower() for k in ["gpt-4.1-mini", "gpt-4.1-nano", 'gpt-4o-mini'])
            latest_or_date = ("latest" in model_id) or bool(date_pattern.search(model_id.lower()))
            if include and exclude_type and exclude_mini and latest_or_date:
                model_list.append(model_id)

        return model_list
        
    except openai.AuthenticationError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        return model_list

def get_anthropic_llm_models(api_key: str = None):
    import anthropic
    from anthropic import Anthropic

    model_list = []

    if not api_key:
        model_list.append("Anthropic API Key가 필요합니다.")
        return model_list

    client = Anthropic(api_key=api_key)

    try:
        model = client.models.list()

        for m in model.data:
            model_id = m.id
            exclude = "claude-3" not in model_id.lower()

            if exclude:
                model_list.append(model_id)
        
        return model_list

    except anthropic.AuthenticationError as e:
        model_list.append(f"Anthropic API 오류 발생: {e}")
        return model_list

def get_google_genai_llm_models(api_key: str = None):
    from google import genai
    from google.api_core import exceptions

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

            preview_check = ("preview" not in m.name.lower()) or any(k in m.name.lower() for k in ["gemini-3-pro-preview", "gemini-3-pro-image-preview"])

            if "generateContent" in m.supported_actions and include and exclude_type and exclude_model and preview_check:
                model_list.append(m.name)

        return model_list

    except exceptions.Unauthenticated as e:
        model_list.append(f"Google AI API 오류 발생: {e}")
        return model_list

def get_perplexity_llm_models(api_key: str = None):
    import perplexity
    from perplexity import Perplexity

    model_list = []

    api_models = [
        "sonar",
        "sonar-pro",
        "sonar-reasoning",
        "sonar-reasoning-pro",
        "sonar-deep-research",
    ]

    if not api_key:
        model_list.append("Perplexity API Key가 필요합니다.")
        return model_list

    client = Perplexity(api_key=api_key)

    try:
        test = client.search.create(query="test")

        for m in api_models:
            model_list.append(m)
            
        return model_list

    except perplexity.AuthenticationError as e:
        model_list.append(f"Perplexity API 오류 발생: {e}")
        return model_list

def get_xai_llm_models(api_key: str = None):
    import xai_sdk

    model_list = []

    if not api_key:
        model_list.append("XAI API Key가 필요합니다.")
        return model_list

    client = xai_sdk.Client(api_key=api_key)

    try:
        model = client.models.list_language_models()
        for m in model:
            model_list.append(m.name)

        return model_list

    except Exception as e:
        model_list.append(f"XAI API 오류 발생: {e}")
        return model_list

def get_mistralai_llm_models(api_key: str = None):
    import mistralai
    from mistralai import Mistral

    model_list = []

    LLM_ALIASES = ["mistral-large-pixtral-2411", "mistral-medium", "mistral-tiny", "mistral-tiny-2312", 'mistral-tiny-2407', 'open-mistral-7b', 'open-mistral-nemo', 'voxtral-mini-transcribe', 'latest']

    if not api_key:
        model_list.append("Mistral AI API Key가 필요합니다.")
        return model_list

    client = Mistral(api_key=api_key)
    
    try:
        model = client.models.list()
        for m in model.data:
            if m.capabilities.completion_chat and not m.deprecation and all(x not in m.id for x in LLM_ALIASES):
                model_list.append(m.id)

        return model_list
    
    except Exception as e:
        model_list.append(f"Mistral AI API 오류 발생: {e}")
        return model_list

llm_api_models = []
lmstudio_models = get_lmstudio_models()
ollama_models = get_ollama_models()
oobabooga_models = get_oobabooga_models()
openai_api_models = get_openai_llm_models(load_env_variables('OPENAI_API_KEY'))
anthropic_api_models = get_anthropic_llm_models(load_env_variables('ANTHROPIC_API_KEY'))
google_genai_api_models = get_google_genai_llm_models(load_env_variables('GEMINI_API_KEY'))
perplexity_api_models = get_perplexity_llm_models(load_env_variables('PERPLEXITY_API_KEY'))
xai_api_models = get_xai_llm_models(load_env_variables('XAI_API_KEY'))
mistralai_api_models = get_mistralai_llm_models(load_env_variables('MISTRAL_API_KEY'))

openrouter_api_models = [
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-4-maverick",
    "qwen/qwen3-235b-a22b:free",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "qwen/qwen3-max",
    "mistralai/mistral-small-3.2-24b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "mistralai/magistral-small-2506",
    "moonshotai/kimi-k2",
    "moonshotai/kimi-k2:free",
    "moonshotai/kimi-k2-0905",
    "z-ai/glm-4.5-air",
    "z-ai/glm-4.5-air:free",
    "z-ai/glm-4.5",
    "z-ai/glm-4.5v",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-chat-v3.1:free",
    "minimax/minimax-01",
    "minimax/minimax-m1",
]

huggingface_inference_api_models = [
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    'meta-llama/Llama-4-Scout-17B-16E-Instruct:groq',
    'meta-llama/Llama-4-Scout-17B-16E-Instruct:novita',
    'meta-llama/Llama-4-Scout-17B-16E-Instruct:nscale',
    'meta-llama/Llama-4-Scout-17B-16E-Instruct:together',
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct:groq",
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:novita',
    'Qwen/Qwen3-VL-30B-A3B-Instruct',
    'Qwen/Qwen3-VL-30B-A3B-Thinking',
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Thinking",
    'Qwen/Qwen3-Next-80B-A3B-Instruct',
    'Qwen/Qwen3-Next-80B-A3B-Instruct:novita',
    'Qwen/Qwen3-Next-80B-A3B-Instruct:hyperbolic',
    'Qwen/Qwen3-Next-80B-A3B-Instruct:together',
    'Qwen/Qwen3-Next-80B-A3B-Thinking',
    'Qwen/Qwen3-Next-80B-A3B-Thinking:novita',
    'Qwen/Qwen3-Next-80B-A3B-Thinking:hyperbolic',
    'Qwen/Qwen3-Next-80B-A3B-Thinking:together',
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2-Instruct-0905",
    "moonshotai/Kimi-K2-Thinking",
    'zai-org/GLM-4.1V-9B-Thinking',
    "zai-org/GLM-4.5",
    "zai-org/GLM-4.5:zai-org",
    "zai-org/GLM-4.5-Air",
    "zai-org/GLM-4.5-Air:zai-org",
    "zai-org/GLM-4.5-Air-FP8",
    'zai-org/GLM-4.5V',
    'zai-org/GLM-4.5V:zai-org',
    'zai-org/GLM-4.5V-FP8'
    "zai-org/GLM-4.6",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-ai/DeepSeek-V3.1",
    'MiniMaxAI/MiniMax-M1-80k',
    "MiniMaxAI/MiniMax-M2",
    "CohereLabs/command-a-vision-07-2025",
    "CohereLabs/command-a-reasoning-08-2025",
]

llm_api_models.extend(lmstudio_models)
llm_api_models.extend(ollama_models)
llm_api_models.extend(openai_api_models)
llm_api_models.extend(anthropic_api_models)
llm_api_models.extend(google_genai_api_models)
llm_api_models.extend(perplexity_api_models)
llm_api_models.extend(xai_api_models)
llm_api_models.extend(mistralai_api_models)
llm_api_models.extend(openrouter_api_models)
llm_api_models.extend(mistralai_api_models)