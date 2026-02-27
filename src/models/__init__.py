from typing import Literal, List

from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES

from src.models.models import default_device
from src.models import provider_llm_models, provider_vision_models
from src.models.api_models import api_models, diffusion_api_models, tts_api_models
from src.models.local_llm_models import transformers_local, vllm_local, gguf_local, mlx_local
from src.models.known_hf_models import known_hf_models
from src.models.local_diffusion_models import diffusers_local, checkpoints_local
from src.models.local_tts_models import vits_local

lmstudio_llm_models = provider_llm_models.lmstudio_models
ollama_llm_models = provider_llm_models.ollama_models
oobabooga_llm_models = provider_llm_models.oobabooga_models
vllm_llm_api_models = provider_llm_models.vllm_api_models
openai_llm_api_models = provider_llm_models.openai_api_models
anthropic_llm_api_models = provider_llm_models.anthropic_api_models
google_genai_llm_api_models = provider_llm_models.google_genai_api_models
perplexity_llm_api_models = provider_llm_models.perplexity_api_models
xai_llm_api_models = provider_llm_models.xai_api_models
mistralai_llm_api_models = provider_llm_models.mistralai_api_models
openrouter_llm_api_models = provider_llm_models.openrouter_api_models
huggingface_inference_llm_api_models = provider_llm_models.huggingface_inference_api_models
llm_api_models = provider_llm_models.llm_api_models

comfyui_image_models = provider_vision_models.comfyui_models
comfyui_image_loras = provider_vision_models.comfyui_loras
comfyui_image_vae = provider_vision_models.comfyui_vae
comfyui_image_controlnet = provider_vision_models.comfyui_controlnet
comfyui_image_text_encoders = provider_vision_models.comfyui_text_encoders
comfyui_image_clip = provider_vision_models.comfyui_clip
comfyui_image_clip_vision = provider_vision_models.comfyui_clip_vision
comfyui_image_embeddings = provider_vision_models.comfyui_embeddings
comfyui_image_diffusion_models = provider_vision_models.comfyui_diffusion_models
comfyui_image_pretrained_models = provider_vision_models.comfyui_pretrained_models
comfyui_image_inpaint_models = provider_vision_models.comfyui_inpaint_models
comfyui_image_ipadapter = provider_vision_models.comfyui_ipadapter
comfyui_image_unet = provider_vision_models.comfyui_unet

google_genai_image_models = provider_vision_models.google_genai_image_api_models
google_genai_video_models = provider_vision_models.google_genai_video_api_models

openai_image_api_models = provider_vision_models.openai_image_api_models
openai_video_api_models = provider_vision_models.openai_video_api_models

__all__=[
    'default_device',
    'api_models',
    'transformers_local',
    'vllm_local',
    'gguf_local',
    'mlx_local',
    'known_hf_models',
    'diffusion_api_models',
    "diffusers_local",
    "checkpoints_local",
    "tts_api_models",
    "vits_local"
]

# PROVIDER_LIST = ["openai", "anthropic", "google-genai", "perplexity", "xai", 'mistralai', "openrouter", "hf-inference", "ollama", "lmstudio", "oobabooga", "self-provided"]

PROVIDER_LIST = ["openai", "anthropic", "google-genai", "perplexity", "xai", 'mistralai', "openrouter", "hf-inference", "ollama", "lmstudio", "vllm-api", "oobabooga", "local-ai", "self-provided"]

# IMAGE_PROVIDER_LIST = ["openai", 'google-genai', 'xai', 'hf-inference', 'comfyui', 'invokeai', 'drawthings', 'sd-webui', 'self-provided']

GPT_IMAGE_ALLOWED_SIZES = ["1024x1024","1024x1536","1536x1024"]

IMAGE_PROVIDER_LIST = ["openai", 'google-genai', 'comfyui', 'self-provided']

TTS_PROVIDER_LIST = ["gtts", "edgetts"]

REASONING_BAN = ["non-reasoning"]
REASONING_CONTROLABLE = ["qwen3", "qwen3.5", "cogito", "gpt-oss", "gpt-5", "claude-sonnet-4", "claude-opus-4", "claude-haiku-4", "gemini-2.5-flash", "gemini-3"]
REASONING_KWD = ["reasoning", "qwq", "r1", "deepseek-r1", "think", "thinking", "deephermes", "hermes-4", "magistral", "o1", "o3", "o4", "gemini-2.5-pro"] + REASONING_CONTROLABLE

IS_MULTIMODAL_LOCAL = list(MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES.values())
IS_ANY_TO_ANY = list(set(IS_MULTIMODAL_LOCAL) - set(list(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values())))
IS_IMAGE_TEXT_TO_TEXT = list(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values())
IS_MULTIMODAL_API = ["gpt-4o", "gpt-4.1", "gpt-5", "claude-3", "claude-opus-4", "claude-sonnet-4", "claude-haiku-4", "gemini-2.5", 'paligemma', "gemini-3", 'gemma-3-4b', 'gemma-3-12b', "gemma-3-27b", 'gemma-3n', "grok-2-vision", "grok-4", "sonar", 'llama-3.2-11b-vision', 'llama-3.2-90b-vision', "llama-4", 'qwen2-vl', 'qwen2.5-vl', "qwen3-vl", "qwen3.5", 'pixtral', 'mistral-small-2506', "mistral-small-3.1", "mistral-small-3.2", 'mistral-medium-2508', 'magistral-small-2509', 'magistral-medium-2509', 'ministral-3', 'ministral-3b-2512', 'ministral-8b-2512', 'ministral-14b-2512',"mistral-large-2512", 'mistral-large-3', "cogito-v2-preview-llama-109b", "kimi-k2.5", 'glm-4v', 'glm-4.1v', 'glm-4.5v', 'glm-4.6v', 'command-a-vision-07-2025', 'aya-vision', 'deepseek-vl', 'janus', 'ernie-4.5-vl', 'fastvlm', 'florence-2', 'kosmos-2.5-chat', 'fuyu', 'idefics', 'instructblip', 'llava', 'smolvlm']
IS_OMNI_API = ['gpt-realtime', "gemini-2.5", "gemini-3", "grok-4", 'voxtral-mini-4b-realtime-2602' 'qwen2.5-omni', 'qwen3-omni' 'phi-4-multimodal']