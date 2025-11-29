from src.models.models import default_device
from src.models import provider_llm_models, provider_diffusion_models
from src.models.api_models import api_models, diffusion_api_models, tts_api_models
from src.models.local_llm_models import transformers_local, gguf_local, mlx_local
from src.models.known_hf_models import known_hf_models
from src.models.local_diffusion_models import diffusers_local, checkpoints_local
from src.models.local_tts_models import vits_local

lmstudio_llm_models = provider_llm_models.lmstudio_models
ollama_llm_models = provider_llm_models.ollama_models
openai_llm_api_models = provider_llm_models.openai_api_models
anthropic_llm_api_models = provider_llm_models.anthropic_api_models
google_genai_llm_api_models = provider_llm_models.google_genai_api_models
perplexity_llm_api_models = provider_llm_models.perplexity_api_models
xai_llm_api_models = provider_llm_models.xai_api_models
mistralai_llm_api_models = provider_llm_models.mistralai_api_models
openrouter_llm_api_models = provider_llm_models.openrouter_api_models
huggingface_inference_llm_api_models = provider_llm_models.huggingface_inference_api_models
llm_api_models = provider_llm_models.llm_api_models


__all__=[
    'default_device',
    'api_models',
    'transformers_local',
    'gguf_local',
    'mlx_local',
    'known_hf_models',
    'diffusion_api_models',
    "diffusers_local",
    "checkpoints_local",
    "tts_api_models",
    "vits_local"
]

PROVIDER_LIST = ["openai", "anthropic", "google-genai", "perplexity", "xai", 'mistralai', "openrouter", "hf-inference", "ollama", "lmstudio", "self-provided"]

REASONING_BAN = ["non-reasoning"]
REASONING_CONTROLABLE = ["qwen3", "gpt-oss", "gpt-5", "claude-sonnet-4", "claude-opus-4", "claude-haiku-4", "gemini-2.5-flash", "gemini-3"]
REASONING_KWD = ["reasoning", "qwq", "r1", "deepseek-r1", "think", "thinking", "deephermes", "hermes-4", "magistral", "o1", "o3", "o4", "gemini-2.5-pro"] + REASONING_CONTROLABLE