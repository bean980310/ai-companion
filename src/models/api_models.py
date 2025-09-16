api_models = []
diffusion_api_models = []
tts_api_models = []

openai_api_models = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o1",
    "o1-pro",
    "o3-mini",
    "o3",
    "o3-pro",
    "o4-mini"
    "gpt-oss-20b",
    "gpt-oss-120b",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat-latest",
]

anthropic_api_models = [
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-7-sonnet-latest",
    "claude-sonnet-4-0",
    "claude-opus-4-0",
    "claude-opus-4-1",
]

google_ai_api_models = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-preview-image-generation",
    "gemma-3-27b-it",
    "gemma-3n-e2b-it",
    "gemma-3n-e4b-it",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

perplexity_api_models = [
    "sonar",
    "sonar-pro",
    "sonar-reasoning",
    "sonar-reasoning-pro",
    "sonar-deep-research",
]

xai_api_models = [
    "grok-3-mini",
    "grok-3",
    "grok-4",
]

openrouter_api_models = [
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-4-scout:free",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-maverick:free",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-235b-a22b:free",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-max",
    "qwen/qwen2.5-vl-72b-instruct",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "mistralai/mistral-small-24b-instruct-2501",
    "mistralai/mistral-small-24b-instruct-2501:free",
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

api_models.extend(openai_api_models)
api_models.extend(anthropic_api_models)
api_models.extend(google_ai_api_models)
api_models.extend(perplexity_api_models)
api_models.extend(xai_api_models)

openai_image_api_models = [
    "dall-e-3",
    "gpt-image-1",
]

google_image_api_models = [
    "imagen-3.0-generate-002",
    "imagen-4.0-generate-preview-06-06",
    "imagen-4.0-ultra-generate-preview-06-06"
]

xai_image_api_models = [
    "grok-2-image"
]
diffusion_api_models.extend(openai_image_api_models)
diffusion_api_models.extend(google_image_api_models)
diffusion_api_models.extend(xai_image_api_models)

diffusion_video_api_models=[
    "veo-2.0-generate-001",
    "veo-3.0-generate-preview"
]

openai_tts_api_models = [
    "tts-1",
    "tts-1-hd",
    "gpt-4o-mini-tts",
]

google_tts_api_models = [
    "gemini-2.5-pro-preview-tts",
    "gemini-2.5-flash-preview-tts"
]

tts_api_models.extend(openai_tts_api_models)
tts_api_models.extend(google_tts_api_models)

embedding_api_models = [
    "text-embedding-3-large",
    "text-embedding-3-small",
    "gemini-embedding-001"
]