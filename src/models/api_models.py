api_models = []
diffusion_api_models = []
diffusion_video_api_models = []
tts_api_models = []
stt_api_models = []
stream_api_models = []

openai_api_models = [
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4.1-2025-04-14",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-5-pro-2025-10-06",
    "gpt-5-chat-latest",
    "gpt-5.1-2025-11-13",
]

anthropic_api_models = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
]

google_ai_api_models = [
    "gemma-3-27b-it",
    "gemma-3n-e2b-it",
    "gemma-3n-e4b-it",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-image",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

perplexity_api_models = [
    "sonar",
    "sonar-pro",
    "sonar-reasoning",
    "sonar-reasoning-pro",
    "sonar-deep-research",
]

xai_api_models = [
    "grok-2-vision-1212",
    "grok-3-mini",
    "grok-3",
    "grok-4-0709",
    "grok-4-fast-non-reasoning",
    "grok-4-fast-reasoning"
]

openrouter_api_models = [
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-4-scout:free",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-maverick:free",
    "qwen/qwen2.5-vl-72b-instruct",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-235b-a22b:free",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "qwen/qwen3-max",
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
    "gpt-image-1-mini"
]

google_image_api_models = [
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-4.0-fast-generate-001"
]

xai_image_api_models = [
    "grok-2-image-1212"
]

diffusion_api_models.extend(openai_image_api_models)
diffusion_api_models.extend(google_image_api_models)
diffusion_api_models.extend(xai_image_api_models)

openai_video_api_models = [
    "sora-2",
    "sora-2-pro"
]

google_video_api_models=[
    "veo-2.0-generate-001",
    "veo-3.0-fast-generate-001",
    "veo-3.0-generate-001",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-generate-preview"
]

diffusion_video_api_models.extend(openai_video_api_models)
diffusion_video_api_models.extend(google_video_api_models)

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

openai_stt_api_models = [
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe"
]

stt_api_models.extend(openai_stt_api_models)

embedding_api_models = [
    "text-embedding-3-large",
    "text-embedding-3-small",
    "gemini-embedding-001"
]

openai_realtime_api_models = [
    "gpt-realtime",
    "gpt-realtime-mini"
]

openai_audio_api_models = [
    "gpt-audio",
    "gpt-audio-mini"
]

google_audio_api_models = [
    "gemini-2.5-flash-native-audio-preview-09-2025",
]

stream_api_models.extend(openai_realtime_api_models)
stream_api_models.extend(openai_audio_api_models)
stream_api_models.extend(google_audio_api_models)
