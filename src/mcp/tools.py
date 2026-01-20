# MCP Tools for AI Companion
# These functions are exposed as MCP tools when mcp_server=True in Gradio launch

import gradio as gr
from typing import Literal, List, Optional, Any
from PIL import Image

from src.models.models import generate_answer, generate_chat_title, get_all_local_models
from src.common.database import (
    get_existing_sessions,
    get_existing_sessions_with_names,
    load_chat_from_db,
    save_chat_history_db
)
from src.models import (
    openai_llm_api_models,
    anthropic_llm_api_models,
    google_genai_llm_api_models,
    openai_image_api_models,
    google_genai_image_models,
    comfyui_image_models
)
from src import logger


def chat_completion(
    message: str,
    model: str = "gpt-4o",
    provider: Literal[
        "openai", "anthropic", "google-genai", "perplexity",
        "xai", "mistralai", "openrouter", "hf-inference",
        "ollama", "lmstudio", "self-provided"
    ] = "openai",
    system_message: str = "You are a helpful AI assistant.",
    api_key: str = "",
    temperature: float = 0.7,
    max_length: int = -1
) -> str:
    """
    Generate a chat completion response using various LLM providers.

    Args:
        message: The user's input message to respond to.
        model: The model identifier to use (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022').
        provider: The AI provider to use for generation.
        system_message: The system prompt that defines the AI's behavior.
        api_key: API key for the selected provider (required for most providers).
        temperature: Controls randomness in generation (0.0-2.0, higher = more random).
        max_length: Maximum tokens to generate (-1 for no limit).

    Returns:
        The AI's response text.
    """
    try:
        # Build conversation history
        history = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [{"type": "text", "text": message}]}
        ]

        # Determine model type for self-provided models
        model_type = None
        if provider == "self-provided":
            local_models = get_all_local_models()
            if model in local_models.get("transformers", []):
                model_type = "transformers"
            elif model in local_models.get("gguf", []):
                model_type = "gguf"
            elif model in local_models.get("mlx", []):
                model_type = "mlx"
            else:
                model_type = "transformers"

        # Generate response
        response = generate_answer(
            history=history,
            selected_model=model,
            provider=provider,
            model_type=model_type,
            api_key=api_key,
            temperature=temperature,
            max_length=max_length,
            seed=42,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.1,
            enable_thinking=False
        )

        return response

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        return f"Error: {str(e)}"


def list_available_models(
    category: Literal["llm", "image", "all"] = "all"
) -> dict:
    """
    List all available AI models for chat and image generation.

    Args:
        category: Filter by model category ('llm' for chat models, 'image' for image models, 'all' for both).

    Returns:
        A dictionary containing available models organized by provider.
    """
    result = {}

    if category in ["llm", "all"]:
        # Get local models
        local_models = get_all_local_models()

        result["llm"] = {
            "openai": openai_llm_api_models,
            "anthropic": anthropic_llm_api_models,
            "google-genai": google_genai_llm_api_models,
            "local_transformers": local_models.get("transformers", []),
            "local_gguf": local_models.get("gguf", []),
            "local_mlx": local_models.get("mlx", [])
        }

    if category in ["image", "all"]:
        result["image"] = {
            "openai": openai_image_api_models,
            "google-genai": google_genai_image_models,
            "comfyui": comfyui_image_models
        }

    return result


def list_chat_sessions() -> List[dict]:
    """
    List all existing chat sessions with their IDs and names.

    Returns:
        A list of dictionaries containing session information (id, name).
    """
    try:
        sessions = get_existing_sessions_with_names()
        return [{"id": sid, "name": name} for sid, name in sessions]
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return []


def get_chat_history(session_id: str) -> List[dict]:
    """
    Retrieve the chat history for a specific session.

    Args:
        session_id: The unique identifier of the chat session.

    Returns:
        A list of message dictionaries containing the conversation history.
    """
    try:
        history = load_chat_from_db(session_id)
        # Filter out system messages for cleaner output
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
            if msg["role"] != "system"
        ]
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []


def generate_title(
    content: str,
    model: str = "gpt-4o-mini",
    provider: Literal["openai", "anthropic", "google-genai"] = "openai"
) -> str:
    """
    Generate a concise title for given content (useful for naming chat sessions).

    Args:
        content: The text content to generate a title for.
        model: The model to use for title generation.
        provider: The AI provider to use.

    Returns:
        A generated title string.
    """
    try:
        title = generate_chat_title(
            first_message=content,
            selected_model=model,
            provider=provider
        )
        return title or "Untitled"
    except Exception as e:
        logger.error(f"Error generating title: {e}")
        return "Untitled"


def translate_text(
    text: str,
    source_language: str = "auto",
    target_language: str = "en",
    model: str = "gpt-4o",
    provider: Literal["openai", "anthropic", "google-genai"] = "openai",
    api_key: str = ""
) -> str:
    """
    Translate text from one language to another using AI.

    Args:
        text: The text to translate.
        source_language: Source language code (e.g., 'ko', 'ja', 'zh') or 'auto' for auto-detection.
        target_language: Target language code (e.g., 'en', 'ko', 'ja').
        model: The model to use for translation.
        provider: The AI provider to use.
        api_key: API key for the selected provider.

    Returns:
        The translated text.
    """
    try:
        if source_language == "auto":
            system_prompt = f"You are a professional translator. Translate the following text to {target_language}. Only output the translation, nothing else."
        else:
            system_prompt = f"You are a professional translator. Translate the following text from {source_language} to {target_language}. Only output the translation, nothing else."

        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": text}]}
        ]

        response = generate_answer(
            history=history,
            selected_model=model,
            provider=provider,
            api_key=api_key,
            temperature=0.3,  # Lower temperature for more consistent translations
            max_length=-1,
            seed=42,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.1,
            enable_thinking=False
        )

        return response

    except Exception as e:
        logger.error(f"Translation error: {e}")
        return f"Error: {str(e)}"


def summarize_text(
    text: str,
    style: Literal["concise", "detailed", "bullet_points"] = "concise",
    model: str = "gpt-4o",
    provider: Literal["openai", "anthropic", "google-genai"] = "openai",
    api_key: str = ""
) -> str:
    """
    Summarize text using AI with different summary styles.

    Args:
        text: The text to summarize.
        style: Summary style - 'concise' (brief), 'detailed' (comprehensive), or 'bullet_points'.
        model: The model to use for summarization.
        provider: The AI provider to use.
        api_key: API key for the selected provider.

    Returns:
        The summarized text.
    """
    try:
        style_prompts = {
            "concise": "Provide a brief, concise summary in 2-3 sentences.",
            "detailed": "Provide a comprehensive summary covering all main points.",
            "bullet_points": "Provide a summary as a bulleted list of key points."
        }

        system_prompt = f"You are a helpful assistant skilled at summarization. {style_prompts.get(style, style_prompts['concise'])} Only output the summary, nothing else."

        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": text}]}
        ]

        response = generate_answer(
            history=history,
            selected_model=model,
            provider=provider,
            api_key=api_key,
            temperature=0.5,
            max_length=-1,
            seed=42,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.1,
            enable_thinking=False
        )

        return response

    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return f"Error: {str(e)}"


def analyze_image(
    image_path: str,
    question: str = "Describe this image in detail.",
    model: str = "gpt-4o",
    provider: Literal["openai", "anthropic", "google-genai"] = "openai",
    api_key: str = ""
) -> str:
    """
    Analyze an image using a vision-capable AI model.

    Args:
        image_path: Path to the image file to analyze.
        question: The question or instruction about the image.
        model: The vision model to use (must support image input).
        provider: The AI provider to use.
        api_key: API key for the selected provider.

    Returns:
        The AI's analysis of the image.
    """
    try:
        import base64

        # Read and encode the image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # Determine MIME type
        ext = image_path.lower().split('.')[-1]
        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'webp': 'image/webp',
            'gif': 'image/gif'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')

        # Build multimodal message
        history = [
            {"role": "system", "content": "You are a helpful assistant that can analyze images."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image_url": f"data:{mime_type};base64,{image_data}"}
                ]
            }
        ]

        response = generate_answer(
            history=history,
            selected_model=model,
            provider=provider,
            api_key=api_key,
            image_input=[image_path],
            temperature=0.7,
            max_length=-1,
            seed=42,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.1,
            enable_thinking=False
        )

        return response

    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return f"Error: {str(e)}"


def register_mcp_tools(demo: gr.Blocks):
    """
    Register MCP tools with a Gradio Blocks instance.

    This function creates Gradio Interface components for each MCP tool
    and adds them to the provided Blocks instance with appropriate api_names.

    Args:
        demo: The Gradio Blocks instance to register tools with.
    """

    # Chat completion tool
    gr.Interface(
        fn=chat_completion,
        inputs=[
            gr.Textbox(label="Message", placeholder="Enter your message..."),
            gr.Textbox(label="Model", value="gpt-4o"),
            gr.Dropdown(
                label="Provider",
                choices=["openai", "anthropic", "google-genai", "perplexity",
                        "xai", "mistralai", "openrouter", "hf-inference",
                        "ollama", "lmstudio", "self-provided"],
                value="openai"
            ),
            gr.Textbox(label="System Message", value="You are a helpful AI assistant."),
            gr.Textbox(label="API Key", type="password"),
            gr.Slider(label="Temperature", minimum=0, maximum=2, value=0.7, step=0.1),
            gr.Number(label="Max Length", value=-1)
        ],
        outputs=gr.Textbox(label="Response"),
        api_name="chat"
    ).render()

    # List models tool
    gr.Interface(
        fn=list_available_models,
        inputs=[
            gr.Dropdown(
                label="Category",
                choices=["all", "llm", "image"],
                value="all"
            )
        ],
        outputs=gr.JSON(label="Available Models"),
        api_name="list_models"
    ).render()

    # List sessions tool
    gr.Interface(
        fn=list_chat_sessions,
        inputs=[],
        outputs=gr.JSON(label="Chat Sessions"),
        api_name="list_sessions"
    ).render()

    # Get chat history tool
    gr.Interface(
        fn=get_chat_history,
        inputs=[
            gr.Textbox(label="Session ID")
        ],
        outputs=gr.JSON(label="Chat History"),
        api_name="get_history"
    ).render()

    # Translation tool
    gr.Interface(
        fn=translate_text,
        inputs=[
            gr.Textbox(label="Text", placeholder="Text to translate..."),
            gr.Textbox(label="Source Language", value="auto"),
            gr.Textbox(label="Target Language", value="en"),
            gr.Textbox(label="Model", value="gpt-4o"),
            gr.Dropdown(
                label="Provider",
                choices=["openai", "anthropic", "google-genai"],
                value="openai"
            ),
            gr.Textbox(label="API Key", type="password")
        ],
        outputs=gr.Textbox(label="Translation"),
        api_name="translate"
    ).render()

    # Summarization tool
    gr.Interface(
        fn=summarize_text,
        inputs=[
            gr.Textbox(label="Text", placeholder="Text to summarize..."),
            gr.Dropdown(
                label="Style",
                choices=["concise", "detailed", "bullet_points"],
                value="concise"
            ),
            gr.Textbox(label="Model", value="gpt-4o"),
            gr.Dropdown(
                label="Provider",
                choices=["openai", "anthropic", "google-genai"],
                value="openai"
            ),
            gr.Textbox(label="API Key", type="password")
        ],
        outputs=gr.Textbox(label="Summary"),
        api_name="summarize"
    ).render()

    # Image analysis tool
    gr.Interface(
        fn=analyze_image,
        inputs=[
            gr.File(label="Image", file_types=["image"]),
            gr.Textbox(label="Question", value="Describe this image in detail."),
            gr.Textbox(label="Model", value="gpt-4o"),
            gr.Dropdown(
                label="Provider",
                choices=["openai", "anthropic", "google-genai"],
                value="openai"
            ),
            gr.Textbox(label="API Key", type="password")
        ],
        outputs=gr.Textbox(label="Analysis"),
        api_name="analyze_image"
    ).render()

    # Title generation tool
    gr.Interface(
        fn=generate_title,
        inputs=[
            gr.Textbox(label="Content", placeholder="Content to generate title for..."),
            gr.Textbox(label="Model", value="gpt-4o-mini"),
            gr.Dropdown(
                label="Provider",
                choices=["openai", "anthropic", "google-genai"],
                value="openai"
            )
        ],
        outputs=gr.Textbox(label="Generated Title"),
        api_name="generate_title"
    ).render()

    logger.info("MCP tools registered successfully")
