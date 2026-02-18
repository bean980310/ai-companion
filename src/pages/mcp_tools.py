import os
import gradio as gr

from src.common.translations import translation_manager, _
from src.common_blocks import create_page_header, get_language_code

from src.mcp.tools import (
    chat_completion,
    list_available_models,
    list_chat_sessions,
    get_chat_history,
    translate_text,
    summarize_text,
    analyze_image,
    generate_title
)

with gr.Blocks() as demo:
    page_header = create_page_header(page_title_key="main_title")
    language_dropdown = page_header.language_dropdown
    gr.Markdown("# AI Companion MCP Tools")
    gr.Markdown("This page exposes AI Companion functionality as MCP tools.")
    gr.Markdown("Access the MCP server at: `http://localhost:{port}/gradio_api/mcp/sse`")

    with gr.Tab("Chat"):
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
        )

    with gr.Tab("Models"):
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
        )

    with gr.Tab("Sessions"):
        gr.Interface(
            fn=list_chat_sessions,
            inputs=[],
            outputs=gr.JSON(label="Chat Sessions"),
            api_name="list_sessions"
        )

        gr.Interface(
            fn=get_chat_history,
            inputs=[gr.Textbox(label="Session ID")],
            outputs=gr.JSON(label="Chat History"),
            api_name="get_history"
        )

    with gr.Tab("Translation"):
        gr.Interface(
            fn=translate_text,
            inputs=[
                gr.Textbox(label="Text", placeholder="Text to translate...", lines=5),
                gr.Textbox(label="Source Language", value="auto", info="Use 'auto' for auto-detection or language codes like 'ko', 'ja', 'zh'"),
                gr.Textbox(label="Target Language", value="en"),
                gr.Textbox(label="Model", value="gpt-4o"),
                gr.Dropdown(
                    label="Provider",
                    choices=["openai", "anthropic", "google-genai"],
                    value="openai"
                ),
                gr.Textbox(label="API Key", type="password")
            ],
            outputs=gr.Textbox(label="Translation", lines=5),
            api_name="translate"
        )

    with gr.Tab("Summarization"):
        gr.Interface(
            fn=summarize_text,
            inputs=[
                gr.Textbox(label="Text", placeholder="Text to summarize...", lines=10),
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
            outputs=gr.Textbox(label="Summary", lines=5),
            api_name="summarize"
        )

    with gr.Tab("Image Analysis"):
        gr.Interface(
            fn=analyze_image,
            inputs=[
                gr.Image(label="Image", type="filepath"),
                gr.Textbox(label="Question", value="Describe this image in detail."),
                gr.Textbox(label="Model", value="gpt-4o"),
                gr.Dropdown(
                    label="Provider",
                    choices=["openai", "anthropic", "google-genai"],
                    value="openai"
                ),
                gr.Textbox(label="API Key", type="password")
            ],
            outputs=gr.Textbox(label="Analysis", lines=10),
            api_name="analyze_image"
        )

    with gr.Tab("Title Generation"):
        gr.Interface(
            fn=generate_title,
            inputs=[
                gr.Textbox(label="Content", placeholder="Content to generate title for...", lines=5),
                gr.Textbox(label="Model", value="gpt-4o-mini"),
                gr.Dropdown(
                    label="Provider",
                    choices=["openai", "anthropic", "google-genai"],
                    value="openai"
                )
            ],
            outputs=gr.Textbox(label="Generated Title"),
            api_name="generate_title"
        )

    # 3. Global Event Handlers (if any remaining)
    # The original grad_ui.py had some global load events.
    # We should ensure those are covered in pages or here.
    
    # demo.load(fn=on_app_start, ...) was in gradio_ui.py
    # Since 'demo' here is the main block, we can attach it.
    # Note: app_state variables are now "Global" in the sense they are created in initialize_global_state()
    # BUT, access to them might need care if they are gr.State objects. 
    # Since python modules share state, it should be fine.
    
    # However, create_main_container used to return settings_button etc.
    # We have removed the global settings button.
        
    # Original load event:
    # demo.load(
    #     fn=on_app_start,
    #     inputs=[], 
    #     outputs=[app_state.session_id_state, app_state.history_state, existing_sessions_dropdown, app_state.character_state, ui_component.text_preset_dropdown, app_state.system_message_state, current_session_display],
    #     queue=False
    # )
    # existing_sessions_dropdown is now inside Settings Page AND Chat Page (session_select_dropdown).
    # This `on_app_start` updates multiple components across different areas.
    # Multipage apps "share the same backend".
    # But components are distinct.
    # If `on_app_start` updates `existing_sessions_dropdown`, it needs a reference to it.
    # But `existing_sessions_dropdown` is inside `settings.demo` or `chat.demo`.
    # We might need to duplicate this load event logic inside each page or pass references.
        
    # In Chat Page:
    # demo.load(fn=chat_bot.refresh_sessions...) is present.
        
    # In Settings Page:
    # We should probably add initialization there too.
        
    pass

if __name__ == "__main__":
    demo.launch()
