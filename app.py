# app.py
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", module="gradio")
warnings.filterwarnings("ignore", module="torchao")
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings('ignore', module='pydantic')

config_dir = Path.home() / ".ai-companion"
env_file = config_dir  / ".env"

if not env_file.exists():
    config_dir.mkdir(exists_ok=True)
    with open(env_file, 'w') as f:
        f.write('')

import gradio as gr
# from gradio_i18n import Translate, translate_blocks, gettext as _
# from src.common.html import css


from translations import i18n

# gr.I18n(lang_store)

from src import os_name, arch, is_wsl, args, __version__, logger
# from src.start_app import initialize_app
# from src import app

from src.start_app import (
    app_state,
    ui_component,
    on_app_start,
    # register_speech_manager_state, # moved to register_global_state
    # shared_on_app_start,
    register_global_state,
    load_initial_data
)

# Import MCP tools
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

# from src.main import header
from src.main.chatbot import chat_main
from src.main.image_generation import diff_main
from src.main.tts import get_tts_models

from src.common_blocks import create_page_header, get_language_code
# from src.main.header import HeaderUIComponent
# from src.common_blocks import HeaderUIComponent, NavbarUIComponent, BottomNavUIComponent

logger.info(f"AI Companion Version: {__version__}")
logger.info(f"Detected OS: {os_name}, Architecture: {arch}")

# CRITICAL: app_state primitives must be loaded BEFORE importing pages,
# because page construction typically reads initial state from app_state.
load_initial_data()

# Import Pages
from src.pages import header
from src.pages import audio, chat, image_gen, storyteller, translator, download, settings, mcp_client, mcp_tools

# Global Initialization
# Creating a dummy block to run initialization if needed, 
# or just running functions that don't require Gradio context (some might).
# However, register_speech_manager_state() creates gr.State(), so it MUST be inside a Blocks context.
# Since app.py is the main Blocks context, we should do it there.

def initialize_global_state():
    register_global_state()

# def render_page_layout(demo: gr.Blocks):
#     """
#     Renders the standard page layout: Header -> Navbar -> Page Content.
#     """
#     HeaderUIComponent.create_header_container()
#     HeaderUIComponent.create_navbar()

#     return demo.render()


# with gr.Blocks(title="AI Companion", fill_height=True, fill_width=True, css_paths="html/css/style.css") as demo:
# header = HeaderUIComponent()


# with gr.Blocks(title="AI Companion", fill_height=True, fill_width=True, css_paths="html/css/style.css") as demo:
with gr.Blocks(title="AI Companion", fill_height=True, fill_width=True) as demo:
    # demo.route("Chat", "/chat")
    # 1. Global State Registration
    initialize_global_state()
    header.demo.render()
    with gr.Tab("Chat", elem_classes="tab"):
        chat.demo.render()
    with gr.Tab("Image Gen", elem_classes="tab"):
        image_gen.demo.render()
    with gr.Tab("Storyteller", elem_classes="tab"):
        storyteller.demo.render()
    with gr.Tab("Audio", elem_classes="tab"):
        audio.demo.render()
    with gr.Tab("Translator", elem_classes="tab"):
        translator.demo.render()
    with gr.Tab("Settings", elem_classes="tab"):
        settings.demo.render()
    with gr.Tab("MCP Client", elem_classes="tab"):
        mcp_client.demo.render()
    with gr.Tab("Download", elem_classes="tab"):
        download.demo.render()

    with gr.Tab("MCP Tools", elem_classes="tab"):
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

    header.language_dropdown.change(
        fn=header.on_header_language_change,
        inputs=[header.language_dropdown],
        outputs=[
            header.page_header.title,
            header.language_dropdown
        ]
    ).then(
        fn=chat.on_chat_language_change,
        inputs=[header.language_dropdown, chat.character_dropdown],
        outputs=[
            chat.text_model_provider_dropdown,
            chat.text_model_type_dropdown,
            chat.text_model_dropdown,
            chat.text_api_key_text,
            chat.text_lora_dropdown,
            chat.system_message_accordion,
            chat.system_message_box,
            chat.text_advanced_settings,
            chat.text_seed_input,
            chat.text_temperature_slider,
            chat.text_top_k_slider,
            chat.text_top_p_slider,
            chat.text_repetition_penalty_slider,
            chat.reset_btn,
            chat.reset_all_btn,
            app_state.selected_language_state
        ]
    ).then(
        fn=image_gen.on_image_gen_language_change,
        inputs=[header.language_dropdown],
        outputs=[
            image_gen.diffusion_model_provider_dropdown,
            image_gen.diffusion_model_type_dropdown,
            image_gen.diffusion_model_dropdown,
            image_gen.diffusion_api_key_text,
            image_gen.diffusion_lora_multiselect,
        ]
    )

# if __name__ == "__main__":
#     demo.launch()

# import gradio as gr

# from src import os_name, arch, args, __version__
# from src.start_app import initialize_app
# from src import app

# with gr.Blocks(title="AI Companion", fill_height=True, fill_width=True, css_paths="html/css/style.css") as demo:
# # with gr.Blocks(title="AI Companion", fill_height=True, fill_width=True) as demo:
#     # initialize_app()
#     app.demo.render()

if __name__=="__main__":
    
    if os_name == "Darwin" and arch == "x86_64":
        raise EnvironmentError("ERROR: AI Companion for Local Machines no longer supports Intel CPU-based Macs.\nIf you are using an Intel CPU-based Macs, we recommend that you consider migrating to an Apple Silicon Based Macs or a Windows PC or Linux machine with an Nvidia GPU environment. If you have difficulty migrating from an Intel CPU-based Macs, you can use a companion application that supports Intel CPU-based Macs instead.")
    if os_name == "Windows" and not is_wsl:
        warnings.warn("AI Companion for Local Machines is optimized for UNIX/Linux kernel-based operating systems. While it can be used on Windows, GPU acceleration is unavailable when running directly on Windows. To properly use AI Companion for Local Machines on Windows, we recommend using it within a WSL2 environment.")
    if args.listen:
        host="0.0.0.0"
    else:
        host="127.0.0.1"
    
    # demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_name=host, server_port=args.port, mcp_server=args.mcp_server, pwa=args.pwa, i18n=i18n)

    demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_name=host, server_port=args.port, mcp_server=args.mcp_server, pwa=args.pwa, css_paths="html/css/style.css", i18n=i18n)

