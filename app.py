# app.py
import os
import warnings
import json
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

from src.server import app, Request, StreamingResponse

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
    tapped_interface = gr.TabbedInterface([chat.demo, image_gen.demo, storyteller.demo, audio.demo, translator.demo, settings.demo, mcp_client.demo, download.demo, mcp_tools.demo], ["Chat", "Image Gen", "Storyteller", "Audio", "Translator", "Settings", "MCP Client", "Download", "MCP Tools"])

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

    demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_name=host, server_port=args.gradio_server_port, mcp_server=args.mcp_server, pwa=args.pwa, css_paths="html/css/style.css", i18n=i18n)
else:
    if os_name == "Darwin" and arch == "x86_64":
        raise EnvironmentError("ERROR: AI Companion for Local Machines no longer supports Intel CPU-based Macs.\nIf you are using an Intel CPU-based Macs, we recommend that you consider migrating to an Apple Silicon Based Macs or a Windows PC or Linux machine with an Nvidia GPU environment. If you have difficulty migrating from an Intel CPU-based Macs, you can use a companion application that supports Intel CPU-based Macs instead.")
    if os_name == "Windows" and not is_wsl:
        warnings.warn("AI Companion for Local Machines is optimized for UNIX/Linux kernel-based operating systems. While it can be used on Windows, GPU acceleration is unavailable when running directly on Windows. To properly use AI Companion for Local Machines on Windows, we recommend using it within a WSL2 environment.")
    if args.listen:
        host="0.0.0.0"
    else:
        host="127.0.0.1"

    app = gr.mount_gradio_app(app, demo, path="/gradio", server_name=host, server_port=args.gradio_server_port, mcp_server=args.mcp_server, pwa=args.pwa, css_paths="html/css/style.css")