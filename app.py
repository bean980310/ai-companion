# app.py
import warnings

warnings.filterwarnings("ignore", module="gradio")
warnings.filterwarnings("ignore", module="torchao")
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings('ignore', module='pydantic')

import gradio as gr
# from gradio_i18n import Translate, translate_blocks, gettext as _
# from src.common.html import css


from translations import i18n

# gr.I18n(lang_store)

from src import os_name, arch, args, __version__, logger
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

# from src.main import header
from src.main.chatbot import chat_main
from src.main.image_generation import diff_main
from src.main.tts import get_tts_models
# from src.main.header import HeaderUIComponent
# from src.common_blocks import HeaderUIComponent, NavbarUIComponent, BottomNavUIComponent

logger.info(f"AI Companion Version: {__version__}")
logger.info(f"Detected OS: {os_name}, Architecture: {arch}")

# CRITICAL: app_state primitives must be loaded BEFORE importing pages,
# because page construction typically reads initial state from app_state.
load_initial_data()

# Import Pages
from src.pages import chat, image_gen, storyteller, tts, translator, download, settings

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
    chat.demo.render()
    # with gr.Row(elem_classes="header-container", scale=1) as head:
    #     with gr.Column(scale=3):
    #         title = gr.Markdown(f"## {i18n('main_title')}", elem_classes="title")
    #         gr.Markdown("### Beta Release")
    #     with gr.Column(scale=1):
    #         settings_button = gr.Button("⚙️", elem_classes="settings-button")
    #     with gr.Column(scale=1):
    #         language_dropdown = gr.Dropdown(
    #             label=i18n('language_select'),
    #             choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
    #             value=translation_manager.get_language_display_name(default_language),
    #             interactive=True,
    #             info=i18n('language_info'),
    #             container=False,
    #             elem_classes="language-selector"
    #         )

    # from src.common import default_language
    # from src.common.translations import translation_manager, _

    # with gr.Row(elem_classes="header-container", scale=1):
    #     with gr.Column(scale=3):
    #         title = gr.Markdown(f"## {_('main_title')}", elem_classes="title")
    #         gr.Markdown("### Beta Release")
    #     with gr.Column(scale=1):
    #         settings_button = gr.Button("⚙️", elem_classes="settings-button")
    #     with gr.Column(scale=1):
    #         language_dropdown = gr.Dropdown(
    #             label=_('language_select'),
    #             choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
    #             value=translation_manager.get_language_display_name(default_language),
    #             interactive=True,
    #             info=_('language_info'),
    #             container=False,
    #             elem_classes="language-selector"
    #         )
    # header.create_header_container()
    # header.create_navbar()
    # chat_page = chat.demo
    # chat_page.render()
    # main_page = render_page_layout(demo)
    # header.demo.render()

# with demo.route("Chat", "/chat"):
#     chat.demo.render()
    
    
    # 2. Pages
    # Home Page (Chat)

# with demo.route("Chat Completion", "/chat"):
#     chat.demo.render()
    
# Other Pages
with demo.route("Image Gen", "/image"):
    # ui_component.head.render()
    # header.demo.render()
    # image_gen_header = HeaderUIComponent()
    # image_gen_header.create_header_container()
    # image_gen_header.create_navbar()
    # header.create_header_container()
    image_gen.demo.render()
    
with demo.route("Storyteller", "/story"):
    # ui_component.head.render()
    # header.demo.render()
    # storyteller_header = HeaderUIComponent()
    # storyteller_header.create_header_container()
    # storyteller_header.create_navbar()
    # header.create_header_container()
    # storyteller_page = storyteller.demo
    storyteller.demo.render()
    
with demo.route("TTS", "/tts"):
    # ui_component.head.render()
    # header.demo.render()
    # tts_header = HeaderUIComponent()
    # tts_header.create_header_container()
    # tts_header.create_navbar()
    # header.create_header_container()
    # tts_page = tts.demo
    tts.demo.render()
    
with demo.route("Translator", "/translate"):
    # ui_component.head.render()
    # header.demo.render()
    # translator_header = HeaderUIComponent()
    # translator_header.create_header_container()
    # translator_header.create_navbar()
    # header.create_header_container()
    # translator_page = translator.demo
    translator.demo.render()
    
with demo.route("Download", "/download"):
#     # ui_component.head.render()
#     # header.demo.render()
#     # download_header = HeaderUIComponent()
#     # download_header.create_header_container()
#     # download_header.create_navbar()
#     # header.create_header_container()
#     download_page = download.demo
    download.demo.render()
    
with demo.route("Settings", "/settings"):
    # ui_component.head.render()
    # header.demo.render()
    # settings_header = HeaderUIComponent()
    # settings_header.create_header_container()
    # settings_header.create_navbar()
    # header.create_header_container()
    # settings_page = settings.demo
    settings.demo.render()
        
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
    if args.listen:
        host="0.0.0.0"
    else:
        host="127.0.0.1"
    
    # demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_name=host, server_port=args.port, mcp_server=args.mcp_server, pwa=args.pwa, i18n=i18n)

    demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_name=host, server_port=args.port, mcp_server=args.mcp_server, pwa=args.pwa, css_paths="html/css/style.css", i18n=i18n)

