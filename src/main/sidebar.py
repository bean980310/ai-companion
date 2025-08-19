import gradio as gr

from ..start_app import create_tab_side
from .chatbot import ChatbotMain, chat_main
from .image_generation import DiffusionMain, diff_main
from .storyteller import create_story_side
from .tts import create_tts_side

def create_sidebar():
    with gr.Sidebar(elem_classes="sidebar-container") as sidebar:
        tab_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab = create_tab_side()

        chatbot_sidebar = chat_main.create_chatbot_side()
        diff_sidebar = diff_main.create_diffusion_side()

        storyteller_side, storytelling_model_type_dropdown, storytelling_model_dropdown, storytelling_api_key_text, storytelling_lora_dropdown = create_story_side()                  
        tts_side, tts_model_type_dropdown, tts_model_dropdown = create_tts_side()
        with gr.Column() as translate_side:
            with gr.Row(elem_classes="model-container"):
                with gr.Column():
                    gr.Markdown("### Under Construction")
                    
    return sidebar, tab_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chatbot_sidebar, diff_sidebar, storyteller_side, storytelling_model_type_dropdown, storytelling_model_dropdown, storytelling_api_key_text, storytelling_lora_dropdown, tts_side, tts_model_type_dropdown, tts_model_dropdown, translate_side