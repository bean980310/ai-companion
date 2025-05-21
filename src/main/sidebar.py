import gradio as gr

from ..start_app import create_tab_side
from .chatbot import create_chatbot_side
from .image_generation import create_diffusion_side
from .storyteller import create_story_side
from .tts import create_tts_side

def create_sidebar():
    with gr.Sidebar(elem_classes="sidebar-container") as sidebar:
        tab_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab = create_tab_side()
        chatbot_side, session_select_dropdown, chat_title_box, add_session_icon_btn, delete_session_icon_btn, model_type_dropdown, model_dropdown, api_key_text, lora_dropdown = create_chatbot_side()
        diffusion_side, diffusion_model_type_dropdown, diffusion_model_dropdown, diffusion_api_key_text, diffusion_refiner_model_dropdown, diffusion_refiner_start, diffusion_with_refiner_image_to_image_start, diffusion_lora_multiselect, diffusion_lora_text_encoder_sliders, diffusion_lora_unet_sliders = create_diffusion_side()       
        storyteller_side, storytelling_model_type_dropdown, storytelling_model_dropdown, storytelling_api_key_text, storytelling_lora_dropdown = create_story_side()                  
        tts_side, tts_model_type_dropdown, tts_model_dropdown = create_tts_side()
        with gr.Column() as translate_side:
            with gr.Row(elem_classes="model-container"):
                with gr.Column():
                    gr.Markdown("### Under Construction")
                    
    return [
        sidebar, tab_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab,
        chatbot_side, session_select_dropdown, chat_title_box, add_session_icon_btn, delete_session_icon_btn, model_type_dropdown, model_dropdown, api_key_text, lora_dropdown,
        diffusion_side, diffusion_model_type_dropdown, diffusion_model_dropdown, diffusion_api_key_text, diffusion_refiner_model_dropdown, diffusion_refiner_start, diffusion_with_refiner_image_to_image_start, diffusion_lora_multiselect, diffusion_lora_text_encoder_sliders, diffusion_lora_unet_sliders,
        storyteller_side, storytelling_model_type_dropdown, storytelling_model_dropdown, storytelling_api_key_text, storytelling_lora_dropdown,
        tts_side, tts_model_type_dropdown, tts_model_dropdown, translate_side
        ]