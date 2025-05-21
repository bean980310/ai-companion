import gradio as gr
from dataclasses import dataclass

@dataclass
class UIComponent:
    title: gr.Markdown = None
    settings_button: gr.Button = None
    language_dropdown: gr.Dropdown = None
    
    session_select_dropdown: gr.Dropdown = None
    chat_title_box: gr.Textbox = None
    add_session_icon_btn: gr.Button = None
    delete_session_icon_btn: gr.Button = None
    
    model_type_dropdown: gr.Radio = None
    model_dropdown: gr.Dropdown = None
    api_key_text: gr.Textbox = None
    lora_dropdown: gr.Dropdown = None
    
    diffusion_model_type_dropdown: gr.Radio = None
    diffusion_model_dropdown: gr.Dropdown = None
    diffusion_api_key_text: gr.Textbox = None
    
    diffusion_refiner_model_dropdown: gr.Dropdown = None
    diffusion_refiner_start: gr.Slider = None
    diffusion_with_refiner_image_to_image_start: gr.Slider = None
    
    diffusion_lora_multiselect: gr.Dropdown = None
    diffusion_lora_text_encoder_sliders: list = None
    diffusion_lora_unet_sliders: list = None
    
    storytelling_model_type_dropdown: gr.Radio = None
    storytelling_model_dropdown: gr.Dropdown = None
    storytelling_api_key_text: gr.Textbox = None
    storytelling_lora_dropdown: gr.Dropdown = None
    
    tts_model_type_dropdown: gr.Radio = None
    tts_model_dropdown: gr.Dropdown = None
    
ui_component = UIComponent()