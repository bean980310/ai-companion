import gradio as gr
from .chatbot import create_chat_container

def create_body_container():
    with gr.Row(elem_classes='tabs'):
        chat_container, system_message_box, chatbot, msg, multimodal_msg, image_input, profile_image, character_dropdown, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, preset_dropdown, change_preset_button, reset_btn, reset_all_btn, status_text, image_info, session_select_info = create_chat_container()
        
    return chat_container, system_message_box, chatbot, msg, multimodal_msg, image_input, profile_image, character_dropdown, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, preset_dropdown, change_preset_button, reset_btn, reset_all_btn, status_text, image_info, session_select_info