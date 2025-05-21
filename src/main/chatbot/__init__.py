import gradio as gr

from .chatbot import get_speech_manager
from .chatbot import Chatbot
from .chatbot import update_system_message_and_profile
from .chatbot import create_reset_confirm_modal
from .chatbot import create_delete_session_modal, get_allowed_llm_models
from .component import create_chatbot_side_session_container, create_chatbot_side_model_container

from ... import os_name, arch
from ...start_app import app_state

def share_allowed_llm_models():
    initial_choices, llm_type_choices = get_allowed_llm_models(os_name, arch)
    
    app_state.initial_choices = initial_choices
    app_state.llm_type_choices = llm_type_choices
    
    return initial_choices, llm_type_choices


def create_chatbot_side():
    with gr.Column() as chatbot_side:
        session_select_dropdown, chat_title_box, add_session_icon_btn, delete_session_icon_btn = create_chatbot_side_session_container()
        model_type_dropdown, model_dropdown, api_key_text, lora_dropdown = create_chatbot_side_model_container()
        
    return chatbot_side, session_select_dropdown, chat_title_box, add_session_icon_btn, delete_session_icon_btn, model_type_dropdown, model_dropdown, api_key_text, lora_dropdown



__all__ = ["Chatbot", "get_speech_manager", "update_system_message_and_profile", "create_reset_confirm_modal", "create_delete_session_modal", "get_allowed_llm_models", "share_allowed_llm_models"]