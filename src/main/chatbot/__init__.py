import gradio as gr

from .chatbot import get_speech_manager
from .chatbot import Chatbot
from .chatbot import update_system_message_and_profile
from .chatbot import create_reset_confirm_modal
from .chatbot import create_delete_session_modal, get_allowed_llm_models
from .component import create_chatbot_side_session_container, create_chatbot_side_model_container, create_chat_container_main_panel, create_chat_container_side_panel
from ...common.translations import _

from ... import os_name, arch
from ...start_app import app_state, ui_component

chat_bot = Chatbot()

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

def apply_session_immediately(chosen_sid):
        """
        메인탭에서 세션이 선택되면 바로 main_tab.apply_session을 호출해 세션 적용.
        """
        return chat_bot.apply_session(chosen_sid)
    
def create_chat_container():
    with gr.Column(elem_classes='tab-container') as chat_container:
        with gr.Row(elem_classes="model-container"):
            gr.Markdown("### Chat")
        with gr.Row(elem_classes="chat-interface"):
            system_message_accordion, system_message_box, chatbot, msg, multimodal_msg, image_input = create_chat_container_main_panel()
            profile_image, character_dropdown, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, preset_dropdown, change_preset_button, reset_btn, reset_all_btn = create_chat_container_side_panel()
                            
        with gr.Row(elem_classes="status-bar"):
            status_text = gr.Markdown("Ready", elem_id="status_text")
            image_info = gr.Markdown("", visible=False)
            session_select_info = gr.Markdown(_('select_session_info'))
            
    ui_component.status_text = status_text
    ui_component.image_info = image_info
    ui_component.session_select_info = session_select_info
            
    return [chat_container,
            system_message_accordion, system_message_box, chatbot, msg, multimodal_msg, image_input,
            profile_image, character_dropdown, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, preset_dropdown, change_preset_button, reset_btn, reset_all_btn,
            status_text, image_info, session_select_info]
    
def create_chat_container_2():
    with gr.Tab("Chat", elem_classes='tab-container') as chat_tab:
        with gr.Row(elem_classes="model-container"):
            gr.Markdown("### Chat")
        with gr.Row(elem_classes="chat-interface"):
            system_message_accordion, system_message_box, chatbot, msg, multimodal_msg, image_input = create_chat_container_main_panel()
            profile_image, character_dropdown, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, preset_dropdown, change_preset_button, reset_btn, reset_all_btn = create_chat_container_side_panel()
                            
        with gr.Row(elem_classes="status-bar"):
            status_text = gr.Markdown("Ready", elem_id="status_text")
            image_info = gr.Markdown("", visible=False)
            session_select_info = gr.Markdown(_('select_session_info'))
            
    ui_component.status_text = status_text
    ui_component.image_info = image_info
    ui_component.session_select_info = session_select_info
            
    return [chat_tab,
            system_message_accordion, system_message_box, chatbot, msg, multimodal_msg, image_input,
            profile_image, character_dropdown, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, preset_dropdown, change_preset_button, reset_btn, reset_all_btn,
            status_text, image_info, session_select_info]




__all__ = ["Chatbot", "get_speech_manager", "update_system_message_and_profile", "create_reset_confirm_modal", "create_delete_session_modal", "get_allowed_llm_models", "share_allowed_llm_models"]