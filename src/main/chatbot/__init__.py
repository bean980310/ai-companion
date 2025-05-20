from .chatbot import get_speech_manager
from .chatbot import Chatbot
from .chatbot import update_system_message_and_profile
from .chatbot import create_reset_confirm_modal
from .chatbot import create_delete_session_modal, get_allowed_llm_models

from ... import os_name, arch
from ...start_app import app_state

def share_allowed_llm_models():
    initial_choices, llm_type_choices = get_allowed_llm_models(os_name, arch)
    
    app_state.initial_choices = initial_choices
    app_state.llm_type_choices = llm_type_choices
    
    return initial_choices, llm_type_choices



__all__ = ["Chatbot", "get_speech_manager", "update_system_message_and_profile", "create_reset_confirm_modal", "create_delete_session_modal", "get_allowed_llm_models", "share_allowed_llm_models"]