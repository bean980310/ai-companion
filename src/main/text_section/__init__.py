import gradio as gr

from ..chatbot import (
    get_speech_manager, 
    Chatbot, 
    update_system_message_and_profile,
    create_reset_confirm_modal,
    create_delete_session_modal, 
    get_allowed_llm_models,
    create_chatbot_side_session_container, 
    create_chatbot_side_model_container, 
    create_chat_container_main_panel, 
    create_chat_container_side_panel,
    share_allowed_llm_models,
    create_chatbot_side,
    apply_session_immediately
)

from ..storyteller import (
    create_story_side, 
    create_story_container_main_panel, 
    create_story_container_side_panel
)

from ...common.translations import _