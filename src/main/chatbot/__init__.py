import gradio as gr

from .chatbot import Chatbot
from .component import ChatbotComponent
from ...common.translations import _

from ... import os_name, arch
from ...start_app import app_state, ui_component
from dataclasses import dataclass

chat_bot = Chatbot()
chat_component = ChatbotComponent()

@dataclass
class ChatbotMain:
    sidebar: gr.Column = None
    session: ChatbotComponent = None
    model: ChatbotComponent = None

    container: gr.Column = None
    main_panel: ChatbotComponent = None
    side_panel: ChatbotComponent = None
    status_bar: ChatbotComponent = None

    @staticmethod
    def share_allowed_llm_models():
        initial_choices, llm_type_choices = chat_bot.get_allowed_llm_models(os_name, arch)
        
        app_state.initial_choices = initial_choices
        app_state.llm_type_choices = llm_type_choices
        
        # return initial_choices, llm_type_choices

    @classmethod
    def create_chatbot_side(cls):
        with gr.Column() as chatbot_side:
            chat_side_session = chat_component.create_chatbot_side_session_container()
            chat_side_model = chat_component.create_chatbot_side_model_container()

        return cls(sidebar=chatbot_side, session=chat_side_session, model=chat_side_model)

    @staticmethod
    def apply_session_immediately(chosen_sid):
            """
            메인탭에서 세션이 선택되면 바로 main_tab.apply_session을 호출해 세션 적용.
            """
            return chat_bot.apply_session(chosen_sid)

    @classmethod
    def create_chat_container(cls):
        with gr.Column(elem_classes='tab-container') as chat_container:
            with gr.Row(elem_classes="model-container"):
                gr.Markdown("### Chat")
            with gr.Row(elem_classes="chat-interface"):
                chat_body_main = chat_component.create_chat_container_main_panel()
                chat_body_side = chat_component.create_chat_container_side_panel()

            with gr.Row(elem_classes="status-bar"):
                chat_body_status = chat_component.create_chat_container_status_bar()

        return cls(container=chat_container, main_panel=chat_body_main, side_panel=chat_body_side, status_bar=chat_body_status)

    @classmethod
    def create_chat_container_2(cls):
        with gr.Tab("Chat", elem_classes='tab-container') as chat_tab:
            with gr.Row(elem_classes="model-container"):
                gr.Markdown("### Chat")
            with gr.Row(elem_classes="chat-interface"):
                chat_body_main = chat_component.create_chat_container_main_panel()
                chat_body_side = chat_component.create_chat_container_side_panel()

            with gr.Row(elem_classes="status-bar"):
                chat_body_status = chat_component.create_chat_container_status_bar()

        return cls(container=chat_tab, main_panel=chat_body_main, side_panel=chat_body_side, status_bar=chat_body_status)


chat_main = ChatbotMain()

__all__ = ["Chatbot"]