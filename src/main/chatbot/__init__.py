import gradio as gr

from .chatbot import Chatbot
from .component import ChatbotComponent
# from ...common.translations import _
# from translations import i18n as _

from ... import os_name, arch
from ...start_app import app_state, ui_component
from dataclasses import dataclass

chat_bot = Chatbot()
chat_component = ChatbotComponent()

@dataclass
class ChatbotMain:
    initial_choices: list[str] = None
    llm_type_choices: list[str] = None

    sidebar: gr.Column = None
    session: ChatbotComponent = None
    model: ChatbotComponent = None

    container: gr.Column = None
    main_panel: ChatbotComponent = None
    side_panel: ChatbotComponent = None
    status_bar: ChatbotComponent = None

    @classmethod
    def share_allowed_llm_models(cls):
        initial_choices, llm_type_choices = chat_bot.get_allowed_llm_models()
        
        app_state.initial_choices = initial_choices
        app_state.llm_type_choices = llm_type_choices

        return cls(initial_choices=initial_choices, llm_type_choices=llm_type_choices)
        
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
            
            # Create status bar components (not rendered)
            chat_body_status = chat_component.create_chat_container_status_bar(render=False)

            with gr.Row(elem_classes="chat-interface"):
                col_main = gr.Column(scale=7)
                col_side = gr.Column(scale=3, elem_classes="side-panel")
            
            with col_side:
                chat_body_side = chat_component.create_chat_container_side_panel()
            
            additional_inputs = [
                app_state.session_id_state,
                ui_component.character_dropdown,
                app_state.selected_language_state,
                ui_component.model_dropdown,
                ui_component.model_provider_dropdown,
                ui_component.lora_dropdown,
                app_state.custom_model_path_state,
                ui_component.api_key_text,
                app_state.selected_device_state,
                app_state.seed_state,
                app_state.max_length_state,
                app_state.temperature_state,
                app_state.top_k_state,
                app_state.top_p_state,
                app_state.repetition_penalty_state,
                app_state.enable_thinking_state,
            ]

            with col_main:
                chat_body_main = chat_component.create_chat_container_main_panel(
                    chat_wrapper_fn=chat_bot.chat_wrapper,
                    additional_inputs=additional_inputs
                )

            with gr.Row(elem_classes="status-bar"):
                chat_body_status.status_text.render()
                chat_body_status.image_info.render()
                chat_body_status.session_select_info.render()

        return cls(container=chat_container, main_panel=chat_body_main, side_panel=chat_body_side, status_bar=chat_body_status)

    @classmethod
    def create_chat_container_2(cls):
        with gr.Tab("Chat", elem_classes='tab-container') as chat_tab:
            with gr.Row(elem_classes="model-container"):
                gr.Markdown("### Chat")
            
            # Create status bar components (not rendered)
            chat_body_status = chat_component.create_chat_container_status_bar()

            with gr.Row(elem_classes="chat-interface"):
                col_main = gr.Column(scale=7)
                col_side = gr.Column(scale=3, elem_classes="side-panel")
            
            with col_side:
                chat_body_side = chat_component.create_chat_container_side_panel()
            
            additional_inputs = [
                app_state.session_id_state,
                ui_component.character_dropdown,
                app_state.selected_language_state,
                ui_component.model_dropdown,
                ui_component.model_provider_dropdown,
                ui_component.lora_dropdown,
                app_state.custom_model_path_state,
                ui_component.api_key_text,
                app_state.selected_device_state,
                app_state.seed_state,
                app_state.max_length_state,
                app_state.temperature_state,
                app_state.top_k_state,
                app_state.top_p_state,
                app_state.repetition_penalty_state,
                app_state.enable_thinking_state,
            ]

            with col_main:
                chat_body_main = chat_component.create_chat_container_main_panel(
                    chat_wrapper_fn=Chatbot().chat_wrapper,
                    additional_inputs=additional_inputs
                )

            with gr.Row(elem_classes="status-bar"):
                chat_body_status.status_text.render()
                chat_body_status.image_info.render()
                chat_body_status.session_select_info.render()

        return cls(container=chat_tab, main_panel=chat_body_main, side_panel=chat_body_side, status_bar=chat_body_status)


chat_main = ChatbotMain()

__all__ = ["Chatbot"]