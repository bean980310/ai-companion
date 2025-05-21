import gradio as gr
from ...start_app import ui_component, app_state
from ...common.translations import _
from ...common.utils import get_all_loras

def create_chatbot_side_session_container():
    with gr.Row(elem_classes="session-container"):
        with gr.Column():
            gr.Markdown("### Chat Session")
            session_select_dropdown = gr.Dropdown(
                label="세션 선택",
                choices=[],  # 앱 시작 시 혹은 별도의 로직으로 세션 목록을 채움
                value=None,
                interactive=True,
                container=False,
                scale=8,
                elem_classes="session-dropdown"
            )
            chat_title_box=gr.Textbox(
                value="",
                interactive=False
            )
            add_session_icon_btn = gr.Button("📝", elem_classes="icon-button", scale=1, variant="secondary")
            delete_session_icon_btn = gr.Button("🗑️", elem_classes="icon-button-delete", scale=1, variant="stop")
            
    
    ui_component.session_select_dropdown = session_select_dropdown
    ui_component.chat_title_box = chat_title_box
    ui_component.add_session_icon_btn = add_session_icon_btn
    ui_component.delete_session_icon_btn = delete_session_icon_btn
    
    return session_select_dropdown, chat_title_box, add_session_icon_btn, delete_session_icon_btn

def create_chatbot_side_model_container():
    with gr.Row(elem_classes="model-container"):
        with gr.Column():
            gr.Markdown("### Model Selection")
            model_type_dropdown = gr.Radio(
                label=_("model_type_label"),
                choices=app_state.llm_type_choices,
                value=app_state.llm_type_choices[0],
                elem_classes="model-dropdown"
            )
            model_dropdown = gr.Dropdown(
                label=_("model_select_label"),
                choices=app_state.initial_choices,
                value=app_state.initial_choices[0] if len(app_state.initial_choices) > 0 else None,
                elem_classes="model-dropdown"
            )
            api_key_text = gr.Textbox(
                label=_("api_key_label"),
                placeholder="sk-...",
                visible=False,
                elem_classes="api-key-input"
            )
            lora_dropdown = gr.Dropdown(
                label="LoRA 모델 선택",
                choices=get_all_loras(),
                value="None",
                interactive=True,
                visible=False,
                elem_classes="model-dropdown"
            )
            
    ui_component.model_type_dropdown = model_type_dropdown
    ui_component.model_dropdown = model_dropdown
    ui_component.api_key_text = api_key_text
    ui_component.lora_dropdown = lora_dropdown
    
    return model_type_dropdown, model_dropdown, api_key_text, lora_dropdown