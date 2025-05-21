import gradio as gr
from ...common.translations import _
from ...common.utils import get_all_loras
from ...start_app import app_state, ui_component

def create_story_side():
    with gr.Column() as storyteller_side:
        with gr.Row(elem_classes="model-container"):
            with gr.Column():
                gr.Markdown("### Model Selection")
                storytelling_model_type_dropdown = gr.Radio(
                    label=_("model_type_label"),
                    choices=app_state.llm_type_choices,
                    value=app_state.llm_type_choices[0],
                    elem_classes="model-dropdown"
                )
                storytelling_model_dropdown = gr.Dropdown(
                    label=_("model_select_label"),
                    choices=app_state.initial_choices,
                    value=app_state.initial_choices[0] if len(app_state.initial_choices) > 0 else None,
                    elem_classes="model-dropdown"
                )
                storytelling_api_key_text = gr.Textbox(
                    label=_("api_key_label"),
                    placeholder="sk-...",
                    visible=False,
                    elem_classes="api-key-input"
                )
                storytelling_lora_dropdown = gr.Dropdown(
                    label="LoRA 모델 선택",
                    choices=get_all_loras(),
                    value="None",
                    interactive=True,
                    visible=False,
                    elem_classes="model-dropdown"
                )
                
    ui_component.storytelling_model_type_dropdown = storytelling_model_type_dropdown
    ui_component.storytelling_model_dropdown = storytelling_model_dropdown
    ui_component.storytelling_api_key_text = storytelling_api_key_text
    ui_component.storytelling_lora_dropdown = storytelling_lora_dropdown
    
    return storyteller_side, storytelling_model_type_dropdown, storytelling_model_dropdown, storytelling_api_key_text, storytelling_lora_dropdown