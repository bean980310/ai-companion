import gradio as gr
from ...start_app import app_state, ui_component
from ...common.translations import _

def create_tts_side():
    with gr.Column() as tts_side:
        with gr.Row(elem_classes="model-container"):
            with gr.Column():
                gr.Markdown("### Model Selection")
                tts_model_type_dropdown = gr.Radio(
                    label=_("model_type_label"),
                    choices=["all", "api", "vits"],
                    value="all",
                    elem_classes="model-dropdown"
                )
                tts_model_dropdown = gr.Dropdown(
                    label=_("model_select_label"),
                    choices=app_state.tts_choices,
                    value=app_state.tts_choices[0] if len(app_state.tts_choices) > 0 else "Put Your Models",
                    elem_classes="model-dropdown"
                )
                
    
    ui_component.tts_model_type_dropdown = tts_model_type_dropdown
    ui_component.tts_model_dropdown = tts_model_dropdown
    
    return tts_side, tts_model_type_dropdown, tts_model_dropdown