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

def create_story_container_main_panel():
    with gr.Column(scale=7):
        storytelling_input = gr.Textbox(
            label="Input",
            placeholder="Enter your message...",
            lines=10,
            elem_classes="message-input",
        )
        storytelling_btn = gr.Button("Storytelling", variant="primary", elem_classes="send-button-alt")
        storytelling_output = gr.Textbox(
            label="Output",
            lines=10,
            elem_classes="message-output"
        )
        
    ui_component.storytelling_input = storytelling_input
    ui_component.storytelling_btn = storytelling_btn
    ui_component.storytelling_output = storytelling_output
    
    return storytelling_input, storytelling_btn, storytelling_output

def create_story_container_side_panel():
    with gr.Column(scale=3, elem_classes="side-panel"):
        with gr.Accordion(_("advanced_setting"), open=False, elem_classes="accordion-container") as story_adv_setting:
            storyteller_seed_input = gr.Number(
                label=_("seed_label"),
                value=42,
                precision=0,
                step=1,
                interactive=True,
                info=_("seed_info"),
                elem_classes="seed-input"
            )
            storyteller_temperature_slider=gr.Slider(
                label=_("temperature_label"),
                minimum=0.0,
                maximum=1.0,
                value=0.6,
                step=0.1,
                interactive=True
            )
            storyteller_top_k_slider=gr.Slider(
                label=_("top_k_label"),
                minimum=0,
                maximum=100,
                value=20,
                step=1,
                interactive=True
            )
            storyteller_top_p_slider=gr.Slider(
                label=_("top_p_label"),
                minimum=0.0,
                maximum=1.0,
                value=0.9,
                step=0.1,
                interactive=True
            )
            storyteller_repetition_penalty_slider=gr.Slider(
                label=_("repetition_penalty_label"),
                minimum=0.0,
                maximum=2.0,
                value=1.1,
                step=0.1,
                interactive=True
            )
            
    ui_component.storyteller_seed_input = storyteller_seed_input
    ui_component.storyteller_temperature_slider = storyteller_temperature_slider
    ui_component.storyteller_top_k_slider = storyteller_top_k_slider
    ui_component.storyteller_top_p_slider = storyteller_top_p_slider
    ui_component.storyteller_repetition_penalty_slider = storyteller_repetition_penalty_slider
    
    return story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider

def create_story_container():
    with gr.Column(elem_classes='tab-container') as story_container:
        with gr.Row(elem_classes="model-container"):
            gr.Markdown("### Storyteller")
        with gr.Row(elem_classes="model-container"):
            gr.Markdown("# Under Construction")
        with gr.Row(elem_classes="chat-interface"):
            storytelling_input, storytelling_btn, storytelling_output = create_story_container_main_panel()
            story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider = create_story_container_side_panel()
            
    return [
        story_container, 
        storytelling_input, storytelling_btn, storytelling_output, 
        story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider
    ]
    
def create_story_container_2():
    with gr.Tab("Storyteller", elem_classes='tab-container') as story_tab:
        with gr.Row(elem_classes="model-container"):
            gr.Markdown("### Storyteller")
        with gr.Row(elem_classes="model-container"):
            gr.Markdown("# Under Construction")
        with gr.Row(elem_classes="chat-interface"):
            storytelling_input, storytelling_btn, storytelling_output = create_story_container_main_panel()
            story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider = create_story_container_side_panel()
            
    return [
        story_tab, 
        storytelling_input, storytelling_btn, storytelling_output, 
        story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider
    ]