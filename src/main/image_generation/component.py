import gradio as gr
from ...common.translations import _
from ...start_app import app_state, ui_component

def create_diffusion_side_model_container():
    with gr.Row(elem_classes="model-container"):
        with gr.Column():
            gr.Markdown("### Model Selection")
            diffusion_model_type_dropdown = gr.Radio(
                label=_("model_type_label"),
                choices=app_state.diffusion_type_choices,
                value=app_state.diffusion_type_choices[0],
                elem_classes="model-dropdown"
            )
            diffusion_model_dropdown = gr.Dropdown(
                label=_("model_select_label"),
                choices=app_state.diffusion_choices,
                value=app_state.diffusion_choices[0] if len(app_state.diffusion_choices) > 0 else None,
                elem_classes="model-dropdown"
            )
            diffusion_api_key_text = gr.Textbox(
                label=_("api_key_label"),
                placeholder="sk-...",
                visible=False,
                elem_classes="api-key-input"
            )
            
    ui_component.diffusion_model_type_dropdown = diffusion_model_type_dropdown
    ui_component.diffusion_model_dropdown = diffusion_model_dropdown
    ui_component.diffusion_api_key_text = diffusion_api_key_text
    
    return diffusion_model_type_dropdown, diffusion_model_dropdown, diffusion_api_key_text

def create_diffusion_side_refiner_model_container():
    with gr.Row(elem_classes="model-container"):
        with gr.Column():
            gr.Markdown("### Refiner Model Selection")
            diffusion_refiner_model_dropdown = gr.Dropdown(
                label=_("refiner_model_select_label"),
                choices=app_state.diffusion_refiner_choices,
                value=app_state.diffusion_refiner_choices[0] if len(app_state.diffusion_refiner_choices) > 0 else None,
                elem_classes="model-dropdown"
            )
            diffusion_refiner_start = gr.Slider(
                label="Refiner Start Step",
                minimum=1,
                maximum=50,
                step=1,
                value=20,
                visible=False
            )
            diffusion_with_refiner_image_to_image_start = gr.Slider(
                label="Image to Image Start Step",
                minimum=1,
                maximum=50,
                step=1,
                value=20,
                visible=False
            )
            
    ui_component.diffusion_refiner_model_dropdown = diffusion_refiner_model_dropdown
    ui_component.diffusion_refiner_start = diffusion_refiner_start
    ui_component.diffusion_with_refiner_image_to_image_start = diffusion_with_refiner_image_to_image_start
    
    return diffusion_refiner_model_dropdown, diffusion_refiner_start, diffusion_with_refiner_image_to_image_start


def create_diffusion_side_lora_container():
    with gr.Row(elem_classes="model-container"):
        with gr.Accordion("LoRA Settings", open=False, elem_classes="accordion-container"):
            diffusion_lora_multiselect=gr.Dropdown(
                label="Select LoRA Models",
                choices=app_state.diffusion_lora_choices,
                value=[],
                interactive=True,
                multiselect=True,
                info="Select LoRA models to apply to the diffusion model.",
                elem_classes="model-dropdown"
            )
            diffusion_lora_text_encoder_sliders=[]
            diffusion_lora_unet_sliders=[]
            for i in range(app_state.max_diffusion_lora_rows):
                text_encoder_slider=gr.Slider(
                    label=f"LoRA {i+1} - Text Encoder Weight",
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.01,
                    value=1.0,
                    visible=False,
                    interactive=True
                )
                unet_slider = gr.Slider(
                    label=f"LoRA {i+1} - U-Net Weight",
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.01,
                    value=1.0,
                    visible=False,
                    interactive=True
                )
                diffusion_lora_text_encoder_sliders.append(text_encoder_slider)
                diffusion_lora_unet_sliders.append(unet_slider)
            diffusion_lora_slider_rows=[]
            for te, unet in zip(diffusion_lora_text_encoder_sliders, diffusion_lora_unet_sliders):
                diffusion_lora_slider_rows.append(gr.Row([te, unet]))
            for row in diffusion_lora_slider_rows:
                row
                
    ui_component.diffusion_lora_multiselect = diffusion_lora_multiselect
    ui_component.diffusion_lora_text_encoder_sliders = diffusion_lora_text_encoder_sliders
    ui_component.diffusion_lora_unet_sliders = diffusion_lora_unet_sliders
    
    return diffusion_lora_multiselect, diffusion_lora_text_encoder_sliders, diffusion_lora_unet_sliders