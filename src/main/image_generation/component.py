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

def create_diffusion_container_image_to_image_panel():
    with gr.Row(elem_classes="model-container"):
        with gr.Accordion("Image to Image", open=False, elem_classes="accordion-container"):
            image_to_image_mode = gr.Radio(
                label="Image to Image Mode",
                choices=["None", "Image to Image", "Inpaint", "Inpaint Upload"],
                value="None",
                elem_classes="model-dropdown"
            )
            with gr.Column():
                with gr.Row():
                    image_to_image_input = gr.Image(
                        label="Image to Image",
                        type="filepath",
                        sources="upload",
                        format="png",
                        visible=False
                    )
                    image_inpaint_input = gr.Image(
                        label="Image Inpaint",
                        type="filepath",
                        sources="upload",
                        format="png",
                        visible=False
                    )
                    image_inpaint_masking = gr.ImageMask(
                        label="Image Inpaint Mask",
                        type="filepath",
                        sources="upload",
                        format="png",
                        visible=False
                    )
                            
            blur_radius_slider = gr.Slider(
                label="Blur Radius",
                minimum=0,
                maximum=10,
                step=0.5,
                value=5,
                visible=False
            )
            blur_expansion_radius_slider = gr.Slider(
                label="Blur Expansion Radius",
                minimum=0,
                maximum=100,
                step=1,
                value=1,
                visible=False
            )
            denoise_strength_slider = gr.Slider(
                label="Denoise Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.5,
                visible=False
            )
            
    ui_component.image_to_image_mode = image_to_image_mode
    ui_component.image_to_image_input = image_to_image_input
    ui_component.image_inpaint_input = image_inpaint_input
    ui_component.image_inpaint_masking = image_inpaint_masking
    ui_component.blur_radius_slider = blur_radius_slider
    ui_component.blur_expansion_radius_slider = blur_expansion_radius_slider
    ui_component.denoise_strength_slider = denoise_strength_slider
    
    return image_to_image_mode, image_to_image_input, image_inpaint_input, image_inpaint_masking, blur_radius_slider, blur_expansion_radius_slider, denoise_strength_slider

def create_diffusion_container_main_panel():
    with gr.Column(scale=7):
        positive_prompt_input = gr.TextArea(
            label="Positive Prompt",
            placeholder="Enter positive prompt...",
            elem_classes="message-input"
        )
        negative_prompt_input = gr.TextArea(
            label="Negative Prompt",
            placeholder="Enter negative prompt...",
            elem_classes="message-input"
        )
                        
        with gr.Row():
            style_dropdown = gr.Dropdown(
                label="Style",
                choices=["Photographic", "Digital Art", "Oil Painting", "Watercolor"],
                value="Photographic"
            )
                            
        with gr.Row():
            width_slider = gr.Slider(
                label="Width",
                minimum=128,
                maximum=2048,
                step=64,
                value=512
            )
            height_slider = gr.Slider(
                label="Height",
                minimum=128,
                maximum=2048,
                step=64,
                value=512
            )
                        
        with gr.Row():
            generation_step_slider=gr.Slider(
                label="Generation Steps",
                minimum=1,
                maximum=50,
                step=1,
                value=20
            )
                        
        with gr.Row():
            random_prompt_btn = gr.Button("🎲 Random Prompt", variant="secondary", elem_classes="random-button")
            generate_btn = gr.Button("🎨 Generate", variant="primary", elem_classes="send-button-alt")
                        
        gallery = gr.Gallery(
            label="Generated Images",
            format="png",
            columns=2,
            rows=2
        )
        
    ui_component.positive_prompt_input = positive_prompt_input
    ui_component.negative_prompt_input = negative_prompt_input
    ui_component.style_dropdown = style_dropdown
    ui_component.width_slider = width_slider
    ui_component.height_slider = height_slider
    ui_component.generation_step_slider = generation_step_slider
    ui_component.random_prompt_btn = random_prompt_btn
    ui_component.generate_btn = generate_btn
    ui_component.gallery = gallery
        
    return positive_prompt_input, negative_prompt_input, style_dropdown, width_slider, height_slider, generation_step_slider, random_prompt_btn, generate_btn, gallery