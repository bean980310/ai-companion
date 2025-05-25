import gradio as gr

from .image_generation import (
    generate_images_wrapper, 
    update_diffusion_model_list,
    toggle_diffusion_api_key_visibility,
    get_allowed_diffusion_models,
)

from .component import create_diffusion_side_model_container, create_diffusion_side_refiner_model_container, create_diffusion_side_lora_container, create_diffusion_container_image_to_image_panel, create_diffusion_container_main_panel

from ...common.utils import get_diffusion_loras, get_diffusion_vae

from ... import os_name, arch
from ...start_app import app_state

__all__ = [
    'generate_images_wrapper', 
    'update_diffusion_model_list', 
    'toggle_diffusion_api_key_visibility',
    'get_allowed_diffusion_models']

def share_allowed_diffusion_models():
    diffusion_choices, diffusion_type_choices = get_allowed_diffusion_models(os_name, arch)
    
    diffusion_lora_choices = get_diffusion_loras()
    diffusion_lora_choices = list(dict.fromkeys(diffusion_lora_choices))
    diffusion_lora_choices = sorted(diffusion_lora_choices)
    
    vae_choices = get_diffusion_vae()
    
    diffusion_refiner_choices, diffusion_refiner_type_choices = get_allowed_diffusion_models(os_name, arch)
    
    if "None" not in diffusion_refiner_choices:
        diffusion_refiner_choices.insert(0, "None")
    
    if "Default" not in vae_choices:
        vae_choices.insert(0, "Default")
    
    app_state.diffusion_choices = diffusion_choices
    app_state.diffusion_type_choices = diffusion_type_choices
    app_state.diffusion_lora_choices = diffusion_lora_choices
    app_state.vae_choices = vae_choices
    app_state.diffusion_refiner_choices = diffusion_refiner_choices
    app_state.diffusion_refiner_type_choices = diffusion_refiner_type_choices
    
    return diffusion_choices, diffusion_type_choices, diffusion_lora_choices, vae_choices, diffusion_refiner_choices, diffusion_refiner_type_choices

def create_diffusion_side():
    with gr.Column() as diffusion_side:
        diffusion_model_type_dropdown, diffusion_model_dropdown, diffusion_api_key_text = create_diffusion_side_model_container()
        diffusion_refiner_model_dropdown, diffusion_refiner_start, diffusion_with_refiner_image_to_image_start = create_diffusion_side_refiner_model_container()
        diffusion_lora_multiselect, diffusion_lora_text_encoder_sliders, diffusion_lora_unet_sliders = create_diffusion_side_lora_container()
        
    return diffusion_side, diffusion_model_type_dropdown, diffusion_model_dropdown, diffusion_api_key_text, diffusion_refiner_model_dropdown, diffusion_refiner_start, diffusion_with_refiner_image_to_image_start, diffusion_lora_multiselect, diffusion_lora_text_encoder_sliders, diffusion_lora_unet_sliders

def create_diffusion_container():
    with gr.Column(elem_classes='tab-container') as diffusion_container:
        with gr.Row(elem_classes="model-container"):
            gr.Markdown("### Image Generation")
        image_to_image_mode, image_to_image_input, image_inpaint_input, image_inpaint_masking, blur_radius_slider, blur_expansion_radius_slider, denoise_strength_slider = create_diffusion_container_image_to_image_panel()
        
        with gr.Row(elem_classes="chat-interface"):   
            positive_prompt_input, negative_prompt_input, style_dropdown, width_slider, height_slider, generation_step_slider, random_prompt_btn, generate_btn, gallery = create_diffusion_container_main_panel()

            with gr.Column(scale=3, elem_classes="side-panel"):
                with gr.Accordion("Advanced Settings", open=False, elem_classes="accordion-container") as diff_adv_setting:
                    sampler_dropdown = gr.Dropdown(
                        label="Sampler",
                        choices=["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim"],
                        value="euler"
                    )
                    scheduler_dropdown = gr.Dropdown(
                        label="Scheduler",
                        choices=["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "linear_quadratic", "lm_optimal"],  # 실제 옵션에 맞게 변경
                        value="normal"
                    )
                    cfg_scale_slider = gr.Slider(
                        label="CFG Scale",
                        minimum=1,
                        maximum=20,
                        step=0.5,
                        value=7.5
                    )
                    with gr.Row():
                        diffusion_seed_input = gr.Number(
                            label="Seed",
                            value=42,
                            precision=0
                        )
                        random_seed_checkbox = gr.Checkbox(
                            label="Random Seed",
                            value=True
                        )
                    with gr.Row():
                        vae_dropdown=gr.Dropdown(
                            label="Select VAE Model",
                            choices=app_state.vae_choices,
                            value="Default",
                            interactive=True,
                            info="Select VAE model to apply to the diffusion model.",
                            elem_classes="model-dropdown"
                        )
                    with gr.Row():
                        clip_skip_slider = gr.Slider(
                            label="Clip Skip",
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=2
                        )
                        enable_clip_skip_checkbox = gr.Checkbox(
                            label="Enable Custom Clip Skip",
                            value=False
                        )
                        clip_g_checkbox = gr.Checkbox(
                            label="Enable Clip G",
                            value=False
                        )
                    with gr.Row():
                        batch_size_input = gr.Number(
                            label="Batch Size",
                            value=1,
                            precision=0
                        )
                        batch_count_input = gr.Number(
                            label="Batch Count",
                            value=1,
                            precision=0
                        )
        with gr.Accordion("History", open=False, elem_classes="accordion-container"):
            image_history = gr.Dataframe(
                headers=["Prompt", "Negative Prompt", "Steps", "Model", "Sampler", "Scheduler", "CFG Scale", "Seed", "Width", "Height"],
                label="Generation History",
                col_count=(10, "dynamic"),
                wrap=True,
                datatype=["str", "str", "str", "str", "str", "str", "str", "str", "str", "str"]
            )
    
    return [diffusion_container, 
            image_to_image_mode, image_to_image_input, image_inpaint_input, image_inpaint_masking, blur_radius_slider, blur_expansion_radius_slider, denoise_strength_slider,
            positive_prompt_input, negative_prompt_input, style_dropdown, width_slider, height_slider, generation_step_slider, random_prompt_btn, generate_btn, gallery]