import gradio as gr

from .image_generation import (
    generate_images_wrapper, 
    update_diffusion_model_list,
    toggle_diffusion_api_key_visibility,
    get_allowed_diffusion_models,
)

from .component import DiffusionComponent
from dataclasses import dataclass

from ...common.utils import get_diffusion_loras, get_diffusion_vae

from ... import os_name, arch
from ...start_app import app_state, ui_component

diff_component = DiffusionComponent()

@dataclass
class DiffusionMain:
    sidebar: gr.Column = None
    model: DiffusionComponent = None
    refiner: DiffusionComponent = None
    lora: DiffusionComponent = None

    container: gr.Column = None
    image_panel: DiffusionComponent = None
    main_panel: DiffusionComponent = None
    side_panel: DiffusionComponent = None
    history_panel: DiffusionComponent = None

    @staticmethod
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
        
        # return diffusion_choices, diffusion_type_choices, diffusion_lora_choices, vae_choices, diffusion_refiner_choices, diffusion_refiner_type_choices

    @classmethod
    def create_diffusion_side(cls):
        with gr.Column() as diffusion_side:
            diff_side_model = diff_component.create_diffusion_side_model_container()
            diff_side_refiner = diff_component.create_diffusion_side_refiner_model_container()
            diff_side_lora = diff_component.create_diffusion_side_lora_container()

            return cls(sidebar=diffusion_side, model=diff_side_model, refiner=diff_side_refiner, lora=diff_side_lora)

    @classmethod
    def create_diffusion_container(cls):
        with gr.Column(elem_classes='tab-container') as diffusion_container:
            with gr.Row(elem_classes="model-container"):
                gr.Markdown("### Image Generation")
            image_to_image_panel = diff_component.create_diffusion_container_image_to_image_panel()

            with gr.Row(elem_classes="chat-interface"):   
                diff_body_main = diff_component.create_diffusion_container_main_panel()

                diff_body_side = diff_component.create_diffusion_container_side_panel()
                
            with gr.Accordion("History", open=False, elem_classes="accordion-container"):
                diff_body_history = diff_component.create_diffusion_container_history_panel()

        return cls(container=diffusion_container, image_panel=image_to_image_panel, main_panel=diff_body_main, side_panel=diff_body_side, history_panel=diff_body_history)

diff_main = DiffusionMain()

__all__ = [
    'generate_images_wrapper', 
    'update_diffusion_model_list', 
    'toggle_diffusion_api_key_visibility',
    'get_allowed_diffusion_models']