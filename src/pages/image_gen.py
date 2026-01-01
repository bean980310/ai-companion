import gradio as gr
from transformers.pipelines import image_classification
from src.main.image_generation import diff_main, image_gen, diff_component
from src.start_app import app_state, ui_component
from comfy_sdk import ComfyUI
from src.common.translations import translation_manager, _
from src.common_blocks import create_page_header, get_language_code
from typing import Any, List, Sequence, Callable
from PIL import Image
import numpy as np
import random

client = ComfyUI(port=8000)

with gr.Blocks() as demo:
    # 0. Page-Specific State Registration
    def register_image_gen_state():
        app_state.max_diffusion_lora_rows = 10
        app_state.stored_image = gr.State()
        app_state.stored_image_inpaint = gr.State()

        # Load Model Lists
        diff_main.share_allowed_diffusion_models()

    register_image_gen_state()

    # 1. Page Header with Language Selector
    page_header = create_page_header(page_title_key="image_gen_title")
    language_dropdown = page_header.language_dropdown

    # 2. UI Construction
    with gr.Sidebar():
        # Replicating create_diffusion_side logic within Sidebar
        diff_side_model = diff_component.create_diffusion_side_model_container()
        diff_side_refiner = diff_component.create_diffusion_side_refiner_model_container()
        diff_side_lora = diff_component.create_diffusion_side_lora_container()
        
    # Main Container
    diff_container_obj = diff_main.create_diffusion_container()
    
    # 2. Extract Components for Event Wiring
    
    # Sidebar components
    diffusion_model_provider_dropdown = diff_side_model.model_provider_dropdown
    diffusion_model_type_dropdown = diff_side_model.model_type_dropdown
    diffusion_model_dropdown = diff_side_model.model_dropdown
    diffusion_api_key_text = diff_side_model.api_key_text

    diffusion_refiner_row = diff_side_refiner.refiner_row
    diffusion_refiner_model_dropdown = diff_side_refiner.refiner_model_dropdown
    diffusion_refiner_start = diff_side_refiner.refiner_start
    diffusion_with_refiner_image_to_image_start = diff_side_refiner.with_refiner_image_to_image_start

    diffusion_lora_row = diff_side_lora.lora_row
    diffusion_lora_multiselect = diff_side_lora.lora_multiselect
    diffusion_lora_text_encoder_sliders = diff_side_lora.lora_text_encoder_sliders
    diffusion_lora_unet_sliders = diff_side_lora.lora_unet_sliders

    vae_row = diff_container_obj.side_panel.vae_row
    
    # Main Container components
    image_to_image_mode = diff_container_obj.image_panel.image_to_image_mode
    image_to_image_input = diff_container_obj.image_panel.image_to_image_input
    image_inpaint_input = diff_container_obj.image_panel.image_inpaint_input
    image_inpaint_masking = diff_container_obj.image_panel.image_inpaint_masking

    blur_radius_slider = diff_container_obj.image_panel.blur_radius_slider
    blur_expansion_radius_slider = diff_container_obj.image_panel.blur_expansion_radius_slider
    denoise_strength_slider = diff_container_obj.image_panel.denoise_strength_slider

    positive_prompt_input = diff_container_obj.main_panel.positive_prompt_input
    negative_prompt_input = diff_container_obj.main_panel.negative_prompt_input
    style_dropdown = diff_container_obj.main_panel.style_dropdown

    width_slider = diff_container_obj.main_panel.width_slider
    height_slider = diff_container_obj.main_panel.height_slider
    generation_step_slider = diff_container_obj.main_panel.generation_step_slider
    random_prompt_btn = diff_container_obj.main_panel.random_prompt_btn
    generate_btn = diff_container_obj.main_panel.generate_btn
    gallery = diff_container_obj.main_panel.gallery

    sampler_dropdown = diff_container_obj.side_panel.sampler_dropdown
    scheduler_dropdown = diff_container_obj.side_panel.scheduler_dropdown
    cfg_scale_slider = diff_container_obj.side_panel.cfg_scale_slider

    diffusion_seed_input = diff_container_obj.side_panel.seed_input
    random_seed_checkbox = diff_container_obj.side_panel.random_seed_checkbox
    vae_dropdown = diff_container_obj.side_panel.vae_dropdown

    clip_skip_slider = diff_container_obj.side_panel.clip_skip_slider
    enable_clip_skip_checkbox = diff_container_obj.side_panel.enable_clip_skip_checkbox
    clip_g_checkbox = diff_container_obj.side_panel.clip_g_checkbox

    batch_size_input = diff_container_obj.side_panel.batch_size_input
    batch_count_input = diff_container_obj.side_panel.batch_count_input

    image_history = diff_container_obj.history_panel.image_history

    # 3. Event Wiring (Copied from src/main/__init__.py)
    
    # gr.on(
    #     triggers=[diffusion_model_dropdown.change, demo.load],
    #     fn=lambda selected_model: (
    #         image_gen.toggle_diffusion_api_key_visibility(selected_model)
    #     ),
    #     inputs=[diffusion_model_dropdown],
    #     outputs=[diffusion_api_key_text]
    # )

    gr.on(
        triggers=[diffusion_model_provider_dropdown.change, diffusion_model_type_dropdown.change, demo.load],
        fn=image_gen.update_diffusion_model_list,
        inputs=[diffusion_model_provider_dropdown, diffusion_model_type_dropdown],
        outputs=[diffusion_model_type_dropdown, diffusion_model_dropdown]
    )

    gr.on(
        triggers=[diffusion_model_provider_dropdown.change, demo.load],
        fn=image_gen.toggle_diffusion_api_key_visibility,
        inputs=[diffusion_model_provider_dropdown],
        outputs=[diffusion_api_key_text]
    ).then(
        fn=image_gen.toggle_diffusion_lora_visible,
        inputs=[diffusion_model_provider_dropdown],
        outputs=[diffusion_lora_row, diffusion_lora_multiselect]
    ).then(
        fn=image_gen.toggle_diffusion_vae_visible,
        inputs=[diffusion_model_provider_dropdown],
        outputs=[vae_row, vae_dropdown]
    ).then(
        fn=image_gen.toggle_diffusion_refiner_visible,
        inputs=[diffusion_model_provider_dropdown],
        outputs=[diffusion_refiner_row, diffusion_refiner_model_dropdown]
    )
    
    diffusion_refiner_model_dropdown.change(
        fn=lambda model: (
            image_gen.toggle_refiner_start_step(model)
            ),
        inputs=[diffusion_refiner_model_dropdown],
        outputs=[diffusion_refiner_start]
    ).then(
        fn=image_gen.toggle_diffusion_with_refiner_image_to_image_start,
        inputs=[diffusion_refiner_model_dropdown, image_to_image_mode],
        outputs=[diffusion_with_refiner_image_to_image_start]
    )
    
    def process_uploaded_image_for_inpaint(image: str | Image.Image | Any):
        # This function appeared in the source but wasn't clearly used in the final wiring in __init__.py 
        # (it had process_uploaded_image_inpaint with @image_inpaint_masking.apply)
        # But let's check the wiring below.
        pass

    # Wiring for stored_image_inpaint
    @image_inpaint_masking.apply(inputs=[image_inpaint_input, image_inpaint_masking], outputs=app_state.stored_image_inpaint)
    def process_uploaded_image_inpaint(original_image: str | Image.Image | Any, mask_image: list[str | Image.Image | Any]):
        mask = client.upload_mask(original_image, mask_image)
        return mask

    def copy_image_for_inpaint(image_input, image) -> gr.update:
        import cv2
        # Ensure image_input path is valid or process it
        if image_input is None: return gr.update(value=image)
        
        # This logic was:
        # im = cv2.imread(image_input) ...
        # But wait, image_input coming from upload might be a filepath.
        # Let's trust the logic from __init__.py if it worked there.
        # Re-implementing briefly:
        try:
            im = cv2.imread(image_input)
            height, width, channels = im.shape[:3]
            image['background'] = image_input
            image['layers'][0] = np.zeros((height, width, 4), dtype=np.uint8)
        except Exception as e:
            print(f"Error in copy_image_for_inpaint: {e}")
        return gr.update(value=image)
    
    image_to_image_input.change(
        fn=image_gen.process_uploaded_image,
        inputs=image_to_image_input,
        outputs=app_state.stored_image
    )
    
    image_inpaint_input.upload(
        fn=image_gen.process_uploaded_image,
        inputs=[image_inpaint_input],
        outputs=app_state.stored_image
    ).then(
        fn=copy_image_for_inpaint,
        inputs=[image_inpaint_input, image_inpaint_masking],
        outputs=image_inpaint_masking
    ).then(
        fn=image_gen.toggle_image_inpaint_mask_interactive,
        inputs=image_inpaint_input,
        outputs=image_inpaint_masking
    )
    
    image_to_image_mode.change(
        fn=lambda mode: (
            image_gen.toggle_image_to_image_input(mode),
            image_gen.toggle_image_inpaint_input(mode),
            image_gen.toggle_image_inpaint_mask(mode),
            image_gen.toggle_denoise_strength_dropdown(mode)
            ),
        inputs=[image_to_image_mode],
        outputs=[image_to_image_input,
                 image_inpaint_input,
                 image_inpaint_masking, 
                 denoise_strength_slider]
    ).then(
        fn=image_gen.toggle_diffusion_with_refiner_image_to_image_start,
        inputs=[diffusion_refiner_model_dropdown, image_to_image_mode],
        outputs=[diffusion_with_refiner_image_to_image_start]
    ).then(
        fn=image_gen.toggle_blur_radius_slider,
        inputs=[image_to_image_mode],
        outputs=[blur_radius_slider, blur_expansion_radius_slider]
    )
    
    def generate_diffusion_lora_weight_sliders(selected_loras: List[str]):
        updates=[]
        for i in range(app_state.max_diffusion_lora_rows):
            if i < len(selected_loras):
                lora_name = selected_loras[i]
                text_update = gr.update(visible=True, label=f"{lora_name} - Text Encoder Weight")
                unet_update = gr.update(visible=True, label=f"{lora_name} - U-Net Weight")
            else:
                text_update = gr.update(visible=False)
                unet_update = gr.update(visible=False)
            updates.append(text_update)
            updates.append(unet_update)
        return updates

    @random_prompt_btn.click(outputs=[positive_prompt_input])
    def get_random_prompt():
        prompts = [
            "A serene mountain landscape at sunset",
            "A futuristic cityscape with flying cars",
            "A mystical forest with glowing mushrooms"
        ]
        return random.choice(prompts)
        
    diffusion_lora_slider_outputs = []
    for te_slider, unet_slider in zip(diffusion_lora_text_encoder_sliders, diffusion_lora_unet_sliders):
        diffusion_lora_slider_outputs.extend([te_slider, unet_slider])
        
    diffusion_lora_multiselect.change(
        fn=generate_diffusion_lora_weight_sliders,
        inputs=[diffusion_lora_multiselect],
        outputs=diffusion_lora_slider_outputs
    )

    generate_btn.click(
        fn=image_gen.generate_images_wrapper,
        inputs=[
            positive_prompt_input,       # Positive Prompt
            negative_prompt_input,       # Negative Prompt
            style_dropdown,              # Style
            generation_step_slider,
            diffusion_with_refiner_image_to_image_start,
            diffusion_refiner_start,
            width_slider,                # Width
            height_slider,               # Height
            diffusion_model_dropdown,    # 선택한 이미지 생성 모델 (체크포인트 파일명 또는 diffusers model id)
            diffusion_refiner_model_dropdown, 
            diffusion_model_type_dropdown,  # "checkpoint" 또는 "diffusers" 선택 (라디오 버튼 등)
            diffusion_lora_multiselect,  # 선택한 LoRA 모델 리스트
            vae_dropdown,                # 선택한 VAE 모델
            clip_skip_slider,
            enable_clip_skip_checkbox,
            clip_g_checkbox,
            sampler_dropdown,
            scheduler_dropdown,
            batch_size_input,
            batch_count_input,
            cfg_scale_slider,
            diffusion_seed_input,
            random_seed_checkbox,
            image_to_image_mode, 
            app_state.stored_image,
            app_state.stored_image_inpaint,
            denoise_strength_slider,
            blur_radius_slider,
            blur_expansion_radius_slider,
            diffusion_api_key_text,
            *diffusion_lora_text_encoder_sliders,
            *diffusion_lora_unet_sliders
        ],
        outputs=[gallery, image_history]
    )

    # Language Change Event
    def on_image_gen_language_change(selected_lang: str):
        lang_code = get_language_code(selected_lang)
        translation_manager.set_language(lang_code)
        return [
            gr.update(value=f"## {_('image_gen_title')}"),
            gr.update(label=_('language_select'), info=_('language_info'))
        ]

    language_dropdown.change(
        fn=on_image_gen_language_change,
        inputs=[language_dropdown],
        outputs=[page_header.title, language_dropdown]
    )

if __name__ == "__main__":
    demo.launch()
