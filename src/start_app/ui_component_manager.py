from __future__ import annotations

import gradio as gr
from dataclasses import dataclass
from typing import Optional


@dataclass
class UIComponent:
    head: Optional[gr.Row] = None
    header_row: Optional[gr.Row] = None
    title: Optional[gr.Markdown] = None
    subtitle: Optional[gr.Markdown] = None
    settings_button: Optional[gr.Button] = None
    language_dropdown: Optional[gr.Dropdown] = None
    navbar: Optional[gr.Navbar] = None

    session_select_dropdown: Optional[gr.Dropdown] = None
    chat_title_box: Optional[gr.Textbox] = None
    add_session_icon_btn: Optional[gr.Button] = None
    delete_session_icon_btn: Optional[gr.Button] = None

    # Session list components
    session_rows: Optional[list] = None
    session_buttons: Optional[list] = None
    session_delete_buttons: Optional[list] = None
    selected_session_id: Optional[gr.Textbox] = None

    model_provider_dropdown: Optional[gr.Dropdown] = None
    model_type_dropdown: Optional[gr.Radio] = None
    model_dropdown: Optional[gr.Dropdown] = None
    api_key_text: Optional[gr.Textbox] = None
    lora_dropdown: Optional[gr.Dropdown] = None
    refresh_button: Optional[gr.Button] = None
    clear_all_btn: Optional[gr.Button] = None

    diffusion_model_provider_dropdown: Optional[str | gr.Dropdown] = None
    diffusion_model_type_dropdown: Optional[gr.Radio] = None
    diffusion_model_dropdown: Optional[gr.Dropdown] = None
    diffusion_api_key_text: Optional[gr.Textbox] = None

    diffusion_refiner_model_dropdown: Optional[gr.Dropdown] = None
    diffusion_refiner_start: Optional[gr.Slider] = None
    diffusion_with_refiner_image_to_image_start: Optional[gr.Slider] = None

    diffusion_lora_multiselect: Optional[gr.Dropdown] = None
    diffusion_lora_text_encoder_sliders: Optional[list] = None
    diffusion_lora_unet_sliders: Optional[list] = None

    storytelling_model_provider_dropdown: Optional[gr.Dropdown] = None
    storytelling_model_type_dropdown: Optional[gr.Radio] = None
    storytelling_model_dropdown: Optional[gr.Dropdown] = None
    storytelling_api_key_text: Optional[gr.Textbox] = None
    storytelling_lora_dropdown: Optional[gr.Dropdown] = None
    storytelling_refresh_button: Optional[gr.Button] = None
    story_clear_all_btn: Optional[gr.Button] = None

    tts_model_type_dropdown: Optional[gr.Radio] = None
    tts_model_dropdown: Optional[gr.Dropdown] = None

    system_message_accordion: Optional[gr.Accordion] = None
    system_message_box: Optional[gr.Textbox] = None
    chatbot: Optional[gr.Chatbot] = None
    msg: Optional[gr.Textbox] = None
    multimodal_msg: Optional[gr.MultimodalTextbox] = None

    profile_image: Optional[gr.Image] = None
    character_dropdown: Optional[gr.Dropdown] = None

    text_advanced_settings: Optional[gr.Accordion] = None
    text_seed_input: Optional[gr.Number] = None
    text_max_length_input: Optional[gr.Slider] = None
    text_temperature_slider: Optional[gr.Slider] = None
    text_top_k_slider: Optional[gr.Slider] = None
    text_top_p_slider: Optional[gr.Slider] = None
    text_repetition_penalty_slider: Optional[gr.Slider] = None
    text_enable_thinking_checkbox: Optional[gr.Checkbox] = None
    text_preset_dropdown: Optional[gr.Dropdown] = None
    text_change_preset_button: Optional[gr.Button] = None
    text_reset_btn: Optional[gr.Button] = None
    text_reset_all_btn: Optional[gr.Button] = None

    status_text: Optional[gr.Markdown] = None
    image_info: Optional[gr.Markdown] = None
    session_select_info: Optional[gr.Markdown] = None

    image_to_image_mode: Optional[gr.Radio] = None
    image_to_image_input: Optional[gr.Image] = None
    image_inpaint_input: Optional[gr.Image] = None
    image_inpaint_masking: Optional[gr.ImageMask] = None

    blur_radius_slider: Optional[gr.Slider] = None
    blur_expansion_radius_slider: Optional[gr.Slider] = None
    denoise_strength_slider: Optional[gr.Slider] = None

    positive_prompt_input: Optional[gr.TextArea] = None
    negative_prompt_input: Optional[gr.TextArea] = None
    style_dropdown: Optional[gr.Dropdown] = None

    width_slider: Optional[gr.Slider] = None
    height_slider: Optional[gr.Slider] = None

    generation_step_slider: Optional[gr.Slider] = None
    random_prompt_btn: Optional[gr.Button] = None
    generate_btn: Optional[gr.Button] = None

    gallery: Optional[gr.Gallery] = None

    diffusion_advanced_settings: Optional[gr.Accordion] = None
    sampler_dropdown: Optional[gr.Dropdown] = None
    scheduler_dropdown: Optional[gr.Dropdown] = None
    cfg_scale_slider: Optional[gr.Slider] = None
    diffusion_seed_input: Optional[gr.Number] = None
    random_seed_checkbox: Optional[gr.Checkbox] = None
    vae_dropdown: Optional[gr.Dropdown] = None
    clip_skip_slider: Optional[gr.Slider] = None
    enable_clip_skip_checkbox: Optional[gr.Checkbox] = None
    clip_g_checkbox: Optional[gr.Checkbox] = None
    batch_size_input: Optional[gr.Number] = None
    batch_count_input: Optional[gr.Number] = None

    image_history: Optional[gr.Dataframe] = None

    storytelling_input: Optional[gr.Textbox] = None
    storytelling_btn: Optional[gr.Button] = None
    storytelling_output: Optional[gr.Textbox] = None

    storyteller_seed_input: Optional[gr.Number] = None
    storyteller_temperature_slider: Optional[gr.Slider] = None
    storyteller_top_k_slider: Optional[gr.Slider] = None
    storyteller_top_p_slider: Optional[gr.Slider] = None
    storyteller_repetition_penalty_slider: Optional[gr.Slider] = None


ui_component = UIComponent()
