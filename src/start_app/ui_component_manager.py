import gradio as gr
from dataclasses import dataclass

@dataclass
class UIComponent:
    head: gr.Row = None
    title: gr.Markdown = None
    settings_button: gr.Button = None
    language_dropdown: gr.Dropdown = None
    navbar: gr.Navbar = None
    
    session_select_dropdown: gr.Dropdown = None
    chat_title_box: gr.Textbox = None
    add_session_icon_btn: gr.Button = None
    delete_session_icon_btn: gr.Button = None

    # Session list components
    session_rows: list = None
    session_buttons: list = None
    session_delete_buttons: list = None
    selected_session_id: gr.Textbox = None
    
    model_provider_dropdown: gr.Dropdown = None
    model_type_dropdown: gr.Radio = None
    model_dropdown: gr.Dropdown = None
    api_key_text: gr.Textbox = None
    lora_dropdown: gr.Dropdown = None
    refresh_button: gr.Button = None
    clear_all_btn: gr.Button = None
    
    diffusion_model_provider_dropdown: gr.Dropdown = None
    diffusion_model_type_dropdown: gr.Radio = None
    diffusion_model_dropdown: gr.Dropdown = None
    diffusion_api_key_text: gr.Textbox = None
    
    diffusion_refiner_model_dropdown: gr.Dropdown = None
    diffusion_refiner_start: gr.Slider = None
    diffusion_with_refiner_image_to_image_start: gr.Slider = None
    
    diffusion_lora_multiselect: gr.Dropdown = None
    diffusion_lora_text_encoder_sliders: list = None
    diffusion_lora_unet_sliders: list = None
    
    storytelling_model_type_dropdown: gr.Radio = None
    storytelling_model_dropdown: gr.Dropdown = None
    storytelling_api_key_text: gr.Textbox = None
    storytelling_lora_dropdown: gr.Dropdown = None
    
    tts_model_type_dropdown: gr.Radio = None
    tts_model_dropdown: gr.Dropdown = None

    system_message_accordion: gr.Accordion = None
    system_message_box: gr.Textbox = None
    chatbot: gr.Chatbot = None
    msg: gr.Textbox = None
    multimodal_msg: gr.MultimodalTextbox = None
    
    profile_image: gr.Image = None
    character_dropdown: gr.Dropdown = None

    text_advanced_settings: gr.Accordion = None
    text_seed_input: gr.Number = None
    text_max_length_input: gr.Slider = None
    text_temperature_slider: gr.Slider = None
    text_top_k_slider: gr.Slider = None
    text_top_p_slider: gr.Slider = None
    text_repetition_penalty_slider: gr.Slider = None
    text_enable_thinking_checkbox: gr.Checkbox = None
    text_preset_dropdown: gr.Dropdown = None
    text_change_preset_button: gr.Button = None
    text_reset_btn: gr.Button = None
    text_reset_all_btn: gr.Button = None
    
    status_text: gr.Markdown = None
    image_info: gr.Markdown = None
    session_select_info: gr.Markdown = None
    
    image_to_image_mode: gr.Radio = None
    image_to_image_input: gr.Image = None
    image_inpaint_input: gr.Image = None
    image_inpaint_masking: gr.ImageMask = None
    
    blur_radius_slider: gr.Slider = None
    blur_expansion_radius_slider: gr.Slider = None
    denoise_strength_slider: gr.Slider = None
    
    positive_prompt_input: gr.TextArea = None
    negative_prompt_input: gr.TextArea = None
    style_dropdown: gr.Dropdown = None
    
    width_slider: gr.Slider = None
    height_slider: gr.Slider = None
    
    generation_step_slider: gr.Slider = None
    random_prompt_btn: gr.Button = None
    generate_btn: gr.Button = None
    
    gallery: gr.Gallery = None

    diffusion_advanced_settings: gr.Accordion = None
    sampler_dropdown: gr.Dropdown = None
    scheduler_dropdown: gr.Dropdown = None
    cfg_scale_slider: gr.Slider = None
    diffusion_seed_input: gr.Number = None
    random_seed_checkbox: gr.Checkbox = None
    vae_dropdown: gr.Dropdown = None
    clip_skip_slider: gr.Slider = None
    enable_clip_skip_checkbox: gr.Checkbox = None
    clip_g_checkbox: gr.Checkbox = None
    batch_size_input: gr.Number = None
    batch_count_input: gr.Number = None
    
    image_history: gr.Dataframe = None
    
    storytelling_input: gr.Textbox = None
    storytelling_btn: gr.Button = None
    storytelling_output: gr.Textbox = None
    
    storyteller_seed_input: gr.Number = None
    storyteller_temperature_slider: gr.Slider = None
    storyteller_top_k_slider: gr.Slider = None
    storyteller_top_p_slider: gr.Slider = None
    storyteller_repetition_penalty_slider: gr.Slider = None
    
ui_component = UIComponent()