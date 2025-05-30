import gradio as gr
from .chatbot import create_chat_container, create_chat_container_2
from .image_generation import create_diffusion_container
from .storyteller import create_story_container, create_story_container_2
from .translator import create_translate_container
from ..tabs.download_tab import create_download_tab
from ..common.translations import _


def create_llm_intergrated_container():
    with gr.Tabs(elem_classes='tabs') as llm_intergrated_container:
        chat_container, system_message_accordion, system_message_box, chatbot, msg, multimodal_msg, image_input, profile_image, character_dropdown, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, preset_dropdown, change_preset_button, reset_btn, reset_all_btn, status_text, image_info, session_select_info = create_chat_container_2()
        story_container, storytelling_input, storytelling_btn, storytelling_output, story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider = create_story_container_2()
        
    return llm_intergrated_container, chat_container, system_message_accordion, system_message_box, chatbot, msg, multimodal_msg, image_input, profile_image, character_dropdown, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, preset_dropdown, change_preset_button, reset_btn, reset_all_btn, status_text, image_info, session_select_info, story_container, storytelling_input, storytelling_btn, storytelling_output, story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider
    

def create_body_container():
    with gr.Row(elem_classes='tabs'):
        chat_container, system_message_accordion, system_message_box, chatbot, msg, multimodal_msg, image_input, profile_image, character_dropdown, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, preset_dropdown, change_preset_button, reset_btn, reset_all_btn, status_text, image_info, session_select_info = create_chat_container()

        diffusion_container, image_to_image_mode, image_to_image_input, image_inpaint_input, image_inpaint_masking, blur_radius_slider, blur_expansion_radius_slider, denoise_strength_slider, positive_prompt_input, negative_prompt_input, style_dropdown, width_slider, height_slider, generation_step_slider, random_prompt_btn, generate_btn, gallery, diff_adv_setting, sampler_dropdown, scheduler_dropdown, cfg_scale_slider, diffusion_seed_input, random_seed_checkbox, vae_dropdown, clip_skip_slider, enable_clip_skip_checkbox, clip_g_checkbox, batch_size_input, batch_count_input, image_history = create_diffusion_container()
        
        story_container, storytelling_input, storytelling_btn, storytelling_output, story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider = create_story_container()
                    
        with gr.Column(elem_classes='tab-container') as tts_container:
            with gr.Row(elem_classes="chat-interface"):
                gr.Markdown("# Coming Soon!")
                
        translate_container = create_translate_container()
        download_container = create_download_tab()

    return chat_container, system_message_accordion, system_message_box, chatbot, msg, multimodal_msg, image_input, profile_image, character_dropdown, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, preset_dropdown, change_preset_button, reset_btn, reset_all_btn, status_text, image_info, session_select_info, diffusion_container, image_to_image_mode, image_to_image_input, image_inpaint_input, image_inpaint_masking, blur_radius_slider, blur_expansion_radius_slider, denoise_strength_slider, positive_prompt_input, negative_prompt_input, style_dropdown, width_slider, height_slider, generation_step_slider, random_prompt_btn, generate_btn, gallery, diff_adv_setting, sampler_dropdown, scheduler_dropdown, cfg_scale_slider, diffusion_seed_input, random_seed_checkbox, vae_dropdown, clip_skip_slider, enable_clip_skip_checkbox, clip_g_checkbox, batch_size_input, batch_count_input, image_history, story_container, storytelling_input, storytelling_btn, storytelling_output, story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider, tts_container, translate_container, download_container