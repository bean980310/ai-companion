import gradio as gr
from .chatbot import create_chat_container
from .image_generation import create_diffusion_container
from .translator import create_translate_container
from ..tabs.download_tab import create_download_tab
from ..common.translations import _

def create_body_container():
    with gr.Row(elem_classes='tabs'):
        chat_container, system_message_box, chatbot, msg, multimodal_msg, image_input, profile_image, character_dropdown, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, preset_dropdown, change_preset_button, reset_btn, reset_all_btn, status_text, image_info, session_select_info = create_chat_container()

        diffusion_container, image_to_image_mode, image_to_image_input, image_inpaint_input, image_inpaint_masking, blur_radius_slider, blur_expansion_radius_slider, denoise_strength_slider, positive_prompt_input, negative_prompt_input, style_dropdown, width_slider, height_slider, generation_step_slider, random_prompt_btn, generate_btn, gallery, diff_adv_setting, sampler_dropdown, scheduler_dropdown, cfg_scale_slider, diffusion_seed_input, random_seed_checkbox, vae_dropdown, clip_skip_slider, enable_clip_skip_checkbox, clip_g_checkbox, batch_size_input, batch_count_input, image_history = create_diffusion_container()
        
        with gr.Column(elem_classes='tab-container') as story_container:
            with gr.Row(elem_classes="model-container"):
                gr.Markdown("### Storyteller")
            with gr.Row(elem_classes="model-container"):
                gr.Markdown("# Under Construction")
            with gr.Row(elem_classes="chat-interface"):
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
                    
        with gr.Column(elem_classes='tab-container') as tts_container:
            with gr.Row(elem_classes="chat-interface"):
                gr.Markdown("# Coming Soon!")
                
        translate_container = create_translate_container()
        download_container = create_download_tab()

    return chat_container, system_message_box, chatbot, msg, multimodal_msg, image_input, profile_image, character_dropdown, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, preset_dropdown, change_preset_button, reset_btn, reset_all_btn, status_text, image_info, session_select_info, diffusion_container, image_to_image_mode, image_to_image_input, image_inpaint_input, image_inpaint_masking, blur_radius_slider, blur_expansion_radius_slider, denoise_strength_slider, positive_prompt_input, negative_prompt_input, style_dropdown, width_slider, height_slider, generation_step_slider, random_prompt_btn, generate_btn, gallery, diff_adv_setting, sampler_dropdown, scheduler_dropdown, cfg_scale_slider, diffusion_seed_input, random_seed_checkbox, vae_dropdown, clip_skip_slider, enable_clip_skip_checkbox, clip_g_checkbox, batch_size_input, batch_count_input, image_history, story_container, storytelling_input, storytelling_btn, storytelling_output, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider, tts_container, translate_container, download_container