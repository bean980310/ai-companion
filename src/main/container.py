import gradio as gr
from .chatbot import chat_main
from .image_generation import diff_main
from .storyteller import create_story_container, create_story_container_2
from .translator import create_translate_container
from ..tabs.download_tab import create_download_tab
from ..common.translations import _

def create_llm_intergrated_container():
    with gr.Tabs(elem_classes='tabs') as llm_intergrated_container:
        chat_container = chat_main.create_chat_container_2()
        story_container, storytelling_input, storytelling_btn, storytelling_output, story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider = create_story_container_2()
        
    return llm_intergrated_container, chat_container, story_container, storytelling_input, storytelling_btn, storytelling_output, story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider
    

def create_body_container():
    with gr.Row(elem_classes='tabs'):
        chat_container = chat_main.create_chat_container()
        diff_container = diff_main.create_diffusion_container()
        
        story_container, storytelling_input, storytelling_btn, storytelling_output, story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider = create_story_container()
                    
        with gr.Column(elem_classes='tab-container') as tts_container:
            with gr.Row(elem_classes="chat-interface"):
                gr.Markdown("# Coming Soon!")
                
        translate_container = create_translate_container()
        # download_container = create_download_tab()

    return chat_container, diff_container, story_container, storytelling_input, storytelling_btn, storytelling_output, story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider, tts_container, translate_container