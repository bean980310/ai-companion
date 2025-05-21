import gradio as gr
from dataclasses import dataclass

@dataclass
class AppState:
    speech_manager_state: gr.State = None
    
    session_id = None
    loaded_history = None
    session_dropdown = None
    last_character = None
    last_preset = None
    system_message = None
    session_label = None
    
    last_sid_state: gr.State = None
    history_state: gr.State = None
    last_character_state: gr.State = None
    session_list_state: gr.State = None
    overwrite_state: gr.State = None
    
    custom_model_path_state: gr.State = None
    session_id_state: gr.State = None
    selected_device_state: gr.State = None
    character_state: gr.State = None
    system_message_state: gr.State = None
    
    seed_state: gr.State = None
    temperature_state: gr.State = None
    top_k_state: gr.State = None
    top_p_state: gr.State = None
    repetition_penalty_state: gr.State = None
    selected_language_state: gr.State = None
    
    reset_confirmation: gr.State = None
    reset_all_confirmation: gr.State = None
    
    max_diffusion_lora_rows = None
    stored_image: gr.State = None
    stored_image_inpaint: gr.State = None
    
    initial_choices = None
    llm_type_choices = None
    
    diffusion_choices = None
    diffusion_type_choices = None
    diffusion_lora_choices = None
    vae_choices = None
    diffusion_refiner_choices = None
    diffusion_refiner_type_choices = None
    
    tts_choices = None
    
app_state = AppState()