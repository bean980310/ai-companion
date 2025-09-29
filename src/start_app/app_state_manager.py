import gradio as gr
from dataclasses import dataclass
from ..characters.persona_speech_manager import PersonaSpeechManager
from typing import Any
@dataclass
class AppState:
    speech_manager_state: gr.State | PersonaSpeechManager | None = None

    session_id: str | None = None
    loaded_history: list[dict[str, str | Any]] | None = None
    session_dropdown: gr.State | None = None
    last_character: gr.State | str | None = None
    last_preset: gr.State | str | None = None
    system_message: gr.State | str | None = None
    session_label: gr.State | str | None = None

    last_sid_state: gr.State | None = None
    history_state: gr.State | None = None
    last_character_state: gr.State | None = None
    session_list_state: gr.State | None = None
    overwrite_state: gr.State | None = None

    custom_model_path_state: gr.State | None = None
    session_id_state: gr.State | None = None
    selected_device_state: gr.State | None = None
    character_state: gr.State | None = None
    system_message_state: gr.State | None = None

    seed_state: gr.State | None = None
    max_length_state: gr.State | None = None
    temperature_state: gr.State | None = None
    top_k_state: gr.State | None = None
    top_p_state: gr.State | None = None
    repetition_penalty_state: gr.State | None = None
    selected_language_state: gr.State | None = None

    reset_confirmation: gr.State | None = None
    reset_all_confirmation: gr.State | None = None

    max_diffusion_lora_rows: int | None = None
    stored_image: gr.State | None = None
    stored_image_inpaint: gr.State | None = None

    initial_choices: list[str] | None = None
    llm_type_choices: list[str] | None = None

    diffusion_choices: list[str] | None = None
    diffusion_type_choices: list[str] | None = None
    diffusion_lora_choices: list[str] | None = None
    vae_choices: list[str] | None = None
    diffusion_refiner_choices: list[str] | None = None
    diffusion_refiner_type_choices: list[str] | None = None

    tts_choices: list[str] | None = None

app_state = AppState()