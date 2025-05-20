# app.py

import warnings
import platform
import importlib
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import gradio as gr
import sqlite3
from src.common.database import (
    initialize_database,
    add_system_preset,
    delete_system_preset,
    ensure_demo_session,
    load_chat_from_db, 
    load_system_presets, 
    get_existing_sessions, 
    get_preset_choices,
    insert_default_presets,
    update_system_message_in_db,
    update_last_character_in_db)
from src.models.models import default_device
from src.common.cache import models_cache
from src.common.translations import translation_manager, _, TranslationManager
from src.characters.persona_speech_manager import PersonaSpeechManager
from src.characters import handle_character_change
from src.common.args import parse_args
from src.common.default_language import default_language
from src.common.tmp_dir import TMP_DIR
from src.common.character_info import characters
from src.common.utils import (
    get_all_loras, 
    get_diffusion_loras, 
    get_diffusion_vae, 
    detect_platform
)
import numpy as np

from src.common.html import css, show_confetti

from presets import (
    AI_ASSISTANT_PRESET, 
    SD_IMAGE_GENERATOR_PRESET, 
    MINAMI_ASUKA_PRESET, 
    MAKOTONO_AOI_PRESET, 
    AINO_KOITO_PRESET, 
    ARIA_PRINCESS_FATE_PRESET, 
    ARIA_PRINCE_FATE_PRESET,
    WANG_MEI_LING_PRESET,
    MISTY_LANE_PRESET
    )

from src.models import api_models, transformers_local, gguf_local, mlx_local, diffusion_api_models, diffusers_local, checkpoints_local, tts_api_models, vits_local
from src.main.chatbot import (
    Chatbot,
    get_speech_manager,
    update_system_message_and_profile,
    create_reset_confirm_modal,
    create_delete_session_modal,
    get_allowed_llm_models,
    share_allowed_llm_models
)
from src.main.image_generation import (
    generate_images_wrapper, 
    update_diffusion_model_list,
    toggle_diffusion_api_key_visibility,
    get_allowed_diffusion_models,
    share_allowed_diffusion_models
)
from src.main.translator import translate_interface, upload_handler, LANGUAGES, create_translate_container
from src.main.tts import text_to_speech

from src.tabs.cache_tab import create_cache_tab
from src.tabs.download_tab import create_download_tab
from src.tabs.util_tab import create_util_tab
from src.tabs.setting_tab_custom_model import create_custom_model_tab
from src.tabs.setting_tab_preset import create_system_preset_management_tab
from src.tabs.setting_tab_save_history import create_save_history_tab
from src.tabs.setting_tab_load_history import create_load_history_tab
from src.tabs.setting_tab_session_manager import create_session_management_tab
from src.tabs.device_setting import set_device, create_device_setting_tab

from presets import __all__ as preset_modules

from src.api.comfy_api import client

# os.environ['GRADIO_TEMP_DIR'] = os.path.abspath(TMP_DIR)

# 로깅 설정
from src import logger, args, os_name, arch

from src.start_app import (
    get_last_used_character, 
    initialize_speech_manager, 
    load_presets_from_files, 
    update_presets_on_start, 
    get_last_used_session,
    initialize_app,
    on_app_start,
    register_speech_manager_state,
    shared_on_app_start,
    register_app_state,
    register_app_state_2,
    register_app_state_3,
    register_app_state_4,
    register_app_state_5
)

chat_bot = Chatbot()
                
##########################################
# 3) Gradio UI
##########################################

def on_character_and_language_select(character_name, language):
    """
    캐릭터와 언어 선택 시 호출되는 함수.
    - 캐릭터와 언어 설정 적용
    - 시스템 메시지 프리셋 업데이트
    """
    try:
        speech_manager_state.set_character_and_language(character_name, language)
        system_message = speech_manager_state.get_system_message()
        return system_message
    except ValueError as ve:
        logger.error(f"Character setting error: {ve}")
        return "시스템 메시지 로딩 중 오류가 발생했습니다."
    
def on_character_change(chosen_character, session_id):
    # 1) set_character_and_language
    speech_manager = get_speech_manager(session_id)
    speech_manager.set_character_and_language(chosen_character, speech_manager.current_language)

    # 2) get updated system message
    updated_system_msg = speech_manager.get_system_message()

    # 3) system_message_box에 반영 (UI 갱신)
    #    그리고 DB에 UPDATE
    system_message_box.update(value=updated_system_msg)
    update_system_message_in_db(session_id, updated_system_msg)
    update_last_character_in_db(session_id, chosen_character)

    return updated_system_msg  # UI에 표시

refresh_session_list=chat_bot.refresh_sessions()

with gr.Blocks(css=css, title="AI Companion") as demo:
    speech_manager_state = register_speech_manager_state()
    
    session_id, loaded_history, session_dropdown, last_character, last_preset, system_message, session_label=shared_on_app_start()
    last_sid_state, history_state, last_character_state, session_list_state, overwrite_state = register_app_state()

    # 단일 history_state와 selected_device_state 정의 (중복 제거)
    custom_model_path_state, session_id_state, selected_device_state, character_state, system_message_state = register_app_state_2()
    seed_state, temperature_state, top_k_state, top_p_state, repetition_penalty_state, selected_language_state = register_app_state_3()
    
    reset_confirmation, reset_all_confirmation = register_app_state_4()
    
    max_diffusion_lora_rows, stored_image, stored_image_inpaint = register_app_state_5()
    
    initial_choices, llm_type_choices = share_allowed_llm_models()
    
    diffusion_choices, diffusion_type_choices, diffusion_lora_choices, vae_choices, diffusion_refiner_choices, diffusion_refiner_type_choices = share_allowed_diffusion_models()
    
    tts_choices = tts_api_models + vits_local
    tts_choices = list(dict.fromkeys(tts_choices))
    tts_choices = sorted(tts_choices)
        
    with gr.Column(elem_classes="main-container"):    
        with gr.Row(elem_classes="header-container"):
            with gr.Column(scale=3):
                title = gr.Markdown(f"## {_('main_title')}", elem_classes="title")
                gr.Markdown("### Beta Release")
            with gr.Column(scale=1):
                settings_button = gr.Button("⚙️", elem_classes="settings-button")
            with gr.Column(scale=1):
                language_dropdown = gr.Dropdown(
                    label=_('language_select'),
                    choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
                    value=translation_manager.get_language_display_name(default_language),
                    interactive=True,
                    info=_('language_info'),
                    container=False,
                    elem_classes="custom-dropdown"
                )
                
        with gr.Sidebar(elem_classes="sidebar-container") as sidebar:
            with gr.Column() as tab_side:
                with gr.Row(elem_classes="session-container"):
                    with gr.Column():
                        gr.Markdown("### AI Companion")
                        chatbot_sidetab = gr.Button(value="Chat", elem_classes="tab")
                        diffusion_sidetab = gr.Button(value="Image Generation", elem_classes="tab")
                        storyteller_sidetab = gr.Button(value="Storyteller", elem_classes="tab")
                        tts_sidetab = gr.Button(value="Text to Speech", elem_classes="tab")
                        translate_sidetab = gr.Button(value="Translator", elem_classes="tab")
                        download_sidetab = gr.Button(value="Download Center", elem_classes="tab")
                        
            with gr.Column() as chatbot_side:
                with gr.Row(elem_classes="session-container"):
                    with gr.Column():
                        gr.Markdown("### Chat Session")
                        session_select_dropdown = gr.Dropdown(
                            label="세션 선택",
                            choices=[],  # 앱 시작 시 혹은 별도의 로직으로 세션 목록을 채움
                            value=None,
                            interactive=True,
                            container=False,
                            scale=8,
                            elem_classes="session-dropdown"
                        )
                        chat_title_box=gr.Textbox(
                            value="",
                            interactive=False
                        )
                        add_session_icon_btn = gr.Button("📝", elem_classes="icon-button", scale=1, variant="secondary")
                        delete_session_icon_btn = gr.Button("🗑️", elem_classes="icon-button-delete", scale=1, variant="stop")
                        
                with gr.Row(elem_classes="model-container"):
                    with gr.Column():
                        gr.Markdown("### Model Selection")
                        model_type_dropdown = gr.Radio(
                            label=_("model_type_label"),
                            choices=llm_type_choices,
                            value=llm_type_choices[0],
                            elem_classes="model-dropdown"
                        )
                        model_dropdown = gr.Dropdown(
                            label=_("model_select_label"),
                            choices=initial_choices,
                            value=initial_choices[0] if len(initial_choices) > 0 else None,
                            elem_classes="model-dropdown"
                        )
                        api_key_text = gr.Textbox(
                            label=_("api_key_label"),
                            placeholder="sk-...",
                            visible=False,
                            elem_classes="api-key-input"
                        )
                        lora_dropdown = gr.Dropdown(
                            label="LoRA 모델 선택",
                            choices=get_all_loras(),
                            value="None",
                            interactive=True,
                            visible=False,
                            elem_classes="model-dropdown"
                        )
                    
            with gr.Column() as diffusion_side:          
                with gr.Row(elem_classes="model-container"):
                    with gr.Column():
                        gr.Markdown("### Model Selection")
                        diffusion_model_type_dropdown = gr.Radio(
                            label=_("model_type_label"),
                            choices=diffusion_type_choices,
                            value=diffusion_type_choices[0],
                            elem_classes="model-dropdown"
                        )
                        diffusion_model_dropdown = gr.Dropdown(
                            label=_("model_select_label"),
                            choices=diffusion_choices,
                            value=diffusion_choices[0] if len(diffusion_choices) > 0 else None,
                            elem_classes="model-dropdown"
                        )
                        diffusion_api_key_text = gr.Textbox(
                            label=_("api_key_label"),
                            placeholder="sk-...",
                            visible=False,
                            elem_classes="api-key-input"
                        )
                            
                with gr.Row(elem_classes="model-container"):
                    with gr.Column():
                        gr.Markdown("### Refiner Model Selection")
                        diffusion_refiner_model_dropdown = gr.Dropdown(
                            label=_("refiner_model_select_label"),
                            choices=diffusion_refiner_choices,
                            value=diffusion_refiner_choices[0] if len(diffusion_refiner_choices) > 0 else None,
                            elem_classes="model-dropdown"
                        )
                        diffusion_refiner_start = gr.Slider(
                            label="Refiner Start Step",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=20,
                            visible=False
                        )
                        diffusion_with_refiner_image_to_image_start = gr.Slider(
                            label="Image to Image Start Step",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=20,
                            visible=False
                        )
                            
                with gr.Row(elem_classes="model-container"):
                    with gr.Accordion("LoRA Settings", open=False, elem_classes="accordion-container"):
                        diffusion_lora_multiselect=gr.Dropdown(
                            label="Select LoRA Models",
                            choices=diffusion_lora_choices,
                            value=[],
                            interactive=True,
                            multiselect=True,
                            info="Select LoRA models to apply to the diffusion model.",
                            elem_classes="model-dropdown"
                        )
                        diffusion_lora_text_encoder_sliders=[]
                        diffusion_lora_unet_sliders=[]
                        for i in range(max_diffusion_lora_rows):
                            text_encoder_slider=gr.Slider(
                                label=f"LoRA {i+1} - Text Encoder Weight",
                                minimum=-2.0,
                                maximum=2.0,
                                step=0.01,
                                value=1.0,
                                visible=False,
                                interactive=True
                            )
                            unet_slider = gr.Slider(
                                label=f"LoRA {i+1} - U-Net Weight",
                                minimum=-2.0,
                                maximum=2.0,
                                step=0.01,
                                value=1.0,
                                visible=False,
                                interactive=True
                            )
                            diffusion_lora_text_encoder_sliders.append(text_encoder_slider)
                            diffusion_lora_unet_sliders.append(unet_slider)
                        diffusion_lora_slider_rows=[]
                        for te, unet in zip(diffusion_lora_text_encoder_sliders, diffusion_lora_unet_sliders):
                            diffusion_lora_slider_rows.append(gr.Row([te, unet]))
                        for row in diffusion_lora_slider_rows:
                            row
                            
            with gr.Column() as storyteller_side:
                with gr.Row(elem_classes="model-container"):
                    with gr.Column():
                        gr.Markdown("### Model Selection")
                        storytelling_model_type_dropdown = gr.Radio(
                            label=_("model_type_label"),
                            choices=llm_type_choices,
                            value=llm_type_choices[0],
                            elem_classes="model-dropdown"
                        )
                        storytelling_model_dropdown = gr.Dropdown(
                            label=_("model_select_label"),
                            choices=initial_choices,
                            value=initial_choices[0] if len(initial_choices) > 0 else None,
                            elem_classes="model-dropdown"
                        )
                        storytelling_api_key_text = gr.Textbox(
                            label=_("api_key_label"),
                            placeholder="sk-...",
                            visible=False,
                            elem_classes="api-key-input"
                        )
                        storytelling_lora_dropdown = gr.Dropdown(
                            label="LoRA 모델 선택",
                            choices=get_all_loras(),
                            value="None",
                            interactive=True,
                            visible=False,
                            elem_classes="model-dropdown"
                        )
                        
            with gr.Column() as tts_side:
                with gr.Row(elem_classes="model-container"):
                    with gr.Column():
                        gr.Markdown("### Model Selection")
                        tts_model_type_dropdown = gr.Radio(
                            label=_("model_type_label"),
                            choices=["all", "api", "vits"],
                            value="all",
                            elem_classes="model-dropdown"
                        )
                        tts_model_dropdown = gr.Dropdown(
                            label=_("model_select_label"),
                            choices=tts_choices,
                            value=tts_choices[0] if len(tts_choices) > 0 else "Put Your Models",
                            elem_classes="model-dropdown"
                        )
                        
            with gr.Column() as translate_side:
                with gr.Row(elem_classes="model-container"):
                    with gr.Column():
                        gr.Markdown("### Under Construction")
                                   
        with gr.Row(elem_classes='tabs'):
            with gr.Column(elem_classes='tab-container') as chat_container:
                with gr.Row(elem_classes="model-container"):
                    gr.Markdown("### Chat")
                with gr.Row(elem_classes="chat-interface"):
                    with gr.Column(scale=7):
                        system_message_box = gr.Textbox(
                            label=_("system_message"),
                            value=system_message,
                            placeholder=_("system_message_placeholder"),
                            elem_classes="system-message"
                        )
                        
                        chatbot = gr.Chatbot(
                            height=400, 
                            label="Chatbot", 
                            type="messages", 
                            elem_classes=["chat-messages"]
                        )
                        
                        with gr.Row(elem_classes="input-area"):
                            msg = gr.Textbox(
                                label=_("message_input_label"),
                                placeholder=_("message_placeholder"),
                                scale=9,
                                show_label=False,
                                elem_classes="message-input",
                                submit_btn=True
                            )
                            multimodal_msg = gr.MultimodalTextbox(
                                label=_("message_input_label"),
                                placeholder=_("message_placeholder"),
                                file_types=["image"],
                                scale=9,
                                show_label=False,
                                elem_classes="message-input",
                                submit_btn=True
                            )
                            send_btn = gr.Button(
                                value=_("send_button"),
                                scale=1,
                                variant="primary",
                                elem_classes="send-button",
                                visible=False
                            )
                            image_input = gr.Image(label=_("image_upload_label"), type="pil", visible=False)
                    with gr.Column(scale=3, elem_classes="side-panel"):
                        profile_image = gr.Image(
                            label=_('profile_image_label'),
                            visible=True,
                            interactive=False,
                            show_label=True,
                            width="auto",
                            height="auto",
                            value=characters[last_character]["profile_image"],
                            elem_classes="profile-image"
                        )
                        character_dropdown = gr.Dropdown(
                            label=_('character_select_label'),
                            choices=list(characters.keys()),
                            value=last_character,
                            interactive=True,
                            info=_('character_select_info'),
                            elem_classes='profile-image'
                        )
                        advanced_setting=gr.Accordion(_("advanced_setting"), open=False, elem_classes="accordion-container")
                        with advanced_setting:
                            seed_input = gr.Number(
                                label=_("seed_label"),
                                value=42,
                                precision=0,
                                step=1,
                                interactive=True,
                                info=_("seed_info"),
                                elem_classes="seed-input"
                            )
                            temperature_slider=gr.Slider(
                                label=_("temperature_label"),
                                minimum=0.0,
                                maximum=1.0,
                                value=0.6,
                                step=0.1,
                                interactive=True
                            )
                            top_k_slider=gr.Slider(
                                label=_("top_k_label"),
                                minimum=0,
                                maximum=100,
                                value=20,
                                step=1,
                                interactive=True
                            )
                            top_p_slider=gr.Slider(
                                label=_("top_p_label"),
                                minimum=0.0,
                                maximum=1.0,
                                value=0.9,
                                step=0.1,
                                interactive=True
                            )
                            repetition_penalty_slider=gr.Slider(
                                label=_("repetition_penalty_label"),
                                minimum=0.0,
                                maximum=2.0,
                                value=1.1,
                                step=0.1,
                                interactive=True
                            )
                            preset_dropdown = gr.Dropdown(
                                label="프리셋 선택",
                                choices=get_preset_choices(default_language),
                                value=last_preset,
                                interactive=True,
                                elem_classes="preset-dropdown"
                            )
                            change_preset_button = gr.Button("프리셋 변경")
                            reset_btn = gr.Button(
                                value=_("reset_session_button"),  # "세션 초기화"에 해당하는 번역 키
                                variant="secondary",
                                scale=1
                            )
                            reset_all_btn = gr.Button(
                                value=_("reset_all_sessions_button"),  # "모든 세션 초기화"에 해당하는 번역 키
                                variant="secondary",
                                scale=1
                            )
                            
                with gr.Row(elem_classes="status-bar"):
                    status_text = gr.Markdown("Ready", elem_id="status_text")
                    image_info = gr.Markdown("", visible=False)
                    session_select_info = gr.Markdown(_('select_session_info'))
                    # 초기화 확인 메시지 및 버튼 추가 (숨김 상태로 시작)
                    with gr.Row(visible=False) as reset_confirm_row:
                        reset_confirm_msg = gr.Markdown("⚠️ **정말로 현재 세션을 초기화하시겠습니까? 모든 대화 기록이 삭제됩니다.**")
                        reset_yes_btn = gr.Button("✅ 예", variant="danger")
                        reset_no_btn = gr.Button("❌ 아니요", variant="secondary")

                    with gr.Row(visible=False) as reset_all_confirm_row:
                        reset_all_confirm_msg = gr.Markdown("⚠️ **정말로 모든 세션을 초기화하시겠습니까? 모든 대화 기록이 삭제됩니다.**")
                        reset_all_yes_btn = gr.Button("✅ 예", variant="danger")
                        reset_all_no_btn = gr.Button("❌ 아니요", variant="secondary")
                        
            with gr.Column(elem_classes='tab-container') as diffusion_container:
                with gr.Row(elem_classes="model-container"):
                    gr.Markdown("### Image Generation")        
                with gr.Row(elem_classes="model-container"):
                    with gr.Accordion("Image to Image", open=False, elem_classes="accordion-container"):
                        image_to_image_mode = gr.Radio(
                            label="Image to Image Mode",
                            choices=["None", "Image to Image", "Inpaint", "Inpaint Upload"],
                            value="None",
                            elem_classes="model-dropdown"
                        )
                        with gr.Column():
                            with gr.Row():
                                image_to_image_input = gr.Image(
                                    label="Image to Image",
                                    type="filepath",
                                    sources="upload",
                                    format="png",
                                    visible=False
                                )
                                image_inpaint_input = gr.Image(
                                    label="Image Inpaint",
                                    type="filepath",
                                    sources="upload",
                                    format="png",
                                    visible=False
                                )
                                image_inpaint_masking = gr.ImageMask(
                                    label="Image Inpaint Mask",
                                    type="filepath",
                                    sources="upload",
                                    format="png",
                                    visible=False
                                )
                            image_inpaint_copy = gr.Button(
                                value="Copy",
                                visible=False
                            )
                            
                        blur_radius_slider = gr.Slider(
                            label="Blur Radius",
                            minimum=0,
                            maximum=10,
                            step=0.5,
                            value=5,
                            visible=False
                        )
                        blur_expansion_radius_slider = gr.Slider(
                            label="Blur Expansion Radius",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=1,
                            visible=False
                        )
                        denoise_strength_slider = gr.Slider(
                            label="Denoise Strength",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.5,
                            visible=False
                        )

                with gr.Row(elem_classes="chat-interface"):
                    with gr.Column(scale=7):
                        positive_prompt_input = gr.TextArea(
                            label="Positive Prompt",
                            placeholder="Enter positive prompt...",
                            elem_classes="message-input"
                        )
                        negative_prompt_input = gr.TextArea(
                            label="Negative Prompt",
                            placeholder="Enter negative prompt...",
                            elem_classes="message-input"
                        )
                        
                        with gr.Row():
                            style_dropdown = gr.Dropdown(
                                label="Style",
                                choices=["Photographic", "Digital Art", "Oil Painting", "Watercolor"],
                                value="Photographic"
                            )
                            
                        with gr.Row():
                            width_slider = gr.Slider(
                                label="Width",
                                minimum=128,
                                maximum=2048,
                                step=64,
                                value=512
                            )
                            height_slider = gr.Slider(
                                label="Height",
                                minimum=128,
                                maximum=2048,
                                step=64,
                                value=512
                            )
                        
                        with gr.Row():
                            generation_step_slider=gr.Slider(
                                label="Generation Steps",
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=20
                            )
                        
                        with gr.Row():
                            random_prompt_btn = gr.Button("🎲 Random Prompt", variant="secondary", elem_classes="random-button")
                            generate_btn = gr.Button("🎨 Generate", variant="primary", elem_classes="send-button-alt")
                        
                        gallery = gr.Gallery(
                            label="Generated Images",
                            format="png",
                            columns=2,
                            rows=2
                        )

                    with gr.Column(scale=3, elem_classes="side-panel"):
                        with gr.Accordion("Advanced Settings", open=False, elem_classes="accordion-container"):
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
                                    choices=vae_choices,
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
                        with gr.Accordion(_("advanced_setting"), open=False, elem_classes="accordion-container"):
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
                            
        reset_modal, single_reset_content, all_reset_content, cancel_btn, confirm_btn = create_reset_confirm_modal()
        delete_modal, delete_message, delete_cancel_btn, delete_confirm_btn = create_delete_session_modal()      
        
    # 아래는 변경 이벤트 등록
    def apply_session_immediately(chosen_sid):
        """
        메인탭에서 세션이 선택되면 바로 main_tab.apply_session을 호출해 세션 적용.
        """
        return chat_bot.apply_session(chosen_sid)

    def init_session_dropdown(sessions):
        if not sessions:
            return gr.update(choices=[], value=None)
        return gr.update(choices=sessions, value=sessions[0])
    
    @add_session_icon_btn.click(inputs=[character_dropdown, selected_language_state, speech_manager_state, history_state],outputs=[session_id_state, history_state, session_select_dropdown, session_select_info, chatbot])
    def create_and_apply_session(chosen_character, chosen_language, speech_manager_state, history_state):
        """
        현재 캐릭터/언어에 맞춰 시스템 메시지를 가져온 뒤,
        새 세션을 생성합니다.
        """
        # 1) SpeechManager 인스턴스 획득
        speech_manager = speech_manager_state  # 전역 gr.State로 관리 중인 persona_speech_manager

        # 2) 캐릭터+언어를 설정하고 시스템 메시지 가져오기
        speech_manager.set_character_and_language(chosen_character, chosen_language)
        new_system_msg = speech_manager.get_system_message()

        # 3) DB에 기록할 새 세션 만들기
        new_sid, info, new_history = chat_bot.create_new_session(new_system_msg, chosen_character)

        sessions = get_existing_sessions()
        return [
            new_sid,
            new_history,
            gr.update(choices=sessions, value=new_sid),
            info,
            chat_bot.filter_messages_for_chatbot(new_history)
        ]
    
    # 이벤트 핸들러
    @delete_session_icon_btn.click(inputs=[session_select_dropdown, session_id_state], outputs=[delete_modal, delete_message])
    def show_delete_confirm(selected_sid, current_sid):
        """삭제 확인 모달 표시"""
        if not selected_sid:
            return gr.update(visible=True), "삭제할 세션을 선택하세요."
        if selected_sid == current_sid:
            return gr.update(visible=True), f"현재 활성 세션 '{selected_sid}'은(는) 삭제할 수 없습니다."
        return gr.update(visible=True), f"세션 '{selected_sid}'을(를) 삭제하시겠습니까?"
            
    def delete_selected_session(chosen_sid):
        # 선택된 세션을 삭제 (주의: None 또는 ""인 경우 처리)
        result_msg, _, updated_dropdown = chat_bot.delete_session(chosen_sid, "demo_session")
        return result_msg, updated_dropdown

    # 취소 버튼
    delete_cancel_btn.click(
        fn=lambda: (gr.update(visible=False), ""),
        outputs=[delete_modal, delete_message]
    )

    # 삭제 확인 버튼
    delete_confirm_btn.click(
        fn=chat_bot.delete_session,
        inputs=[session_select_dropdown, session_id_state],
        outputs=[delete_modal, delete_message, session_select_dropdown]
    ).then(
        fn=chat_bot.refresh_sessions,
        inputs=[],
        outputs=[session_select_dropdown]
    )
    
    demo.load(None, None, None).then(
        fn=lambda evt: (gr.update(visible=False), "") if evt.key == "Escape" else (gr.update(), ""),
        inputs=[],
        outputs=[delete_modal, delete_message]
    )
    
    # 시드 입력과 상태 연결
    seed_input.change(
        fn=lambda seed: seed if seed is not None else 42,
        inputs=[seed_input],
        outputs=[seed_state]
    )
    temperature_slider.change(
        fn=lambda temp: temp if temp is not None else 0.6,
        inputs=[temperature_slider],
        outputs=[temperature_state]
    )
    top_k_slider.change(
        fn=lambda top_k: top_k if top_k is not None else 20,
        inputs=[top_k_slider],
        outputs=[top_k_state]
    )
    top_p_slider.change(
        fn=lambda top_p: top_p if top_p is not None else 0.9,
        inputs=[top_p_slider],
        outputs=[top_p_state]
    )
    repetition_penalty_slider.change(
        fn=lambda repetition_penalty: repetition_penalty if repetition_penalty is not None else 1.1,
        inputs=[repetition_penalty_slider],
        outputs=[repetition_penalty_state]
    )
            
    # 프리셋 변경 버튼 클릭 시 호출될 함수 연결
    gr.on(
        triggers=[character_dropdown.change, change_preset_button.click],
        fn=chat_bot.handle_change_preset,
        inputs=[preset_dropdown, history_state, selected_language_state],
        outputs=[history_state, system_message_box, profile_image]
    )

    character_dropdown.change(
        fn=update_system_message_and_profile,
        inputs=[character_dropdown, language_dropdown, speech_manager_state, session_id_state],
        outputs=[system_message_box, profile_image, preset_dropdown]
    )
    
    diffusion_model_dropdown.change(
        fn=lambda selected_model: (
            toggle_diffusion_api_key_visibility(selected_model)
        ),
        inputs=[diffusion_model_dropdown],
        outputs=[diffusion_api_key_text]
    )
    
    demo.load(
        fn=lambda selected_model: (
            toggle_diffusion_api_key_visibility(selected_model)
        ),
        inputs=[diffusion_model_dropdown],
        outputs=[diffusion_api_key_text]
    )
        
    # 모델 선택 변경 시 가시성 토글
    model_dropdown.change(
        fn=lambda selected_model: (
            chat_bot.toggle_api_key_visibility(selected_model),
            chat_bot.toggle_lora_visibility(selected_model),
            chat_bot.toggle_multimodal_msg_input_visibility(selected_model),
            chat_bot.toggle_standard_msg_input_visibility(selected_model)
        ),
        inputs=[model_dropdown],
        outputs=[api_key_text, lora_dropdown, multimodal_msg, msg]
    )
    
    storytelling_model_dropdown.change(
        fn=lambda selected_model: (
            chat_bot.toggle_api_key_visibility(selected_model),
            chat_bot.toggle_lora_visibility(selected_model),
        ),
        inputs=[storytelling_model_dropdown],
        outputs=[storytelling_api_key_text, storytelling_lora_dropdown]
    )
    
    demo.load(
         fn=lambda selected_model: (
            chat_bot.toggle_api_key_visibility(selected_model),
            chat_bot.toggle_lora_visibility(selected_model),
        ),
        inputs=[storytelling_model_dropdown],
        outputs=[storytelling_api_key_text, storytelling_lora_dropdown]
    )
    storytelling_model_type_dropdown.change(
        fn=chat_bot.update_model_list,
        inputs=[storytelling_model_type_dropdown],
        outputs=[storytelling_model_dropdown]
    )
        
    model_type_dropdown.change(
        fn=chat_bot.update_model_list,
        inputs=[model_type_dropdown],
        outputs=[model_dropdown]
    )
    
    diffusion_model_type_dropdown.change(
        fn=update_diffusion_model_list,
        inputs=[diffusion_model_type_dropdown],
        outputs=[diffusion_model_dropdown]
    )
    
    def toggle_refiner_start_step(model):
        slider_visible = model != "None"
        return gr.update(visible=slider_visible)
    
    def toggle_denoise_strength_dropdown(mode):
        slider_visible = mode != "None"
        return gr.update(visible=slider_visible)
    
    def toggle_blur_radius_slider(mode):
        slider_visible = mode == "Inpaint" or mode == "Inpaint Upload"
        return gr.update(visible=slider_visible), gr.update(visible=slider_visible)
    
    def toggle_diffusion_with_refiner_image_to_image_start(model, mode):
        slider_visible = model != "None" and mode != "None"
        return gr.update(visible=slider_visible)
    
    diffusion_refiner_model_dropdown.change(
        fn=lambda model: (
            toggle_refiner_start_step(model)
            ),
        inputs=[diffusion_refiner_model_dropdown],
        outputs=[diffusion_refiner_start]
    ).then(
        fn=toggle_diffusion_with_refiner_image_to_image_start,
        inputs=[diffusion_refiner_model_dropdown, image_to_image_mode],
        outputs=[diffusion_with_refiner_image_to_image_start]
    )
    
    def process_uploaded_image(image):
        print(image)
        image = client.upload_image(image, overwrite=True)
        return image
    
    def process_uploaded_image_for_inpaint(image):
        print(image)
        im = {
            "background": image,
            "layers": [],
            "composite": None
        }
        image = client.upload_image(image, overwrite=True)
        return image, gr.update(value=im)
    
    @image_inpaint_masking.apply(inputs=[image_inpaint_input, image_inpaint_masking], outputs=stored_image_inpaint)
    def process_uploaded_image_inpaint(original_image, mask_image):
        print(original_image)
        print(mask_image)
        mask = client.upload_mask(original_image, mask_image)
        return mask
        
    def toggle_image_to_image_input(mode):
        image_visible = mode == "Image to Image"
        return gr.update(visible=image_visible)
    
    def toggle_image_inpaint_input(mode):
        image_visible = mode == "Inpaint"
        return gr.update(visible=image_visible)
    
    def toggle_image_inpaint_mask(mode):
        image_visible = mode == "Inpaint"
        return gr.update(visible=image_visible)
    
    def toggle_image_inpaint_copy(mode, image):
        button_visible = mode == "Inpaint" and image is not None
        return gr.update(visible=button_visible)
        
    def toggle_image_inpaint_mask_interactive(image):
        image_interactive = image is not None
        return gr.update(interactive=image_interactive)
    
    def copy_image_for_inpaint(image_input, image):
        import cv2
        print(type(image_input))
        im = cv2.imread(image_input)
        height, width, channels = im.shape[:3]
        image['background']=image_input
        image['layers'][0]=np.zeros((height, width, 4), dtype=np.uint8)
        
        return gr.update(value=image)
        
    
    image_to_image_input.change(
        fn=process_uploaded_image,
        inputs=image_to_image_input,
        outputs=stored_image
    )
    
    image_inpaint_input.upload(
        fn=process_uploaded_image,
        inputs=[image_inpaint_input],
        outputs=stored_image
    ).then(
        fn=copy_image_for_inpaint,
        inputs=[image_inpaint_input, image_inpaint_masking],
        outputs=image_inpaint_masking
    ).then(
        fn=toggle_image_inpaint_mask_interactive,
        inputs=image_inpaint_input,
        outputs=image_inpaint_masking
    )
    
    image_inpaint_copy.click(
        fn=toggle_image_inpaint_mask_interactive,
        inputs=image_inpaint_input,
        outputs=image_inpaint_masking
    )
    
    image_to_image_mode.change(
        fn=lambda mode: (
            toggle_image_to_image_input(mode),
            toggle_image_inpaint_input(mode),
            toggle_image_inpaint_mask(mode),
            toggle_denoise_strength_dropdown(mode)
            ),
        inputs=[image_to_image_mode],
        outputs=[image_to_image_input,
                 image_inpaint_input,
                 image_inpaint_masking, 
                 denoise_strength_slider]
    ).then(
        fn=toggle_diffusion_with_refiner_image_to_image_start,
        inputs=[diffusion_refiner_model_dropdown, image_to_image_mode],
        outputs=[diffusion_with_refiner_image_to_image_start]
    ).then(
        fn=toggle_blur_radius_slider,
        inputs=[image_to_image_mode],
        outputs=[blur_radius_slider, blur_expansion_radius_slider]
    )
        
    bot_message_inputs = [session_id_state, history_state, model_dropdown, custom_model_path_state, image_input, api_key_text, selected_device_state, seed_state]
    
    demo.load(
        fn=lambda selected_model: (
            chat_bot.toggle_api_key_visibility(selected_model),
            # main_tab.toggle_image_input_visibility(selected_model),
            chat_bot.toggle_lora_visibility(selected_model),
            chat_bot.toggle_multimodal_msg_input_visibility(selected_model),
            chat_bot.toggle_standard_msg_input_visibility(selected_model)
        ),
        inputs=[model_dropdown],
        outputs=[api_key_text, lora_dropdown, multimodal_msg, msg]
        # outputs=[api_key_text, image_input, lora_dropdown]
    )
        
    def update_character_languages(selected_language, selected_character):
        """
        인터페이스 언어에 따라 선택된 캐릭터의 언어를 업데이트합니다.
        """
        speech_manager = get_speech_manager(session_id_state)
        if selected_language in characters[selected_character]["languages"]:
            # 인터페이스 언어가 캐릭터의 지원 언어에 포함되면 해당 언어로 설정
            speech_manager.current_language = selected_language
        else:
            # 지원하지 않는 언어일 경우 기본 언어로 설정
            speech_manager.current_language = characters[selected_character]["default_language"]
        return gr.update()
    
    def generate_diffusion_lora_weight_sliders(selected_loras: List[str]):
        updates=[]
        for i in range(max_diffusion_lora_rows):
            if i < len(selected_loras):
                # 선택된 LoRA가 있으면 해당 행을 보이게 하고 label 업데이트
                lora_name = selected_loras[i]
                text_update = gr.update(visible=True, label=f"{lora_name} - Text Encoder Weight")
                unet_update = gr.update(visible=True, label=f"{lora_name} - U-Net Weight")
            else:
                # 선택된 LoRA가 없는 행은 숨김 처리
                text_update = gr.update(visible=False)
                unet_update = gr.update(visible=False)
            updates.append(text_update)
            updates.append(unet_update)
        return updates

    @random_prompt_btn.click(outputs=[positive_prompt_input])
    def get_random_prompt():
        """랜덤 프롬프트 생성 함수"""
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

    # 이벤트 핸들러 연결
    generate_btn.click(
        fn=generate_images_wrapper,
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
            stored_image,
            stored_image_inpaint,
            denoise_strength_slider,
            blur_radius_slider,
            blur_expansion_radius_slider,
            diffusion_api_key_text,
            *diffusion_lora_text_encoder_sliders,
            *diffusion_lora_unet_sliders
        ],
        outputs=[gallery, image_history]
    ).then(
        fn=None,
        js=show_confetti
    )

    @language_dropdown.change(
        inputs=[language_dropdown, character_dropdown],
        outputs=[title, session_select_info, language_dropdown, system_message_box, model_type_dropdown, model_dropdown, character_dropdown, api_key_text, image_input, msg, multimodal_msg, send_btn, advanced_setting, seed_input, temperature_slider, top_k_slider, top_p_slider, repetition_penalty_slider, reset_btn, reset_all_btn, diffusion_model_type_dropdown, diffusion_model_dropdown, diffusion_api_key_text]
    )
    def change_language(selected_lang, selected_character):
        """언어 변경 처리 함수"""
        lang_map = {
            "한국어": "ko",
            "日本語": "ja",
            "中文(简体)": "zh_CN",
            "中文(繁體)": "zh_TW",
            "English": "en"
        }
        
        lang_code = lang_map.get(selected_lang, "ko")
        
        if translation_manager.set_language(lang_code):
            if selected_lang in characters[selected_character]["languages"]:
                speech_manager_state.current_language = selected_lang
            else:
                speech_manager_state.current_language = characters[selected_character]["languages"][0]
                
            
            system_presets = {
                "AI 비서 (AI Assistant)": AI_ASSISTANT_PRESET,
                "Image Generator": SD_IMAGE_GENERATOR_PRESET,
                "미나미 아스카 (南飛鳥, みなみあすか, Minami Asuka)": MINAMI_ASUKA_PRESET,
                "마코토노 아오이 (真琴乃葵, まことのあおい, Makotono Aoi)": MAKOTONO_AOI_PRESET,
                "아이노 코이토 (愛野小糸, あいのこいと, Aino Koito)": AINO_KOITO_PRESET,
                "아리아 프린세스 페이트 (アリア·プリンセス·フェイト, Aria Princess Fate)": ARIA_PRINCESS_FATE_PRESET,
                "아리아 프린스 페이트 (アリア·プリンス·フェイト, Aria Prince Fate)": ARIA_PRINCE_FATE_PRESET,
                "왕 메이린 (王美玲, ワン·メイリン, Wang Mei-Ling)": WANG_MEI_LING_PRESET,
                "미스티 레인 (ミスティ·レーン, Misty Lane)": MISTY_LANE_PRESET,
            }
                
            preset_name = system_presets.get(selected_character, AI_ASSISTANT_PRESET)
            system_content = preset_name.get(lang_code, "당신은 유용한 AI 비서입니다.")
            
            return [
                gr.update(value=f"## {_('main_title')}"),
                gr.update(value=_('select_session_info')),
                gr.update(label=_('language_select'),
                info=_('language_info')),
                gr.update(
                    label=_("system_message"),
                    value=system_content,
                    placeholder=_("system_message_placeholder")
                ),
                gr.update(label=_("model_type_label")),
                gr.update(label=_("model_select_label")),
                gr.update(label=_('character_select_label'), info=_('character_select_info')),
                gr.update(label=_("api_key_label")),
                gr.update(label=_("image_upload_label")),
                gr.update(
                    label=_("message_input_label"),
                    placeholder=_("message_placeholder")
                ),
                gr.update(
                    label=_("message_input_label"),
                    placeholder=_("message_placeholder")
                ),
                gr.update(value=_("send_button")),
                gr.update(label=_("advanced_setting")),
                gr.update(label=_("seed_label"), info=_("seed_info")),
                gr.update(label=_("temperature_label")),
                gr.update(label=_("top_k_label")),
                gr.update(label=_("top_p_label")),
                gr.update(label=_("repetition_penalty_label")),
                gr.update(value=_("reset_session_button")),
                gr.update(value=_("reset_all_sessions_button")),
                gr.update(label=_("model_type_label")),
                gr.update(label=_("model_select_label")),
                gr.update(label=_("api_key_label"))
            ]
        else:
            # 언어 변경 실패 시 아무 것도 하지 않음
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        # 메시지 전송 시 함수 연결
    msg.submit(
        fn=chat_bot.process_message_user,
        inputs=[
            msg,  # 사용자 입력
            session_id_state,
            history_state,
            system_message_box,
            character_dropdown,
            selected_language_state
        ],
        outputs=[
            msg,            # 사용자 입력 필드 초기화
            history_state,  # 히스토리 업데이트
            chatbot,        # Chatbot UI 업데이트
        ],
        queue=False
    ).then(
        fn=chat_bot.process_message_bot,
        inputs=[
            session_id_state,
            history_state,
            model_dropdown,
            lora_dropdown,
            custom_model_path_state,
            image_input,
            api_key_text,
            selected_device_state,
            seed_state,
            temperature_state,
            top_k_state,
            top_p_state,
            repetition_penalty_state,
            selected_language_state
        ],
        outputs=[
            history_state,
            chatbot,
            status_text,  # 상태 메시지
            chat_title_box
        ],
        queue=True  # 모델 추론이 들어가므로 True
    )
    
    multimodal_msg.submit(
        fn=chat_bot.process_message_user,
        inputs=[
            multimodal_msg,  # 사용자 입력
            session_id_state,
            history_state,
            system_message_box,
            character_dropdown,
            selected_language_state
        ],
        outputs=[
            multimodal_msg, # 사용자 입력 필드 초기화
            history_state,  # 히스토리 업데이트
            chatbot,        # Chatbot UI 업데이트
        ],
        queue=False
    ).then(
        fn=chat_bot.process_message_bot,
        inputs=[
            session_id_state,
            history_state,
            model_dropdown,
            lora_dropdown,
            custom_model_path_state,
            multimodal_msg,
            api_key_text,
            selected_device_state,
            seed_state,
            temperature_state,
            top_k_state,
            top_p_state,
            repetition_penalty_state,
            selected_language_state
        ],
        outputs=[
            history_state,
            chatbot,
            status_text,  # 상태 메시지
            chat_title_box
        ],
        queue=True  # 모델 추론이 들어가므로 True
    )

    send_btn.click(
        fn=chat_bot.process_message_user,
        inputs=[
            msg,  # 사용자 입력
            session_id_state,
            history_state,
            system_message_box,
            character_dropdown,
            selected_language_state
        ],
        outputs=[
            msg,            # 사용자 입력 필드 초기화
            history_state,  # 히스토리 업데이트
            chatbot,        # Chatbot UI 업데이트
        ],
        queue=False
    ).then(
        fn=chat_bot.process_message_bot,
        inputs=[
            session_id_state,
            history_state,
            model_dropdown,
            lora_dropdown,
            custom_model_path_state,
            image_input,
            api_key_text,
            selected_device_state,
            seed_state,
            temperature_state,
            top_k_state,
            top_p_state,
            repetition_penalty_state,
            selected_language_state
        ],
        outputs=[
            history_state,
            chatbot,
            status_text,  # 상태 메시지
            chat_title_box
        ],
        queue=True  # 모델 추론이 들어가므로 True
    )
        
    demo.load(
        fn=chat_bot.refresh_sessions,
        inputs=[],
        outputs=[session_select_dropdown],
        queue=False
    )
        
    session_select_dropdown.change(
        fn=apply_session_immediately,
        inputs=[session_select_dropdown],
        outputs=[history_state, session_id_state, session_select_info]
    ).then(
        fn=chat_bot.filter_messages_for_chatbot,
        inputs=[history_state],
        outputs=[chatbot]
    )
    
    reset_btn.click(
        fn=lambda: chat_bot.show_reset_modal("single"),
        outputs=[reset_modal, single_reset_content, all_reset_content]
    )
    reset_all_btn.click(
        fn=lambda: chat_bot.show_reset_modal("all"),
        outputs=[reset_modal, single_reset_content, all_reset_content]
    )
    
    cancel_btn.click(
        fn=chat_bot.hide_reset_modal,
        outputs=[reset_modal, single_reset_content, all_reset_content]
    )
    
    confirm_btn.click(
        fn=chat_bot.handle_reset_confirm,
        inputs=[history_state, chatbot, system_message_box, selected_language_state, session_id_state],
        outputs=[reset_modal, single_reset_content, all_reset_content, 
                msg, history_state, chatbot, status_text]
    ).then(
        fn=chat_bot.refresh_sessions,  # 세션 목록 갱신 (전체 초기화의 경우)
        outputs=[session_select_dropdown]
    )
    
    @gr.on(triggers=[chatbot_sidetab.click, demo.load], inputs=[], outputs=[chatbot_side, diffusion_side, storyteller_side, tts_side, translate_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_container, diffusion_container, story_container, tts_container, translate_container, download_container])
    def select_chat_tab():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(elem_classes="tab-active"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    @diffusion_sidetab.click(inputs=[], outputs=[chatbot_side, diffusion_side, storyteller_side, tts_side, translate_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_container, diffusion_container, story_container, tts_container, translate_container, download_container])
    def select_image_generation_tab():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(elem_classes="tab"), gr.update(elem_classes="tab-active"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    @storyteller_sidetab.click(inputs=[], outputs=[chatbot_side, diffusion_side, storyteller_side, tts_side, translate_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_container, diffusion_container, story_container, tts_container, translate_container, download_container])
    def select_storyteller_tab():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab-active"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    @tts_sidetab.click(inputs=[], outputs=[chatbot_side, diffusion_side, storyteller_side, tts_side, translate_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_container, diffusion_container, story_container, tts_container, translate_container, download_container])
    def select_tts_tab():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab-active"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    
    @translate_sidetab.click(inputs=[], outputs=[chatbot_side, diffusion_side, storyteller_side, tts_side, translate_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_container, diffusion_container, story_container, tts_container, translate_container, download_container])
    def select_translate_tab():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab-active"), gr.update(elem_classes="tab"), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    @download_sidetab.click(inputs=[], outputs=[chatbot_side, diffusion_side, storyteller_side, tts_side, translate_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_container, diffusion_container, story_container, tts_container, translate_container, download_container])
    def select_download_tab():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab-active"), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    
    demo.load(None, None, None).then(
        fn=lambda evt: (
            gr.update(visible=False),  # reset_modal
            gr.update(visible=False),  # single_content
            gr.update(visible=False),  # all_content
            None,  # msg (변경 없음)
            None,  # history (변경 없음)
            None,  # chatbot (변경 없음)
            None   # status (변경 없음)
        ) if evt.key == "Escape" else (
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            None,
            None,
            None
        ),
        inputs=[],
        outputs=[
            reset_modal,
            single_reset_content,
            all_reset_content,
            msg,
            history_state,
            chatbot,
            status_text
        ]
    )
            
    with gr.Column(visible=False, elem_classes="settings-popup") as settings_popup:
        with gr.Row(elem_classes="popup-header"):
            gr.Markdown("## Settings")
            with gr.Column():
                close_settings_btn = gr.Button("✕", elem_classes="close-button")
            
        with gr.Tabs():
            create_cache_tab(model_dropdown, language_dropdown)
            create_util_tab()
        
            with gr.Tab("설정"):
                gr.Markdown("### 설정")

                with gr.Tabs():
                    # 사용자 지정 모델 경로 설정 섹션
                    create_custom_model_tab(custom_model_path_state)
                    create_system_preset_management_tab(
                        default_language=default_language,
                        session_id_state=session_id_state,
                        history_state=history_state,
                        selected_language_state=selected_language_state,
                        system_message_box=system_message_box,
                        profile_image=profile_image,
                        chatbot=chatbot
                    )
                    # 프리셋 Dropdown 초기화
                    demo.load(
                        fn=chat_bot.initial_load_presets,
                        inputs=[],
                        outputs=[preset_dropdown],
                        queue=False
                    )                        
                    create_save_history_tab(history_state)
                    create_load_history_tab(history_state)
                    setting_session_management_tab, existing_sessions_dropdown, current_session_display=create_session_management_tab(session_id_state, history_state, session_select_dropdown, system_message_box, chatbot)
                    device_tab, device_dropdown=create_device_setting_tab(default_device)
                    
        with gr.Row(elem_classes="popup-footer"):
            setting_cancel_btn = gr.Button("Cancel", variant="secondary")
            save_settings_btn = gr.Button("Save Changes", variant="primary")
            
        with gr.Column(visible=False, elem_classes="confirm-dialog") as save_confirm_dialog:
            gr.Markdown("### Save Changes?")
            gr.Markdown("Do you want to save the changes you made?")
            with gr.Row():
                confirm_no_btn = gr.Button("No", variant="secondary")
                confirm_yes_btn = gr.Button("Yes", variant="primary")
                
        with gr.Column(visible=False, elem_classes="confirm-dialog") as discard_confirm_dialog:
            gr.Markdown("### Change triggered.")
            gr.Markdown("Change triggered. Do you want to discard the changes you made?")
            with gr.Row():
                confirm_discard_no_btn = gr.Button("No", variant="secondary")
                confirm_discard_yes_btn = gr.Button("Yes", variant="primary")
        
    # 팝업 동작을 위한 이벤트 핸들러 추가
    @settings_button.click(outputs=settings_popup)
    def toggle_settings_popup():
        return gr.update(visible=True)

    @close_settings_btn.click(outputs=settings_popup)
    def close_settings_popup():
        return gr.update(visible=False)

    def handle_escape_key(evt: gr.SelectData):
        """ESC 키를 누르면 팝업을 닫는 함수"""
        if evt.key == "Escape":
            return gr.update(visible=False)

    # 키보드 이벤트 리스너 추가
    demo.load(None, None, None).then(
        fn=handle_escape_key,
        inputs=[],
        outputs=[settings_popup]
    )

    # 설정 변경 시 저장 여부 확인
    def save_settings():
        """설정 저장 함수"""
        # 설정 저장 로직
        return gr.update(visible=False)

    @save_settings_btn.click(outputs=save_confirm_dialog)
    def show_save_confirm():
        """설정 저장 확인 다이얼로그 표시"""
        return gr.update(visible=True)
    
    @confirm_no_btn.click(outputs=save_confirm_dialog)
    def hide_save_confirm():
        """저장 확인 다이얼로그 숨김"""
        return gr.update(visible=False)
    
    @confirm_yes_btn.click(outputs=[save_confirm_dialog, settings_popup])
    def save_and_close():
        """설정 저장 후 팝업 닫기"""
        # 여기에 실제 설정 저장 로직 구현
        return gr.update(visible=False), gr.update(visible=False)
    
    def hide_cancel_confirm():
        return gr.update(visible=False)

    # 설정 변경 여부 추적을 위한 상태 변수 추가
    settings_changed = gr.State(False)
    
    def update_settings_state():
        """설정이 변경되었음을 표시"""
        return True

    # 설정 변경을 감지하여 상태 업데이트
    for input_component in [model_type_dropdown, model_dropdown, device_dropdown, preset_dropdown, system_message_box]:
        input_component.change(
            fn=update_settings_state,
            outputs=settings_changed
        )

    # 취소 버튼 클릭 시 변경사항 확인
    @setting_cancel_btn.click(inputs=[settings_changed], outputs=[discard_confirm_dialog, settings_popup])
    def handle_cancel(changed):
        """취소 버튼 처리"""
        if changed:
            return gr.update(visible=True), gr.update()  # 변경사항이 있으면 확인 다이얼로그 표시
        return gr.update(visible=False), gr.update(visible=False)  # 변경사항이 없으면 바로 닫기
    
    confirm_discard_no_btn.click(
        fn=hide_save_confirm,
        outputs=discard_confirm_dialog
    )
        
    demo.load(
        fn=on_app_start,
        inputs=[], # 언어 상태는 이미 초기화됨
        outputs=[session_id_state, history_state, existing_sessions_dropdown, character_state, preset_dropdown, system_message_state, current_session_display],
        queue=False
    )

if __name__=="__main__":
    print(f"Detected OS: {os_name}, Architecture: {arch}")
    if os_name == "Darwin" and arch == "x86_64":
        raise EnvironmentError("ERROR: Support and compatibility for Intel CPU Based Mac is discontinued.")
    initialize_app()

    demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_port=args.port, width=800)