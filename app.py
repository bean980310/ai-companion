# app.py

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
from src.common.args import parse_args
from src.common.default_language import default_language
from src.common.tmp_dir import TMP_DIR
from src.common.character_info import characters
from src.common.utils import get_all_loras, get_diffusion_loras, get_diffusion_vae
import numpy as np

from presets import AI_ASSISTANT_PRESET, SD_IMAGE_GENERATOR_PRESET, MINAMI_ASUKA_PRESET, MAKOTONO_AOI_PRESET, AINO_KOITO_PRESET

from src.models import api_models, transformers_local, gguf_local, mlx_local, diffusion_api_models, diffusers_local, checkpoints_local, vits_local, svc_local
from src.main.chatbot import (
    MainTab,
    get_speech_manager,
    update_system_message_and_profile,
    create_reset_confirm_modal,
    create_delete_session_modal
)
from src.main.image_generation import generate_images_wrapper
from src.main.translator import translate_interface, upload_handler, LANGUAGES

from src.tabs.cache_tab import create_cache_tab
from src.tabs.download_tab import create_download_tab
from src.tabs.util_tab import create_util_tab
from src.tabs.setting_tab_custom_model import create_custom_model_tab
from src.tabs.setting_tab_preset import create_system_preset_management_tab
from src.tabs.setting_tab_save_history import create_save_history_tab
from src.tabs.setting_tab_load_history import create_load_history_tab
from src.tabs.setting_tab_session_manager import create_session_management_tab
from src.tabs.device_setting import set_device, create_device_setting_tab
from src.tabs.sd_prompt_generator_tab import create_sd_prompt_generator_tab

from presets import __all__ as preset_modules

from src.api.comfy_api import client

# os.environ['GRADIO_TEMP_DIR'] = os.path.abspath(TMP_DIR)

# 로깅 설정
from src import logger

args=parse_args()

main_tab=MainTab()

def get_last_used_character(session_id):
    try:
        with sqlite3.connect("chat_history.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_character FROM sessions WHERE id=?", (session_id,))
            row = cursor.fetchone()
            if row and row[0]:
                return row[0]
            else:
                return list(characters.keys())[0]  # 기본 캐릭터로 설정
    except Exception as e:
        logger.error(f"Error fetching last used character: {e}")
        return list(characters.keys())[0]
    
def initialize_speech_manager():
    return PersonaSpeechManager(translation_manager, characters)

def handle_character_change(selected_character, language, speech_manager: PersonaSpeechManager):
    try:
        speech_manager.set_character_and_language(selected_character, language)
        system_message = speech_manager.get_system_message()
        return system_message, gr.update(value=speech_manager.characters[selected_character]["profile_image"])
    except ValueError as e:
        logger.error(str(e))
        return "❌ 선택한 캐릭터가 유효하지 않습니다.", gr.update(value=None)
    
def load_presets_from_files(presets_dir: str) -> List[Dict[str, Any]]:
    """
    presets 디렉토리 내의 모든 프리셋 파일을 로드하여 프리셋 리스트를 반환합니다.
    각 프리셋은 여러 언어로 정의될 수 있습니다.
    """
    presets = []
    presets_path = Path(presets_dir)
    for preset_file in presets_path.glob("*.py"):
        module_name = preset_file.stem
        try:
            module = importlib.import_module(f"presets.{module_name}")
            # __all__ 에 정의된 프리셋 변수들 로드
            for preset_var in getattr(module, "__all__", []):
                preset = getattr(module, preset_var, None)
                if preset:
                    # 각 언어별로 분리하여 추가
                    for lang, content in preset.items():
                        presets.append({
                            "name": preset_var,
                            "language": lang,
                            "content": content.strip()
                        })
        except Exception as e:
            logger.error(f"프리셋 파일 {preset_file} 로드 중 오류 발생: {e}")
    return presets

def update_presets_on_start(presets_dir: str):
    """
    앱 시작 시 presets 디렉토리의 프리셋을 로드하고 데이터베이스를 업데이트합니다.
    """
    # 현재 데이터베이스에 저장된 프리셋 로드
    existing_presets = load_system_presets()  # {(name, language): content, ...}

    # 파일에서 로드한 프리셋
    loaded_presets = load_presets_from_files(presets_dir)

    loaded_preset_keys = set()
    for preset in loaded_presets:
        name = preset["name"]
        language = preset["language"]
        content = preset["content"]
        loaded_preset_keys.add((name, language))
        existing_content = existing_presets.get((name, language))

        if not existing_content:
            # 새로운 프리셋 추가
            success, message = add_system_preset(name, language, content)
            if success:
                logger.info(f"새 프리셋 추가: {name} ({language})")
            else:
                logger.warning(f"프리셋 추가 실패: {name} ({language}) - {message}")
        elif existing_content != content:
            # 기존 프리셋 내용 업데이트
            success, message = add_system_preset(name, language, content, overwrite=True)
            if success:
                logger.info(f"프리셋 업데이트: {name} ({language})")
            else:
                logger.warning(f"프리셋 업데이트 실패: {name} ({language}) - {message}")

    # 데이터베이스에 있지만 파일에는 없는 프리셋 삭제 여부 결정
    for (name, language) in existing_presets.keys():
        if (name, language) not in loaded_preset_keys:
            success, message = delete_system_preset(name, language)
            if success:
                logger.info(f"프리셋 삭제: {name} ({language})")
            else:
                logger.warning(f"프리셋 삭제 실패: {name} ({language}) - {message}")
                
def get_last_used_session():
    try:
        with sqlite3.connect("chat_history.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id
                FROM sessions
                WHERE last_activity IS NOT NULL
                ORDER BY last_activity DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                return None
    except Exception as e:
        logger.error(f"마지막 사용 세션 조회 오류: {e}")
        return None
                
##########################################
# 3) Gradio UI
##########################################
def initialize_app():
    """
    애플리케이션 초기화 함수.
    - 기본 프리셋 삽입
    - 세션 초기화
    """
    initialize_database()
    ensure_demo_session()
    insert_default_presets(translation_manager, overwrite=True)
    return on_app_start(default_language)

def on_app_start(language=None):  # language 매개변수에 기본값 설정
    """
    Gradio 앱이 로드되면서 실행될 콜백.
    """
    if language is None:
        language = default_language
        
    # (1) 마지막으로 사용된 세션 ID 조회
    last_sid = get_last_used_session()
    if last_sid:
        sid = last_sid
        logger.info(f"마지막 사용 세션: {sid}")
    else:
        sid = "demo_session"
        logger.info("마지막 사용 세션이 없어 demo_session 사용")
        
    loaded_history = load_chat_from_db(sid)
    logger.info(f"앱 시작 시 불러온 히스토리: {loaded_history}")
    
    if sid != "demo_session":
        last_character = get_last_used_character(sid)
        logger.info(f"마지막 사용 캐릭터: {last_character}")
    else:
        last_character = list(characters.keys())[0]
        logger.info("demo_session이므로 기본 캐릭터 설정")
    
    sessions = get_existing_sessions()
    logger.info(f"불러온 세션 목록: {sessions}")

    presets = load_system_presets(language=language)
    if len(presets) > 0:
        if last_character in presets:
            preset_name = last_character
        else:
            preset_name = list(system_presets.keys())[0]
        display_system = presets[preset_name]
    else:
        preset_name = None
        display_system = _("system_message_default")
    logger.info(f"로드된 프리셋: {presets}")
    
    preset_list = get_preset_choices(default_language)
    for i in range(len(preset_list)):
        if list(preset_list)[i] == last_character:
            last_preset = list(preset_list)[i]
            break
        
    if not loaded_history:
        system_presets = load_system_presets(language)
        if len(system_presets) > 0:
            preset_name = list(system_presets.keys())[0]
            display_system = system_presets[preset_name]
        else:
            preset_name = None
            display_system = _("system_message_default")
        default_system = {
            "role": "system",
            "content": display_system
        }
        loaded_history = [default_system]
            
    return (
        sid, 
        loaded_history,
        gr.update(choices=sessions, value=sid if sessions else None),
        last_character,
        gr.update(value=last_preset),
        display_system,
        f"현재 세션: {sid}"
    )

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

refresh_session_list=main_tab.refresh_sessions()

with open("html/css/style.css", 'r') as f:
    css = f.read()
    
with open("html/js/script.js", 'r') as f:
    js = f.read()

with gr.Blocks(css=css) as demo:
    speech_manager_state = gr.State(initialize_speech_manager)
    
    session_id, loaded_history, session_dropdown, last_character, last_preset, system_message, session_label=on_app_start()
    last_sid_state=gr.State()
    history_state = gr.State(loaded_history)
    last_character_state = gr.State()
    session_list_state = gr.State()
    overwrite_state = gr.State(False) 

    # 단일 history_state와 selected_device_state 정의 (중복 제거)
    custom_model_path_state = gr.State("")
    session_id_state = gr.State(session_id)
    selected_device_state = gr.State(default_device)
    character_state = gr.State(last_character)
    # preset_state = gr.State(last_preset)
    system_message_state = gr.State(system_message)
    seed_state = gr.State(args.seed)  # 시드 상태 전역 정의
    temperature_state = gr.State(0.6)
    top_k_state = gr.State(20)
    top_p_state = gr.State(0.9)
    repetition_penalty_state = gr.State(1.1)
    selected_language_state = gr.State(default_language)
    
    reset_confirmation = gr.State(False)
    reset_all_confirmation = gr.State(False)
    
    initial_choices = api_models + transformers_local + gguf_local + mlx_local
    initial_choices = list(dict.fromkeys(initial_choices))
    initial_choices = sorted(initial_choices)  # 정렬 추가
    
    diffusion_choices = diffusion_api_models + checkpoints_local + diffusers_local
    diffusion_choices = list(dict.fromkeys(diffusion_choices))
    diffusion_choices = sorted(diffusion_choices)  # 정렬 추가
    
    diffusion_lora_choices = get_diffusion_loras()
    diffusion_lora_choices = list(dict.fromkeys(diffusion_lora_choices))
    diffusion_lora_choices = sorted(diffusion_lora_choices)
    
    vae_choices = get_diffusion_vae()
    
    diffusion_refiner_choices = diffusion_api_models + checkpoints_local + diffusers_local
    diffusion_refiner_choices = list(dict.fromkeys(diffusion_refiner_choices))
    diffusion_refiner_choices = sorted(diffusion_refiner_choices)  # 정렬 추가
    
    tts_choices = vits_local + svc_local
    tts_choices = list(dict.fromkeys(tts_choices))
    tts_choices = sorted(tts_choices)
    
    if len(tts_choices) == 0:
        tts_choices.insert(0, "Put Your Models")
    
    if "None" not in diffusion_refiner_choices:
        diffusion_refiner_choices.insert(0, "None")
    
    if "Default" not in vae_choices:
        vae_choices.insert(0, "Default")
        
    with gr.Column(elem_classes="main-container"):    
        with gr.Row(elem_classes="header-container"):
            with gr.Column(scale=3):
                title = gr.Markdown(f"## {_('main_title')}", elem_classes="title")
            with gr.Column(scale=1):
                settings_button = gr.Button("⚙️ Settings", elem_classes="settings-button")
                language_dropdown = gr.Dropdown(
                    label=_('language_select'),
                    choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
                    value=translation_manager.get_language_display_name(default_language),
                    interactive=True,
                    info=_('language_info'),
                    container=False,
                    elem_classes="custom-dropdown"
                )
        with gr.Sidebar():
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
                    
        with gr.Tabs(elem_classes='tabs') as tabs:
            with gr.Tab('Chat', elem_classes='tab'):
                with gr.Accordion(label="Model Selection", open=False, elem_classes="accordion-container"):
                    with gr.Row(elem_classes="model-container"):
                        with gr.Column(scale=8):
                            model_type_dropdown = gr.Radio(
                                label=_("model_type_label"),
                                choices=["all", "api", "transformers", "gguf", "mlx"],
                                value="all",
                                elem_classes="model-dropdown"
                            )
                        with gr.Column(scale=10):
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
                            width=400,
                            height=400,
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
                        
            with gr.Tab('Image Generation', elem_classes='tab'):
                max_diffusion_lora_rows=10
                stored_image=gr.State()
                stored_image_inpaint=gr.State()
                with gr.Accordion(label="Model Selection", open=False, elem_classes="accordion-container"):          
                    with gr.Row(elem_classes="model-container"):
                        with gr.Column(scale=8):
                            diffusion_model_type_dropdown = gr.Radio(
                                label=_("model_type_label"),
                                choices=["all", "api", "diffusers", "checkpoints"],
                                value="all",
                                elem_classes="model-dropdown"
                            )
                        with gr.Column(scale=10):
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
                        positive_prompt_input = gr.Textbox(
                            label="Positive Prompt",
                            placeholder="Enter positive prompt...",
                            lines=3,
                            elem_classes="message-input"
                        )
                        negative_prompt_input = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Enter negative prompt...",
                            lines=3,
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
            with gr.Tab('Storyteller', elem_classes='tab'):
                with gr.Accordion(label="Model Selection", open=False, elem_classes="accordion-container"):
                    with gr.Row(elem_classes="model-container"):
                        with gr.Column(scale=8):
                            storytelling_model_type_dropdown = gr.Radio(
                                label=_("model_type_label"),
                                choices=["all", "api", "transformers", "gguf", "mlx"],
                                value="all",
                                elem_classes="model-dropdown"
                            )
                        with gr.Column(scale=10):
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
                with gr.Row(elem_classes="chat-interface"):
                    with gr.Column():
                        storytelling_input = gr.Textbox(
                            label="Input",
                            placeholder="Enter your message...",
                            lines=10,
                            elem_classes="message-input",
                        )
                        storytelling_btn = gr.Button("Storytelling", variant="primary", elem_classes="send-button-alt")
                    with gr.Column():
                        storytelling_output = gr.Textbox(
                            label="Output",
                            lines=10,
                            elem_classes="message-output"
                        )
                        
            with gr.Tab('Text to Speech', elem_classes='tab'):
                with gr.Row(elem_classes="chat-interface"):
                    gr.Markdown("# Coming Soon!")
                
            with gr.Tab('Translator', elem_classes='tab'):
                with gr.Row(elem_classes="model-container"):
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                upload_file_lang = gr.Dropdown(
                                    choices=list(dict.fromkeys(LANGUAGES)),
                                    value=list(dict.fromkeys(LANGUAGES))[0],
                                    label="File Language"
                                )
                                upload_file_btn = gr.Button(
                                    "Upload File",
                                    variant="primary",
                                    elem_classes="send-button-alt"
                                )
                            upload_file_input = gr.File(
                                label="Upload File",
                                file_types=["text", "image"],
                                file_count="single"
                            )
                        
                with gr.Row(elem_classes="chat-interface"):
                    with gr.Column():
                        with gr.Row():
                            src_lang_dropdown=gr.Dropdown(
                                choices=list(dict.fromkeys(LANGUAGES)), 
                                value=list(dict.fromkeys(LANGUAGES))[0], 
                                label="Source Language"
                            )
                            tgt_lang_dropdown=gr.Dropdown(
                                choices=list(dict.fromkeys(LANGUAGES)), 
                                value=list(dict.fromkeys(LANGUAGES))[1], 
                                label="Target Language"
                            )
                        with gr.Row():
                            src_textbox=gr.Textbox(
                                label="Source Text",
                                lines=10
                            )
                            translate_result=gr.Textbox(
                                label='Translate result',
                                lines=10
                            )
                        with gr.Row():
                            translate_btn = gr.Button("Translate", variant="primary", elem_classes="send-button-alt")
                            
            create_download_tab()
                            
        reset_modal, single_reset_content, all_reset_content, cancel_btn, confirm_btn = create_reset_confirm_modal()
        delete_modal, delete_message, delete_cancel_btn, delete_confirm_btn = create_delete_session_modal()      
        
    # 아래는 변경 이벤트 등록
    def apply_session_immediately(chosen_sid):
        """
        메인탭에서 세션이 선택되면 바로 main_tab.apply_session을 호출해 세션 적용.
        """
        return main_tab.apply_session(chosen_sid)

    def init_session_dropdown(sessions):
        if not sessions:
            return gr.update(choices=[], value=None)
        return gr.update(choices=sessions, value=sessions[0])
        
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
        new_sid, info, new_history = main_tab.create_new_session(new_system_msg, chosen_character)

        sessions = get_existing_sessions()
        return [
            new_sid,
            new_history,
            gr.update(choices=sessions, value=new_sid),
            info,
            main_tab.filter_messages_for_chatbot(new_history)
        ]
    
    # 이벤트 핸들러
    def show_delete_confirm(selected_sid, current_sid):
        """삭제 확인 모달 표시"""
        if not selected_sid:
            return gr.update(visible=True), "삭제할 세션을 선택하세요."
        if selected_sid == current_sid:
            return gr.update(visible=True), f"현재 활성 세션 '{selected_sid}'은(는) 삭제할 수 없습니다."
        return gr.update(visible=True), f"세션 '{selected_sid}'을(를) 삭제하시겠습니까?"
            
    add_session_icon_btn.click(
        fn=create_and_apply_session,
        inputs=[
            character_dropdown,    # chosen_character
            selected_language_state,  # chosen_language
            speech_manager_state, # persona_speech_manager
            history_state # current history
        ],
        outputs=[
            session_id_state,
            history_state,
            session_select_dropdown,
            session_select_info,
            chatbot]  # create_session이 (new_sid, info)를 반환하므로, 필요하면 여기서 받음
    )
        
    def delete_selected_session(chosen_sid):
        # 선택된 세션을 삭제 (주의: None 또는 ""인 경우 처리)
        result_msg, _, updated_dropdown = main_tab.delete_session(chosen_sid, "demo_session")
        return result_msg, updated_dropdown
        
    # 삭제 버튼 클릭 시 모달 표시
    delete_session_icon_btn.click(
        fn=show_delete_confirm,
        inputs=[session_select_dropdown, session_id_state],
        outputs=[delete_modal, delete_message]
    )

    # 취소 버튼
    delete_cancel_btn.click(
        fn=lambda: (gr.update(visible=False), ""),
        outputs=[delete_modal, delete_message]
    )

    # 삭제 확인 버튼
    delete_confirm_btn.click(
        fn=main_tab.delete_session,
        inputs=[session_select_dropdown, session_id_state],
        outputs=[delete_modal, delete_message, session_select_dropdown]
    ).then(
        fn=main_tab.refresh_sessions,
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
    change_preset_button.click(
        fn=main_tab.handle_change_preset,
        inputs=[preset_dropdown, history_state, selected_language_state],
        outputs=[history_state, system_message_box, profile_image]
    )
            
    character_dropdown.change(
        fn=update_system_message_and_profile,
        inputs=[character_dropdown, language_dropdown, speech_manager_state, session_id_state],
        outputs=[system_message_box, profile_image, preset_dropdown]
    ).then(
        fn=main_tab.handle_change_preset,
        inputs=[preset_dropdown, history_state, selected_language_state],
        outputs=[history_state, system_message_box, profile_image]
    )
        
    # 모델 선택 변경 시 가시성 토글
    model_dropdown.change(
        fn=lambda selected_model: (
            main_tab.toggle_api_key_visibility(selected_model),
            main_tab.toggle_lora_visibility(selected_model),
            main_tab.toggle_multimodal_msg_input_visibility(selected_model),
            main_tab.toggle_standard_msg_input_visibility(selected_model)
        ),
        inputs=[model_dropdown],
        outputs=[api_key_text, lora_dropdown, multimodal_msg, msg]
    )
    
    storytelling_model_dropdown.change(
        fn=lambda selected_model: (
            main_tab.toggle_api_key_visibility(selected_model),
            main_tab.toggle_lora_visibility(selected_model),
        ),
        inputs=[storytelling_model_dropdown],
        outputs=[storytelling_api_key_text, storytelling_lora_dropdown]
    )
    
    demo.load(
         fn=lambda selected_model: (
            main_tab.toggle_api_key_visibility(selected_model),
            main_tab.toggle_lora_visibility(selected_model),
        ),
        inputs=[storytelling_model_dropdown],
        outputs=[storytelling_api_key_text, storytelling_lora_dropdown]
    )
    storytelling_model_type_dropdown.change(
        fn=main_tab.update_model_list,
        inputs=[storytelling_model_type_dropdown],
        outputs=[storytelling_model_dropdown]
    )
        
    model_type_dropdown.change(
        fn=main_tab.update_model_list,
        inputs=[model_type_dropdown],
        outputs=[model_dropdown]
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
    
    image_inpaint_masking.apply(
        fn=process_uploaded_image_inpaint,
        inputs=[image_inpaint_input, image_inpaint_masking],
        outputs=stored_image_inpaint
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
            main_tab.toggle_api_key_visibility(selected_model),
            # main_tab.toggle_image_input_visibility(selected_model),
            main_tab.toggle_lora_visibility(selected_model),
            main_tab.toggle_multimodal_msg_input_visibility(selected_model),
            main_tab.toggle_standard_msg_input_visibility(selected_model)
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
    
    translate_btn.click(
        fn=translate_interface,
        inputs=[src_textbox, src_lang_dropdown, tgt_lang_dropdown],
        outputs=[translate_result]
    )
    
    upload_file_btn.click(
        fn=upload_handler,
        inputs=[upload_file_input, upload_file_lang],
        outputs=src_textbox
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
            *diffusion_lora_text_encoder_sliders,
            *diffusion_lora_unet_sliders
        ],
        outputs=[gallery, image_history]
    )

    random_prompt_btn.click(
        fn=get_random_prompt,
        outputs=[positive_prompt_input]
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
                "AI 비서(AI Assistant)": AI_ASSISTANT_PRESET,
                "Image Generator": SD_IMAGE_GENERATOR_PRESET,
                "미나미 아스카 (南飛鳥, みなみあすか, Minami Asuka)": MINAMI_ASUKA_PRESET,
                "마코토노 아오이 (真琴乃葵, まことのあおい, Makotono Aoi)": MAKOTONO_AOI_PRESET,
                "아이노 코이토 (愛野小糸, あいのこいと, Aino Koito)": AINO_KOITO_PRESET
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

    # 언어 변경 이벤트 연결
    language_dropdown.change(
        fn=change_language,
        inputs=[language_dropdown, character_dropdown],
        outputs=[
            title,
            session_select_info,
            language_dropdown,
            system_message_box,
            model_type_dropdown,
            model_dropdown,
            character_dropdown,
            api_key_text,
            image_input,
            msg,
            multimodal_msg,
            send_btn,
            advanced_setting,
            seed_input,
            temperature_slider,
            top_k_slider,
            top_p_slider,
            repetition_penalty_slider,
            reset_btn,
            reset_all_btn,
            diffusion_model_type_dropdown,
            diffusion_model_dropdown,
            diffusion_api_key_text
        ]
    )
    
        # 메시지 전송 시 함수 연결
    msg.submit(
        fn=main_tab.process_message_user,
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
        fn=main_tab.process_message_bot,
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
        fn=main_tab.process_message_user,
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
        fn=main_tab.process_message_bot,
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
        fn=main_tab.process_message_user,
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
        fn=main_tab.process_message_bot,
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
        fn=main_tab.refresh_sessions,
        inputs=[],
        outputs=[session_select_dropdown],
        queue=False
    )
        
    session_select_dropdown.change(
        fn=apply_session_immediately,
        inputs=[session_select_dropdown],
        outputs=[history_state, session_id_state, session_select_info]
    ).then(
        fn=main_tab.filter_messages_for_chatbot,
        inputs=[history_state],
        outputs=[chatbot]
    )
    
    reset_btn.click(
        fn=lambda: main_tab.show_reset_modal("single"),
        outputs=[reset_modal, single_reset_content, all_reset_content]
    )
    reset_all_btn.click(
        fn=lambda: main_tab.show_reset_modal("all"),
        outputs=[reset_modal, single_reset_content, all_reset_content]
    )
    
    cancel_btn.click(
        fn=main_tab.hide_reset_modal,
        outputs=[reset_modal, single_reset_content, all_reset_content]
    )
    
    confirm_btn.click(
        fn=main_tab.handle_reset_confirm,
        inputs=[history_state, chatbot, system_message_box, selected_language_state, session_id_state],
        outputs=[reset_modal, single_reset_content, all_reset_content, 
                msg, history_state, chatbot, status_text]
    ).then(
        fn=main_tab.refresh_sessions,  # 세션 목록 갱신 (전체 초기화의 경우)
        outputs=[session_select_dropdown]
    )
    
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
                        fn=main_tab.initial_load_presets,
                        inputs=[],
                        outputs=[preset_dropdown],
                        queue=False
                    )                        
                    create_save_history_tab(history_state)
                    create_load_history_tab(history_state)
                    setting_session_management_tab, existing_sessions_dropdown, current_session_display=create_session_management_tab(session_id_state, history_state, session_select_dropdown, system_message_box, chatbot)
                    device_tab, device_dropdown=create_device_setting_tab(default_device)
                    
            create_sd_prompt_generator_tab()
        with gr.Row(elem_classes="popup-footer"):
            cancel_btn = gr.Button("Cancel", variant="secondary")
            save_settings_btn = gr.Button("Save Changes", variant="primary")
            
        with gr.Column(visible=False, elem_classes="confirm-dialog") as save_confirm_dialog:
            gr.Markdown("### Save Changes?")
            gr.Markdown("Do you want to save the changes you made?")
            with gr.Row():
                confirm_no_btn = gr.Button("No", variant="secondary")
                confirm_yes_btn = gr.Button("Yes", variant="primary")
        
    # 팝업 동작을 위한 이벤트 핸들러 추가
    def toggle_settings_popup():
        return gr.update(visible=True)

    def close_settings_popup():
        return gr.update(visible=False)

    settings_button.click(
        fn=toggle_settings_popup,
        outputs=settings_popup
    )

    close_settings_btn.click(
        fn=close_settings_popup,
        outputs=settings_popup
    )
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

    def show_save_confirm():
        """설정 저장 확인 다이얼로그 표시"""
        return gr.update(visible=True)
    
    def hide_save_confirm():
        """저장 확인 다이얼로그 숨김"""
        return gr.update(visible=False)
    
    def save_and_close():
        """설정 저장 후 팝업 닫기"""
        # 여기에 실제 설정 저장 로직 구현
        return gr.update(visible=False), gr.update(visible=False) 
    
    # 이벤트 연결
    save_settings_btn.click(
        fn=show_save_confirm,
        outputs=save_confirm_dialog
    )

    confirm_no_btn.click(
        fn=hide_save_confirm,
        outputs=save_confirm_dialog
    )

    confirm_yes_btn.click(
        fn=save_and_close,
        outputs=[save_confirm_dialog, settings_popup]
    )

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
    def handle_cancel(changed):
        """취소 버튼 처리"""
        if changed:
            return gr.update(visible=True)  # 변경사항이 있으면 확인 다이얼로그 표시
        return gr.update(visible=False), gr.update(visible=False)  # 변경사항이 없으면 바로 닫기

    cancel_btn.click(
        fn=handle_cancel,
        inputs=[settings_changed],
        outputs=[save_confirm_dialog, settings_popup]
    )
        
    demo.load(
        fn=on_app_start,
        inputs=[], # 언어 상태는 이미 초기화됨
        outputs=[session_id_state, history_state, existing_sessions_dropdown, character_state, preset_dropdown, system_message_state, current_session_display],
        queue=False
    )

if __name__=="__main__":
    
    initialize_app()

    demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_port=args.port, width=800)