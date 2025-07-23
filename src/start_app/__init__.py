import sqlite3
import gradio as gr
import importlib
from typing import Optional, Dict, List, Any, Literal
from pathlib import Path
from dataclasses import dataclass

from .. import logger, args
from ..common.character_info import characters
from ..common.translations import translation_manager, _
from ..common.database import (
    load_system_presets, 
    add_system_preset, 
    delete_system_preset, 
    initialize_database, 
    ensure_demo_session,
    insert_default_presets,
    load_chat_from_db,
    get_existing_sessions,
    get_preset_choices
)

from ..models import default_device

from ..common.default_language import default_language

from ..characters.persona_speech_manager import PersonaSpeechManager

from .app_state_manager import app_state
from .ui_component_manager import ui_component

def register_speech_manager_state():
    speech_manager_state = gr.State(initialize_speech_manager)
    app_state.speech_manager_state = speech_manager_state
    
    return speech_manager_state

def shared_on_app_start():
    session_id, loaded_history, session_dropdown, last_character, last_preset, system_message, session_label=on_app_start()
    app_state.session_id = session_id
    app_state.loaded_history = loaded_history
    app_state.session_dropdown = session_dropdown
    app_state.last_character = last_character
    app_state.last_preset = last_preset
    app_state.system_message = system_message
    app_state.session_label = session_label
    
    return session_id, loaded_history, session_dropdown, last_character, last_preset, system_message, session_label

def register_app_state():
    last_sid_state=gr.State()
    history_state = gr.State(app_state.loaded_history)
    last_character_state = gr.State()
    session_list_state = gr.State()
    overwrite_state = gr.State(False)
    
    app_state.last_sid_state = last_sid_state
    app_state.history_state = history_state
    app_state.last_character_state = last_character_state
    app_state.session_list_state = session_list_state
    app_state.overwrite_state = overwrite_state
    
    return last_sid_state, history_state, last_character_state, session_list_state, overwrite_state

def register_app_state_2():
    custom_model_path_state = gr.State("")
    session_id_state = gr.State(app_state.session_id)
    selected_device_state = gr.State(default_device)
    character_state = gr.State(app_state.last_character)
    system_message_state = gr.State(app_state.system_message)
    
    app_state.custom_model_path_state = custom_model_path_state
    app_state.session_id_state = session_id_state
    app_state.selected_device_state = selected_device_state
    app_state.character_state = character_state
    app_state.system_message_state = system_message_state
    
    return custom_model_path_state, session_id_state, selected_device_state, character_state, system_message_state

def register_app_state_3():
    seed_state = gr.State(args.seed)  # 시드 상태 전역 정의
    temperature_state = gr.State(0.6)
    top_k_state = gr.State(20)
    top_p_state = gr.State(0.9)
    repetition_penalty_state = gr.State(1.1)
    selected_language_state = gr.State(default_language)
    
    app_state.seed_state = seed_state
    app_state.temperature_state = temperature_state
    app_state.top_k_state = top_k_state
    app_state.top_p_state = top_p_state
    app_state.repetition_penalty_state = repetition_penalty_state
    app_state.selected_language_state = selected_language_state
    
    return seed_state, temperature_state, top_k_state, top_p_state, repetition_penalty_state, selected_language_state

def register_app_state_4():
    reset_confirmation = gr.State(False)
    reset_all_confirmation = gr.State(False)
    
    app_state.reset_confirmation = reset_confirmation
    app_state.reset_all_confirmation = reset_all_confirmation
    
    return reset_confirmation, reset_all_confirmation

def register_app_state_5():
    max_diffusion_lora_rows=10
    stored_image = gr.State()
    stored_image_inpaint = gr.State()
    
    app_state.max_diffusion_lora_rows = max_diffusion_lora_rows
    app_state.stored_image = stored_image
    app_state.stored_image_inpaint = stored_image_inpaint
    
    return max_diffusion_lora_rows, stored_image, stored_image_inpaint

def create_header_container():
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
            
    ui_component.title = title
    ui_component.settings_button = settings_button
    ui_component.language_dropdown = language_dropdown
    
    return title, settings_button, language_dropdown

def create_tab_side():
    with gr.Column() as tab_side:
        with gr.Row(elem_classes="session-container"):
            with gr.Column():
                gr.Markdown("### AI Companion")
                # llm_integrated_sidetab = gr.Button(value="LLM", elem_classes="tab")
                chatbot_sidetab = gr.Button(value="Text Section", elem_classes="tab")
                diffusion_sidetab = gr.Button(value="Vision Section", elem_classes="tab")
                storyteller_sidetab = gr.Button(value="Storyteller", elem_classes="tab")
                tts_sidetab = gr.Button(value="Audio Section", elem_classes="tab")
                translate_sidetab = gr.Button(value="Translator", elem_classes="tab")
                download_sidetab = gr.Button(value="Download Center", elem_classes="tab")

    return tab_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab

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
    
def initialize_app():
    """
    애플리케이션 초기화 함수.
    - 기본 프리셋 삽입
    - 세션 초기화
    """
    # print(f"Detected OS: {os_name}, Architecture: {arch}")
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
        last_preset = last_character
            
    return (
        sid, 
        loaded_history,
        gr.update(choices=sessions, value=sid if sessions else None),
        last_character,
        gr.update(value=last_preset),
        display_system,
        f"현재 세션: {sid}"
    )
    
