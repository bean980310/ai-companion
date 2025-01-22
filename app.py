# app.py

import importlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
import gradio as gr
import logging
from logging.handlers import RotatingFileHandler
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
    update_system_message_in_db)
from src.models.models import default_device
from src.common.cache import models_cache
from src.common.translations import translation_manager, _, TranslationManager
from src.characters.persona_speech_manager import PersonaSpeechManager
from src.common.args import parse_args
from src.common.default_language import default_language

from src.tabs.main_tab import (
    api_models, 
    transformers_local, 
    gguf_local, 
    mlx_local,
    MainTab,
    characters,
    get_speech_manager,
    update_system_message_and_profile,
    create_reset_confirm_modal,
    create_delete_session_modal
)

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
import json

from src.common.css import css
from src.common.js import js

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 로그 포맷 정의
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 콘솔 핸들러 추가
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 파일 핸들러 추가 (로테이팅 파일 핸들러 사용)
log_file = "app.log"  # 원하는 로그 파일 경로로 변경 가능
rotating_file_handler = RotatingFileHandler(
    log_file, maxBytes=5*1024*1024, backupCount=5  # 5MB마다 새로운 파일로 교체, 최대 5개 백업
)
rotating_file_handler.setFormatter(formatter)
logger.addHandler(rotating_file_handler)


args=parse_args()

main_tab=MainTab()

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
    
    sessions = get_existing_sessions()
    logger.info(f"불러온 세션 목록: {sessions}")

    presets = load_system_presets(language=language)
    logger.info(f"로드된 프리셋: {presets}")
    
    if not loaded_history:
        system_presets = load_system_presets(language)
        if len(system_presets) > 0:
            preset_name = list(system_presets.keys())[0]
            default_system = {
                "role": "system",
                "content": system_presets[preset_name]
            }
            loaded_history = [default_system]
        else:
            default_system = {
                "role": "system",
                "content": "당신은 유용한 AI 비서입니다."
            }
            loaded_history = [default_system]
            
    return (
        sid, 
        loaded_history,
        gr.update(choices=sessions, value=sid if sessions else None),
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

    return updated_system_msg  # UI에 표시

refresh_session_list=main_tab.refresh_sessions()

with gr.Blocks(css=css) as demo:
    speech_manager_state = gr.State(initialize_speech_manager)
    
    session_id, loaded_history, session_dropdown, session_label=on_app_start()
    last_sid_state=gr.State()
    history_state = gr.State(loaded_history)
    session_list_state = gr.State()
    overwrite_state = gr.State(False) 

    # 단일 history_state와 selected_device_state 정의 (중복 제거)
    custom_model_path_state = gr.State("")
    session_id_state = gr.State()
    selected_device_state = gr.State(default_device)
    seed_state = gr.State(args.seed)  # 시드 상태 전역 정의
    selected_language_state = gr.State(default_language)
    
    reset_confirmation = gr.State(False)
    reset_all_confirmation = gr.State(False)
    
    initial_choices = api_models + transformers_local + gguf_local + mlx_local
    initial_choices = list(dict.fromkeys(initial_choices))
    initial_choices = sorted(initial_choices)  # 정렬 추가
    
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
        with gr.Row(elem_classes="session-container"):
            session_select_dropdown = gr.Dropdown(
                label="세션 선택",
                choices=[],  # 앱 시작 시 혹은 별도의 로직으로 세션 목록을 채움
                value=None,
                interactive=True,
                container=False,
                scale=8,
                elem_classes="session-dropdown"
            )
            add_session_icon_btn = gr.Button("📝", elem_classes="icon-button", scale=1, variant="secondary")
            delete_session_icon_btn = gr.Button("🗑️", elem_classes="icon-button-delete", scale=1, variant="stop")
            
            delete_modal, delete_message, delete_cancel_btn, delete_confirm_btn = create_delete_session_modal()
        
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
        with gr.Row(elem_classes="chat-interface"):
            with gr.Column(scale=7):
                system_message_box = gr.Textbox(
                    label=_("system_message"),
                    value=_("system_message_default"),
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
                    elem_classes="message-input"
                    )
                    send_btn = gr.Button(
                        value=_("send_button"),
                        scale=1,
                        variant="primary",
                        elem_classes="send-button"
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
                    value=characters[list(characters.keys())[0]]["profile_image"],
                    elem_classes="profile-image"
                )
                character_dropdown = gr.Dropdown(
                    label=_('character_select_label'),
                    choices=list(characters.keys()),
                    value=list(characters.keys())[0],
                    interactive=True,
                    info=_('character_select_info'),
                    elem_classes='profile-image'
                )
                advanced_setting=gr.Accordion(_("advanced_setting"), open=False)
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
                    reset_modal, single_reset_content, all_reset_content, cancel_btn, confirm_btn = create_reset_confirm_modal()
                    preset_dropdown = gr.Dropdown(
                        label="프리셋 선택",
                        choices=get_preset_choices(default_language),
                        value=list(get_preset_choices(default_language))[0] if get_preset_choices(default_language) else None,
                        interactive=True,
                        elem_classes="preset-dropdown"
                    )
                    change_preset_button = gr.Button("프리셋 변경")
                    character_conversation_dropdown = gr.CheckboxGroup(
                        label="대화할 캐릭터 선택",
                        choices=get_preset_choices(default_language),  # 추가 캐릭터 이름
                        value=list(get_preset_choices(default_language))[0] if get_preset_choices(default_language) else None,
                        interactive=True
                    )
                    start_conversation_button = gr.Button("대화 시작")
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
        new_sid, info, new_history = main_tab.create_new_session(new_system_msg)

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
            
    # 프리셋 변경 버튼 클릭 시 호출될 함수 연결
    change_preset_button.click(
        fn=main_tab.handle_change_preset,
        inputs=[preset_dropdown, history_state, selected_language_state],
        outputs=[history_state, system_message_box, profile_image]
    )
            
    character_dropdown.change(
        fn=update_system_message_and_profile,
        inputs=[character_dropdown, language_dropdown, speech_manager_state, session_id_state],
        outputs=[system_message_box, profile_image]
    )
        
    # 모델 선택 변경 시 가시성 토글
    model_dropdown.change(
        fn=lambda selected_model: (
            main_tab.toggle_api_key_visibility(selected_model),
            main_tab.toggle_image_input_visibility(selected_model)
        ),
        inputs=[model_dropdown],
        outputs=[api_key_text, image_input]
    )
        
    model_type_dropdown.change(
        fn=main_tab.update_model_list,
        inputs=[model_type_dropdown],
        outputs=[model_dropdown]
    )
        
    bot_message_inputs = [session_id_state, history_state, model_dropdown, custom_model_path_state, image_input, api_key_text, selected_device_state, seed_state]
    
    demo.load(
        fn=lambda selected_model: (
            main_tab.toggle_api_key_visibility(selected_model),
            main_tab.toggle_image_input_visibility(selected_model)
        ),
        inputs=[model_dropdown],
        outputs=[api_key_text, image_input]
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
            system_presets = load_system_presets(lang_code)
                
            if len(system_presets) > 0:
                preset_name = list(system_presets.keys())[0]
                system_content = system_presets[preset_name]
            else:
                system_content = _("system_message_default")

            return [
                gr.update(value=f"## {_('main_title')}"),
                gr.update(value=_('select_session_info')),
                gr.update(label=_('language_select'),
                info=_('language_info')),
                gr.update(
                    label=_("system_message"),
                    value=_("system_message_default"),
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
                gr.update(value=_("send_button")),
                gr.update(value=_("advanced_setting")),
                gr.update(label=_("seed_label"), info=_("seed_info")),
                gr.update(value=_("reset_session_button")),
                gr.update(value=_("reset_all_sessions_button")),
            ]
        else:
            # 언어 변경 실패 시 아무 것도 하지 않음
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

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
            send_btn,
            advanced_setting,
            seed_input,
            reset_btn,
            reset_all_btn,
        ]
    )
    
        # 메시지 전송 시 함수 연결
    msg.submit(
        fn=main_tab.process_message,
        inputs=[
            msg,  # 사용자 입력
            session_id_state,
            history_state,
            system_message_box,
            model_dropdown,
            custom_model_path_state,
            image_input,
            api_key_text,
            selected_device_state,
            seed_state,
            selected_language_state,
            character_dropdown
        ],
        outputs=[
            msg,            # 사용자 입력 필드 초기화
            history_state,  # 히스토리 업데이트
            chatbot,        # Chatbot UI 업데이트
            status_text     # 상태 메시지 업데이트
        ],
        queue=False
    ).then(
        fn=main_tab.filter_messages_for_chatbot,
        inputs=[history_state],
        outputs=chatbot,
        queue=False
    )

    send_btn.click(
        fn=main_tab.process_message,
        inputs=[
            msg, 
            session_id_state, 
            history_state, 
            system_message_box, 
            model_dropdown, 
            custom_model_path_state, 
            image_input, 
            api_key_text, 
            selected_device_state, 
            seed_state,
            selected_language_state,
            character_dropdown
        ],
        outputs=[
            msg, 
            history_state, 
            chatbot, 
            status_text
        ],
        queue=False
    ).then(
        fn=main_tab.filter_messages_for_chatbot,            # 추가된 부분
        inputs=[history_state],
        outputs=chatbot,                           # chatbot에 최종 전달
        queue=False
    )
        
    start_conversation_button.click(
        fn=main_tab.process_character_conversation,
        inputs=[
            history_state,
            character_conversation_dropdown,
            model_type_dropdown, 
            model_dropdown,
            custom_model_path_state,
            image_input,
            api_key_text,
            selected_device_state,
            seed_state
        ],
        outputs=[history_state, profile_image]
    ).then(
        fn=main_tab.filter_messages_for_chatbot,  # 히스토리를 채팅창에 표시하기 위한 필터링
        inputs=[history_state],
        outputs=[chatbot]
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
            create_download_tab()
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
        outputs=[session_id_state, history_state, existing_sessions_dropdown,
        current_session_display],
        queue=False
    )

if __name__=="__main__":
    
    initialize_app()

    demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_port=args.port, width=800)