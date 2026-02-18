import gradio as gr
from src.main.chatbot import chat_main, chat_bot, chat_component
from src.main.chatbot.component import MAX_VISIBLE_SESSIONS
from src.start_app import app_state, ui_component, initialize_speech_manager
from src.common.character_info import characters
from src.common.translations import translation_manager, _
from src.common_blocks import create_page_header, get_language_code
from src.pages import header
from src.characters import PersonaSpeechManager
from src.common.database import get_existing_sessions, get_existing_sessions_with_names
from presets import (
    AI_ASSISTANT_PRESET,
    SD_IMAGE_GENERATOR_PRESET,
    MINAMI_ASUKA_PRESET,
    MAKOTONO_AOI_PRESET,
    AINO_KOITO_PRESET,
    ARIA_PRINCESS_FATE_PRESET,
    ARIA_PRINCE_FATE_PRESET,
    WANG_MEI_LING_PRESET,
    MISTY_LANE_PRESET,
    LILY_EMPRESS_PRESET,
    CHOI_YUNA_PRESET,
    CHOI_YURI_PRESET
    )

from src import args
from src.models import default_device
from src.common.default_language import default_language

with gr.Blocks() as demo:
    # 0. Page-Specific State Registration
    def register_chat_state():
        # Chat Parameters
        app_state.seed_state = gr.State(args.seed)
        app_state.max_length_state = gr.State(-1)
        app_state.temperature_state = gr.State(0.6)
        app_state.top_k_state = gr.State(20)
        app_state.top_p_state = gr.State(0.9)
        app_state.repetition_penalty_state = gr.State(1.1)
        app_state.enable_thinking_state = gr.State(False)
        
        # Session & History
        # loaded_history is set in register_global_state, but we need a gr.State for it in the chat context if we want to mutate it?
        # Actually, create_chat_container usually takes app_state.history_state.
        # old register_app_state: history_state = gr.State(app_state.loaded_history)
        app_state.history_state = gr.State(app_state.loaded_history)
        
        app_state.last_sid_state = gr.State()
        app_state.last_character_state = gr.State()
        app_state.session_list_state = gr.State()
        app_state.overwrite_state = gr.State(False)
        
        app_state.custom_model_path_state = gr.State("")
        app_state.character_state = gr.State(app_state.initial_last_character)
        app_state.system_message_state = gr.State(app_state.initial_system_message)
        
        app_state.reset_confirmation = gr.State(False)
        app_state.reset_all_confirmation = gr.State(False)

        # Temporary session state
        app_state.is_temp_session_state = gr.State(False)

        # We also need these if they were formerly global
        # selected_language_state is GLOBAL (in app.py)
        # selected_device_state is GLOBAL (in app.py)
        
        # Load Model Lists
        chat_main.share_allowed_llm_models()
        
        # Initialize Shared States Locally for this Page
        # These are needed by create_chat_container's additional_inputs
        app_state.session_id_state = gr.State(getattr(app_state, "initial_session_id", "demo_session"))
        app_state.selected_language_state = gr.State(default_language)
        app_state.selected_device_state = gr.State(default_device)
        app_state.speech_manager_state = gr.State(initialize_speech_manager)
        
    register_chat_state()

    # 1. Page Header with Language Selector
    # page_header = header.page_header
    # language_dropdown = header.language_dropdown

    # 2. UI Construction
    with gr.Sidebar():
        # Model container first, then session container (as per plan)
        chat_side_model = chat_component.create_chatbot_side_model_container()
        chat_side_session = chat_component.create_chatbot_side_session_container()

    # Main Content
    # chat_main.create_chat_container() creates a Column with tab-container class.
    # We can reuse it.
    
    chat_container_obj = chat_main.create_chat_container()

    
    
    # Now we need to extract the components for wiring.
    # References from chat_side - Session list components
    session_select_dropdown = chat_side_session.session_select_dropdown
    chat_title_box = chat_side_session.chat_title_box
    add_session_icon_btn = chat_side_session.add_session_icon_btn
    delete_session_icon_btn = chat_side_session.delete_session_icon_btn

    # New session list components
    session_rows = chat_side_session.session_rows
    session_buttons = chat_side_session.session_buttons
    session_delete_buttons = chat_side_session.session_delete_buttons
    selected_session_id = chat_side_session.selected_session_id

    # References from chat_model
    text_model_provider_dropdown = chat_side_model.model_provider_dropdown
    text_model_type_dropdown = chat_side_model.model_type_dropdown
    text_model_dropdown = chat_side_model.model_dropdown
    text_api_key_text = chat_side_model.api_key_text
    text_lora_dropdown = chat_side_model.lora_dropdown
    text_model_refresh_button = chat_side_model.refresh_button
    text_model_clear_all_btn = chat_side_model.clear_all_btn

    # References from chat_container
    # The create_chat_container creates a NEW ChatbotMain instance, 
    # so we need to use the one returned: chat_container_obj
    
    system_message_accordion = chat_container_obj.main_panel.system_message_accordion
    system_message_box = chat_container_obj.main_panel.system_message_box
    chatbot = chat_container_obj.main_panel.chatbot
    msg = chat_container_obj.main_panel.msg
    
    profile_image = chat_container_obj.side_panel.profile_image
    character_dropdown = chat_container_obj.side_panel.character_dropdown
    
    text_advanced_settings = chat_container_obj.side_panel.advanced_setting
    
    text_seed_input = chat_container_obj.side_panel.seed_input
    text_max_length_input = chat_container_obj.side_panel.max_length_input
    text_temperature_slider = chat_container_obj.side_panel.temperature_slider
    text_top_k_slider = chat_container_obj.side_panel.top_k_slider
    text_top_p_slider = chat_container_obj.side_panel.top_p_slider
    text_repetition_penalty_slider = chat_container_obj.side_panel.repetition_penalty_slider
    text_enable_thinking_checkbox = chat_container_obj.side_panel.enable_thinking_checkbox
    preset_dropdown = chat_container_obj.side_panel.preset_dropdown
    change_preset_button = chat_container_obj.side_panel.change_preset_button
    reset_btn = chat_container_obj.side_panel.reset_btn
    reset_all_btn = chat_container_obj.side_panel.reset_all_btn
    
    status_text = chat_container_obj.status_bar.status_text
    image_info = chat_container_obj.status_bar.image_info
    session_select_info = chat_container_obj.status_bar.session_select_info

    # Modals from chat_bot (these were in create_main_container)
    reset_modal, single_reset_content, all_reset_content, cancel_btn, confirm_btn = chat_bot.create_reset_confirm_modal()
    delete_modal, delete_message, delete_cancel_btn, delete_confirm_btn = chat_bot.create_delete_session_modal()

    # 2. Event Wiring (Copied from src/main/__init__.py and adapted)

    # ===== Session List Helper Functions =====

    def refresh_session_list_ui(current_session_id: str = None):
        """
        세션 목록 UI를 갱신합니다.
        각 세션 row의 visibility와 버튼 텍스트를 업데이트합니다.
        """
        sessions = get_existing_sessions_with_names()
        updates = []

        for i in range(MAX_VISIBLE_SESSIONS):
            if i < len(sessions):
                sid, name = sessions[i]
                # Row visibility
                updates.append(gr.update(visible=True))
                # Button text (session name/title)
                updates.append(gr.update(value=name))
            else:
                # Hide unused rows
                updates.append(gr.update(visible=False))
                updates.append(gr.update(value=""))

        return updates

    def apply_session_with_character(chosen_sid: str, selected_language: str):
        """세션 변경 시 히스토리와 캐릭터 정보를 함께 로드하고 UI 업데이트"""
        if not chosen_sid:
            return ([], None, "세션을 선택하세요.", [], gr.update(), gr.update(), False)

        loaded_history, session_id, last_character, status_msg = chat_bot.apply_session(chosen_sid)

        # 캐릭터 정보가 없으면 기본값 사용
        if not last_character or last_character not in characters:
            last_character = list(characters.keys())[0]

        # 캐릭터의 프로필 이미지 가져오기
        profile_img = characters[last_character].get("profile_image", None)

        # chatbot 형식으로 변환
        chatbot_history = chat_bot.filter_messages_for_chatbot(loaded_history)

        return (
            loaded_history,
            session_id,
            status_msg,
            chatbot_history,
            gr.update(value=last_character),  # character_dropdown 업데이트
            gr.update(value=profile_img) if profile_img else gr.update(),  # profile_image 업데이트
            False,  # is_temp_session = False (기존 세션 선택시)
        )

    # ===== Session Button Click Handlers =====

    def make_session_click_handler(index: int):
        """세션 버튼 클릭 핸들러를 생성합니다."""
        def handler():
            sessions = get_existing_sessions_with_names()
            if index < len(sessions):
                return sessions[index][0]  # Return session ID
            return None
        return handler

    def make_delete_click_handler(index: int):
        """삭제 버튼 클릭 핸들러를 생성합니다."""
        def handler(current_sid: str):
            sessions = get_existing_sessions_with_names()
            if index < len(sessions):
                selected_sid = sessions[index][0]
                selected_name = sessions[index][1]
                if selected_sid == current_sid:
                    return gr.update(visible=True), f"현재 활성 세션 '{selected_name}'은(는) 삭제할 수 없습니다.", selected_sid
                return gr.update(visible=True), f"세션 '{selected_name}'을(를) 삭제하시겠습니까?", selected_sid
            return gr.update(visible=False), "", ""
        return handler

    # Wire session button click events
    for i in range(MAX_VISIBLE_SESSIONS):
        session_buttons[i].click(
            fn=make_session_click_handler(i),
            outputs=[selected_session_id]
        ).then(
            fn=apply_session_with_character,
            inputs=[selected_session_id, app_state.selected_language_state],
            outputs=[
                app_state.history_state,
                app_state.session_id_state,
                session_select_info,
                chatbot,
                character_dropdown,
                profile_image,
                app_state.is_temp_session_state,
            ]
        )

        session_delete_buttons[i].click(
            fn=make_delete_click_handler(i),
            inputs=[app_state.session_id_state],
            outputs=[delete_modal, delete_message, selected_session_id]
        )

    # ===== New Chat (Temporary Session) Handler =====

    def create_temp_session_handler(chosen_character: str, chosen_language: str, speech_manager_state: PersonaSpeechManager):
        """임시 세션을 생성합니다 (DB에 저장하지 않음)"""
        speech_manager = speech_manager_state
        speech_manager.set_character_and_language(chosen_character, chosen_language)
        new_system_msg = speech_manager.get_system_message()

        is_temp, temp_history, temp_sys_msg, temp_char, chatbot_display, status = chat_bot.create_temp_session(
            new_system_msg, chosen_character
        )

        return [
            is_temp,  # is_temp_session_state
            temp_history,  # history_state
            chatbot_display,  # chatbot
            status,  # session_select_info
            "",  # session_id_state (empty for temp session)
        ]

    add_session_icon_btn.click(
        fn=create_temp_session_handler,
        inputs=[character_dropdown, app_state.selected_language_state, app_state.speech_manager_state],
        outputs=[
            app_state.is_temp_session_state,
            app_state.history_state,
            chatbot,
            session_select_info,
            app_state.session_id_state,
        ]
    )

    # ===== Legacy dropdown change handler (for backward compatibility) =====
    session_select_dropdown.change(
        fn=apply_session_with_character,
        inputs=[session_select_dropdown, app_state.selected_language_state],
        outputs=[
            app_state.history_state,
            app_state.session_id_state,
            session_select_info,
            chatbot,
            character_dropdown,
            profile_image,
            app_state.is_temp_session_state,
        ]
    )

    # ===== Delete Session Handlers =====

    @delete_session_icon_btn.click(inputs=[session_select_dropdown, app_state.session_id_state], outputs=[delete_modal, delete_message])
    def show_delete_confirm(selected_sid: str, current_sid: str):
        if not selected_sid:
            return gr.update(visible=True), "삭제할 세션을 선택하세요."
        if selected_sid == current_sid:
            return gr.update(visible=True), f"현재 활성 세션 '{selected_sid}'은(는) 삭제할 수 없습니다."
        return gr.update(visible=True), f"세션 '{selected_sid}'을(를) 삭제하시겠습니까?"

    delete_cancel_btn.click(
        fn=lambda: (gr.update(visible=False), ""),
        outputs=[delete_modal, delete_message]
    )

    # Build outputs for session list refresh
    session_list_outputs = []
    for i in range(MAX_VISIBLE_SESSIONS):
        session_list_outputs.append(session_rows[i])
        session_list_outputs.append(session_buttons[i])

    def delete_and_refresh_session_list(selected_sid: str, current_sid: str):
        """세션을 삭제하고 세션 목록을 갱신합니다."""
        modal_visible, message, dropdown_update = chat_bot.delete_session(selected_sid, current_sid)

        # Refresh session list
        list_updates = refresh_session_list_ui(current_sid)

        return [modal_visible, message] + list_updates

    delete_confirm_btn.click(
        fn=delete_and_refresh_session_list,
        inputs=[selected_session_id, app_state.session_id_state],
        outputs=[delete_modal, delete_message] + session_list_outputs
    )
    
    # State synchronization
    text_seed_input.change(lambda seed: seed if seed is not None else 42, inputs=[text_seed_input], outputs=[app_state.seed_state])
    text_max_length_input.change(lambda max_length: max_length if max_length is not None else -1, inputs=[text_max_length_input], outputs=[app_state.max_length_state])
    text_temperature_slider.change(lambda temp: temp if temp is not None else 0.6, inputs=[text_temperature_slider], outputs=[app_state.temperature_state])
    text_top_k_slider.change(lambda top_k: top_k if top_k is not None else 20, inputs=[text_top_k_slider], outputs=[app_state.top_k_state])
    text_top_p_slider.change(lambda top_p: top_p if top_p is not None else 0.9, inputs=[text_top_p_slider], outputs=[app_state.top_p_state])
    text_repetition_penalty_slider.change(lambda repetition_penalty: repetition_penalty if repetition_penalty is not None else 1.1, inputs=[text_repetition_penalty_slider], outputs=[app_state.repetition_penalty_state])
    text_enable_thinking_checkbox.change(lambda enable: enable if enable is True else False, inputs=[text_enable_thinking_checkbox], outputs=[app_state.enable_thinking_state])

    # Preset & Character
    character_dropdown.change(
        fn=chat_bot.update_system_message_and_profile,
        inputs=[character_dropdown, header.language_dropdown, app_state.session_id_state],
        outputs=[system_message_box, profile_image, preset_dropdown]
    )

    character_dropdown.change(
        fn=chat_bot.handle_change_preset,
        inputs=[preset_dropdown, app_state.history_state, app_state.selected_language_state],
        outputs=[app_state.history_state, system_message_box, profile_image]
    )

    # character_dropdown.change(
    #     fn=chat_bot.update_system_message_and_profile,
    #     inputs=[character_dropdown, language_dropdown, app_state.session_id_state],
    #     outputs=[system_message_box, profile_image, preset_dropdown]
    # )

    # Note: language_dropdown is NOT here (it will likely be in the Navbar or Global Header if we keep one).
    # If we want language support PER PAGE or Global, we need to decide. 
    # For now, let's assume we might need to inject it or access app_state.selected_language_state properly.
    # The original code had a global language_dropdown in the Header. 
    # We might need to handle language changes via a shared mechanism or reload.
    
    # Model visibility
    gr.on(
        triggers=[text_model_dropdown.change, demo.load],
        fn=chat_bot.toggle_enable_thinking_visibility,
        inputs=[text_model_dropdown],
        outputs=[text_enable_thinking_checkbox]
    )
    
    gr.on(
        triggers=[text_model_provider_dropdown.change, text_model_type_dropdown.change, demo.load],
        fn=chat_bot.update_model_list,
        inputs=[text_model_provider_dropdown, text_model_type_dropdown],
        outputs=[text_model_type_dropdown, text_model_dropdown]
    )

    gr.on(
        triggers=[text_model_provider_dropdown.change, demo.load],
        fn=lambda provider: (
            chat_bot.toggle_api_key_visibility(provider),
            chat_bot.toggle_lora_visibility(provider),
        ),
        inputs=[text_model_provider_dropdown],
        outputs=[text_api_key_text, text_lora_dropdown]
    )
    
    # Reset Logic
    reset_btn.click(fn=lambda: chat_bot.show_reset_modal("single"), outputs=[reset_modal, single_reset_content, all_reset_content])
    reset_all_btn.click(fn=lambda: chat_bot.show_reset_modal("all"), outputs=[reset_modal, single_reset_content, all_reset_content])
    cancel_btn.click(fn=chat_bot.hide_reset_modal, outputs=[reset_modal, single_reset_content, all_reset_content])

    def handle_reset_with_session_update(history, chatbot_state, system_msg, selected_character, language, session_id):
        """
        초기화 처리 후 세션 드롭다운과 세션 ID를 업데이트합니다.
        단일 세션 초기화와 전체 세션 초기화 모두 동일한 출력 형식을 반환합니다.
        """
        result = chat_bot.handle_reset_confirm(
            history=history,
            chatbot=chatbot_state,
            system_msg=system_msg,
            selected_character=selected_character,
            language=language,
            session_id=session_id
        )

        # reset_all_sessions는 9개 값 반환 (session_id 포함)
        # reset_session은 8개 값 반환
        if len(result) == 9:
            # 전체 초기화: session_id도 업데이트
            return result[:8] + (result[8],)  # 8개 + session_id
        else:
            # 단일 초기화: session_id는 그대로 유지
            return result + (session_id,)

    confirm_btn.click(
        fn=handle_reset_with_session_update,
        inputs=[app_state.history_state, chatbot, system_message_box, character_dropdown, app_state.selected_language_state, app_state.session_id_state],
        outputs=[reset_modal, single_reset_content, all_reset_content, msg, app_state.history_state, chatbot, status_text, session_select_dropdown, app_state.session_id_state]
    )
    
    # Load Init - Refresh session list on page load
    demo.load(
        fn=refresh_session_list_ui,
        inputs=[],
        outputs=session_list_outputs,
        queue=False
    )

    # Also keep legacy dropdown updated for backward compatibility
    demo.load(
        fn=chat_bot.refresh_sessions,
        inputs=[],
        outputs=[session_select_dropdown],
        queue=False
    )

    # Refresh session list when session_id changes (e.g., after temp session becomes permanent)
    app_state.session_id_state.change(
        fn=refresh_session_list_ui,
        inputs=[],
        outputs=session_list_outputs
    )

    # System Message Init
    def init_system_message_accordion():
        return gr.update(open=False)

    demo.load(
        fn=init_system_message_accordion,
        inputs=[],
        outputs=[system_message_accordion]
    )

    # Language Change Event
    def on_chat_language_change(selected_lang: str, selected_character: str):
        lang_code = get_language_code(selected_lang)
        translation_manager.set_language(lang_code)

        if selected_lang in characters[selected_character]["languages"]:
            app_state.speech_manager_state.current_language = selected_lang
        else:
            app_state.speech_manager_state.current_language = characters[selected_character]["languages"][0]
                
            
        system_presets: dict[str, dict[str, str]] = {
            "AI 비서 (AI Assistant)": AI_ASSISTANT_PRESET,
            "Image Generator": SD_IMAGE_GENERATOR_PRESET,
            "미나미 아스카 (南飛鳥, みなみあすか, Minami Asuka)": MINAMI_ASUKA_PRESET,
            "마코토노 아오이 (真琴乃葵, まことのあおい, Makotono Aoi)": MAKOTONO_AOI_PRESET,
            "아이노 코이토 (愛野小糸, あいのこいと, Aino Koito)": AINO_KOITO_PRESET,
            "아리아 프린세스 페이트 (アリア·プリンセス·フェイト, Aria Princess Fate)": ARIA_PRINCESS_FATE_PRESET,
            "아리아 프린스 페이트 (アリア·プリンス·フェイト, Aria Prince Fate)": ARIA_PRINCE_FATE_PRESET,
            "왕 메이린 (王美玲, ワン·メイリン, Wang Mei-Ling)": WANG_MEI_LING_PRESET,
            "미스티 레인 (ミスティ·レーン, Misty Lane)": MISTY_LANE_PRESET,
            '릴리 엠프레스 (リリー·エンプレス, Lily Empress)': LILY_EMPRESS_PRESET,
            "최유나 (崔有娜, チェ·ユナ, Choi Yuna)": CHOI_YUNA_PRESET,
            "최유리 (崔有莉, チェ·ユリ, Choi Yuri)": CHOI_YURI_PRESET
        }
                
        preset_name = system_presets.get(selected_character, AI_ASSISTANT_PRESET)
        system_content = preset_name.get(lang_code, "당신은 유용한 AI 비서입니다.")

        return [
            gr.update(label=_("model_provider_label")),
            gr.update(label=_("model_type_label")),
            gr.update(label=_("model_select_label")),
            gr.update(label=_("api_key_label")),
            gr.update(label=_("lora_select_label")),
            gr.update(label=_('system_message')),
            gr.update(label=_('system_message'), value=system_content),
            gr.update(label=_('advanced_setting')),
            gr.update(label=_('seed_label'), info=_('seed_info')),
            gr.update(label=_('temperature_label')),
            gr.update(label=_('top_k_label')),
            gr.update(label=_('top_p_label')),
            gr.update(label=_('repetition_penalty_label')),
            gr.update(value=_('reset_session_button')),
            gr.update(value=_('reset_all_sessions_button')),
            lang_code
        ]

    # language_dropdown.change(
    #     fn=on_chat_language_change,
    #     inputs=[language_dropdown, character_dropdown],
    #     outputs=[
    #         page_header.title,
    #         language_dropdown,
    #         system_message_accordion,
    #         system_message_box,
    #         text_advanced_settings,
    #         text_seed_input,
    #         text_temperature_slider,
    #         text_top_k_slider,
    #         text_top_p_slider,
    #         text_repetition_penalty_slider,
    #         reset_btn,
    #         reset_all_btn,
    #         app_state.selected_language_state
    #     ]
    # )


if __name__ == "__main__":
    demo.launch()
