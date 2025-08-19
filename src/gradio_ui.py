# app.py

import gradio as gr

from src.common.html import css

from src.start_app import app_state
from src.main import create_main_container
from src.main.chatbot import chat_main
from src.main.image_generation import diff_main
from src.main.tts import get_tts_models

from src.tabs import create_settings_popup

from src.start_app import (
    on_app_start,
    register_speech_manager_state,
    shared_on_app_start,
    register_app_state,
    register_app_state_2,
    register_app_state_3,
    register_app_state_4,
    register_app_state_5,
)
                
##########################################
# 3) Gradio UI
##########################################

def create_app():
    with gr.Blocks(css_paths="html/css/style.css", title="AI Companion", fill_height=True) as demo:
        register_speech_manager_state()
        
        shared_on_app_start()
        register_app_state()

        # 단일 history_state와 selected_device_state 정의 (중복 제거)
        register_app_state_2()
        register_app_state_3()
        register_app_state_4()
        register_app_state_5()
        chat_main.share_allowed_llm_models()
        
        diff_main.share_allowed_diffusion_models()
        
        tts_choices = get_tts_models()
            
        settings_button, model_type_dropdown, model_dropdown, system_message_box, preset_dropdown = create_main_container(demo=demo)
                
        settings_popup, close_settings_btn, setting_session_management_tab, existing_sessions_dropdown, current_session_display, device_tab, device_dropdown, setting_cancel_btn, save_settings_btn, save_confirm_dialog, confirm_no_btn, confirm_yes_btn, discard_confirm_dialog, confirm_discard_no_btn, confirm_discard_yes_btn = create_settings_popup(demo=demo)
            
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
            outputs=[app_state.session_id_state, app_state.history_state, existing_sessions_dropdown, app_state.character_state, preset_dropdown, app_state.system_message_state, current_session_display],
            queue=False
        )
        
    return demo