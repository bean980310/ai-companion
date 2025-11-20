from typing import Tuple
import gradio as gr

from ..start_app import app_state, ui_component
from ..main.chatbot.chatbot import Chatbot


chat_bot=Chatbot()

def create_session_management_tab()-> Tuple[gr.Tab, gr.Dropdown, gr.Textbox]:
    setting_session_management_tab = gr.Tab("세션 관리")
    with setting_session_management_tab:
        gr.Markdown("### 세션 관리")
        with gr.Row():
            refresh_sessions_btn = gr.Button("세션 목록 갱신")
            existing_sessions_dropdown = gr.Dropdown(
                label="기존 세션 목록",
                choices=[],  # 초기에는 비어 있다가, 버튼 클릭 시 갱신
                value=None,
                interactive=True
            )
            current_session_display = gr.Textbox(
                label="현재 세션 ID",
                value="",
                interactive=False
            )
                        
        with gr.Row():
            create_new_session_btn = gr.Button("새 세션 생성")
            apply_session_btn = gr.Button("세션 적용")
            delete_session_btn = gr.Button("세션 삭제")
                        
        session_manage_info = gr.Textbox(
            label="세션 관리 결과",
            interactive=False
        )
                        
        current_session_display = gr.Textbox(
            label="현재 세션 ID",
            value="",
            interactive=False
        )

        app_state.session_id_state.change(
            fn=lambda sid: f"현재 세션: {sid}" if sid else "세션 없음",
            inputs=[app_state.session_id_state],
            outputs=[current_session_display]
        )
                        
        refresh_sessions_btn.click(
            fn=chat_bot.refresh_sessions,
            inputs=[],
            outputs=[existing_sessions_dropdown]
        ).then(
            fn=chat_bot.refresh_sessions,
            inputs=[],
            outputs=[ui_component.session_select_dropdown]
        )
                        
        # (2) 새 세션 생성
        create_new_session_btn.click(
            fn=lambda: chat_bot.create_new_session(ui_component.system_message_box.value),
            inputs=[],
            outputs=[app_state.session_id_state, session_manage_info]
        ).then(
            fn=lambda: [],
            inputs=[],
            outputs=[app_state.history_state]
        ).then(
            fn=chat_bot.filter_messages_for_chatbot,
            inputs=[app_state.history_state],
            outputs=[ui_component.chatbot]
        ).then(
            fn=chat_bot.refresh_sessions,
            inputs=[],
            outputs=[ui_component.session_select_dropdown]
        )
                        
        apply_session_btn.click(
            fn=chat_bot.apply_session,
            inputs=[existing_sessions_dropdown],
            outputs=[app_state.history_state, app_state.session_id_state, session_manage_info]
        ).then(
            fn=chat_bot.filter_messages_for_chatbot,
            inputs=[app_state.history_state],
            outputs=[ui_component.chatbot]
        ).then(
            fn=chat_bot.refresh_sessions,
            inputs=[],
            outputs=[ui_component.session_select_dropdown]
        )
                        
        with gr.Row(visible=False) as delete_session_confirm_row:
            delete_session_confirm_msg = gr.Markdown("⚠️ **정말로 선택한 세션을 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.**")
            delete_session_yes_btn = gr.Button("✅ 예", variant="danger")
            delete_session_no_btn = gr.Button("❌ 아니요", variant="secondary")

        # “세션 삭제” 버튼 클릭 시, 확인창(문구/버튼) 보이기
        delete_session_btn.click(
            fn=lambda: (
                gr.update(visible=True),
                gr.update(visible=True), 
                gr.update(visible=True)
            ),
            inputs=[],
            outputs=[delete_session_confirm_row, delete_session_yes_btn, delete_session_no_btn]
        )

        # (5) 예 버튼 → 실제 세션 삭제
        delete_session_yes_btn.click(
            fn=chat_bot.delete_session,
            inputs=[existing_sessions_dropdown, app_state.session_id_state],
            outputs=[session_manage_info, delete_session_confirm_msg, existing_sessions_dropdown]
        ).then(
            fn=lambda: (gr.update(visible=False)),
            inputs=[],
            outputs=[delete_session_confirm_row],
            queue=False
        ).then(
            fn=chat_bot.refresh_sessions,
            inputs=[],
            outputs=[ui_component.session_select_dropdown]
        )

        # “아니요” 버튼: “취소되었습니다” 메시지 + 문구/버튼 숨기기
        delete_session_no_btn.click(
            fn=lambda: (
                "❌ 삭제가 취소되었습니다.",
                gr.update(visible=False)
            ),
            inputs=[],
            outputs=[session_manage_info, delete_session_confirm_row],
            queue=False
        )
    
    return setting_session_management_tab, existing_sessions_dropdown, current_session_display