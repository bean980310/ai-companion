import gradio as gr
from src.common.database import save_chat_history_db, save_chat_history_csv, save_chat_button_click

def create_save_history_tab(history_state):
    with gr.Tab("채팅 기록 저장"):
        save_button = gr.Button("채팅 기록 저장", variant="secondary")
        save_info = gr.Textbox(label="저장 결과", interactive=False)

        save_csv_button = gr.Button("채팅 기록 CSV 저장", variant="secondary")
        save_csv_info = gr.Textbox(label="CSV 저장 결과", interactive=False)

        save_db_button = gr.Button("채팅 기록 DB 저장", variant="secondary")
        save_db_info = gr.Textbox(label="DB 저장 결과", interactive=False)

        def save_chat_button_click_csv(history):
            if not history:
                return "채팅 이력이 없습니다."
            saved_path = save_chat_history_csv(history)
            if saved_path is None:
                return "❌ 채팅 기록 CSV 저장 실패"
            else:
                return f"✅ 채팅 기록 CSV가 저장되었습니다: {saved_path}"
                            
        def save_chat_button_click_db(history):
            if not history:
                return "채팅 이력이 없습니다."
            ok = save_chat_history_db(history, session_id="demo_session")
            if ok:
                return f"✅ DB에 채팅 기록이 저장되었습니다 (session_id=demo_session)"
            else:
                return "❌ DB 저장 실패"

        save_csv_button.click(
            fn=save_chat_button_click_csv,
            inputs=[history_state],
            outputs=save_csv_info
        )

        # save_button이 클릭되면 save_chat_button_click 실행
        save_button.click(
            fn=save_chat_button_click,
            inputs=[history_state],
            outputs=save_info
        )
                        
        save_db_button.click(
            fn=save_chat_button_click_db,
            inputs=[history_state],
            outputs=save_db_info
        )