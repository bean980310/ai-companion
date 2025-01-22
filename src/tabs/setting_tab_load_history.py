import gradio as gr
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_load_history_tab(history_state):
    with gr.Tab("채팅 히스토리 재로드"):
        upload_json = gr.File(label="대화 JSON 업로드", file_types=[".json"])
        load_info = gr.Textbox(label="로딩 결과", interactive=False)
                        
        def load_chat_from_json(json_file):
            """
            업로드된 JSON 파일을 파싱하여 history_state에 주입
            """
            if not json_file:
                return [], "파일이 없습니다."
            try:
                with open(json_file.name, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    return [], "JSON 구조가 올바르지 않습니다. (list 형태가 아님)"
                # data를 그대로 history_state로 반환
                return data, "✅ 대화가 로딩되었습니다."
            except Exception as e:
                logger.error(f"JSON 로드 오류: {e}")
                return [], f"❌ 로딩 실패: {e}"

        upload_json.change(
            fn=load_chat_from_json,
            inputs=[upload_json],
            outputs=[history_state, load_info]
        )