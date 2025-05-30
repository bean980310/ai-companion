import gradio as gr

from ..start_app import app_state

def create_custom_model_tab():
    with gr.Tab("사용자 지정 모델 경로 설정"):
        custom_path_text = gr.Textbox(
            label="사용자 지정 모델 경로",
            placeholder="./models/custom-model",
        )
        apply_custom_path_btn = gr.Button("경로 적용")

        # custom_path_text -> custom_model_path_state 저장
        def update_custom_path(path):
            return path

        apply_custom_path_btn.click(
            fn=update_custom_path,
            inputs=[custom_path_text],
            outputs=[app_state.custom_model_path_state]
        )