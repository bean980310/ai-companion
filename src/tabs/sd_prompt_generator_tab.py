import gradio as gr
from src.models.models import generate_stable_diffusion_prompt_cached
from src.tabs.main_tab import generator_choices
from src.models.api_models import api_models
from src.models.local_models import transformers_local, gguf_local, mlx_local

def create_sd_prompt_generator_tab():
    with gr.Tab("SD Prompt 생성"):
        gr.Markdown("# Stable Diffusion 프롬프트 생성기")
                
        with gr.Row():
            user_input_sd = gr.Textbox(
                label="이미지 설명",
                placeholder="예: 해질녘의 아름다운 해변 풍경",
                lines=2
            )
            generate_prompt_btn = gr.Button("프롬프트 생성")
                
        with gr.Row():
            selected_model_sd = gr.Dropdown(
                label="언어 모델 선택",
                choices=generator_choices,
                value="gpt-3.5-turbo",
                interactive=True
            )
            model_type_sd = gr.Dropdown(
                label="모델 유형",
                choices=["api", "transformers", "gguf", "mlx"],
                value="api",
                interactive=False  # 자동 설정되므로 사용자가 변경하지 못하도록 설정
            )
        
        api_key_sd = gr.Textbox(
            label="OpenAI API Key",
            type="password",
            visible=True
        )
                
        prompt_output_sd = gr.Textbox(
            label="생성된 프롬프트",
            placeholder="여기에 생성된 프롬프트가 표시됩니다...",
            lines=4,
            interactive=False
        )
                
        # 사용자 지정 모델 경로 입력 필드
        custom_model_path_sd = gr.Textbox(
            label="사용자 지정 모델 경로",
            placeholder="./models/custom-model",
            visible=False
        )
                
        # 모델 선택 시 모델 유형 자동 설정 및 API Key 필드 가시성 제어
        def update_model_type(selected_model):
            if selected_model in api_models:
                model_type = "api"
                api_visible = True
                custom_visible = False
            elif selected_model in transformers_local:
                model_type = "transformers"
                api_visible = False
                custom_visible = False
            elif selected_model in gguf_local:
                model_type = "gguf"
                api_visible = False
                custom_visible = False
            elif selected_model in mlx_local:
                model_type = "mlx"
                api_visible = False
                custom_visible = False
            elif selected_model == "사용자 지정 모델 경로 변경":
                model_type = "transformers"  # 기본값 설정 (필요 시 수정)
                api_visible = False
                custom_visible = True
            else:
                model_type = "transformers"
                api_visible = False
                custom_visible = False
                    
            return gr.update(value=model_type), gr.update(visible=api_visible), gr.update(visible=custom_visible)
                
        selected_model_sd.change(
            fn=update_model_type,
            inputs=[selected_model_sd],
            outputs=[model_type_sd, api_key_sd, custom_model_path_sd]
        )
                
        # 프롬프트 생성 버튼 클릭 시 함수 연결
        generate_prompt_btn.click(
            fn=generate_stable_diffusion_prompt_cached,
            inputs=[user_input_sd, selected_model_sd, model_type_sd, custom_model_path_sd, api_key_sd],
            outputs=prompt_output_sd
        )