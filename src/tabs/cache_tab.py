import gradio as gr
from src.common.translations import _, translation_manager
from src.models.models import get_all_local_models
from src.common.utils import clear_all_model_cache
from src.tabs.main_tab import MainTab
import logging

main_tab=MainTab()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_cache_tab(model_dropdown, language_dropdown):    
    with gr.Tab(_("cache_tab_title")):
        with gr.Row():
            with gr.Column():
                refresh_button = gr.Button(_("refresh_model_list_button"))
                refresh_info = gr.Textbox(label=_("refresh_info_label"), interactive=False)
            with gr.Column():
                clear_all_btn = gr.Button(_("cache_clear_all_button"))
                clear_all_result = gr.Textbox(label=_("clear_all_result_label"), interactive=False)

        def refresh_model_list():
            """
            수동 새로고침 시 호출되는 함수.
            - 새로 scan_local_models()
            - DropDown 모델 목록 업데이트
            """
            # 새로 스캔
            new_local_models = get_all_local_models()
            # 새 choices: API 모델 + 로컬 모델 + 사용자 지정 모델 경로 변경
            api_models = [
                "gpt-3.5-turbo",
                "gpt-4o-mini",
                "gpt-4o"
                # 필요 시 추가
            ]
            local_models = new_local_models["transformers"] + new_local_models["gguf"] + new_local_models["mlx"]
            new_choices = api_models + local_models
            new_choices = list(dict.fromkeys(new_choices))
            new_choices = sorted(new_choices)  # 정렬 추가
            # 반환값:
            logger.info(_("refresh_model_list_button"))
            return gr.update(choices=new_choices), "모델 목록을 새로고침 했습니다."
            
        refresh_button.click(
            fn=refresh_model_list,
            inputs=[],
            outputs=[model_dropdown, refresh_info]
        )
        clear_all_btn.click(
            fn=clear_all_model_cache,
            inputs=[],
            outputs=clear_all_result
        )
        
        def change_language(selected_lang: str):
            """언어 변경 처리 함수"""
            lang_map = {
                "한국어": "ko",
                "日本語": "ja",
                "中文(简体)": "zh_CN",
                "中文(繁體)": "zh_TW",
                "English": "en"
            }
            lang_code = lang_map.get(selected_lang, "ko")
            translation_manager.set_language(lang_code)
            
            return [
                gr.update(value=_("refresh_model_list_button")),
                gr.update(label=_("refresh_info_label")),
                gr.update(value=_("cache_clear_all_button")),
                gr.update(label=_("clear_all_result_label"))
            ]

        language_dropdown.change(
            fn=change_language,
            inputs=[language_dropdown],
            outputs=[
                refresh_button,
                refresh_info,
                clear_all_btn,
                clear_all_result
            ]
        )