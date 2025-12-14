import gradio as gr

from src.start_app import ui_component
from src.tabs.cache_tab import create_cache_tab
from src.tabs.util_tab import create_util_tab
from src.tabs.setting_tab_custom_model import create_custom_model_tab
from src.tabs.setting_tab_preset import create_system_preset_management_tab
from src.tabs.setting_tab_save_history import create_save_history_tab
from src.tabs.setting_tab_load_history import create_load_history_tab
from src.tabs.setting_tab_session_manager import create_session_management_tab
from src.tabs.device_setting import create_device_setting_tab
from src.main.chatbot import chat_bot
from src.common.translations import translation_manager, _
from src.common_blocks import create_page_header, get_language_code

with gr.Blocks() as demo:
    # Page Header with Language Selector
    page_header = create_page_header(page_title_key="settings_title")
    language_dropdown = page_header.language_dropdown

    with gr.Tabs():
        create_cache_tab()
        create_util_tab()
    
        with gr.Tab("Configuration"):
            with gr.Tabs():
                # 사용자 지정 모델 경로 설정 섹션
                create_custom_model_tab()
                create_system_preset_management_tab()
                
                # 프리셋 Dropdown 초기화 (Wire this event here)
                demo.load(
                    fn=chat_bot.initial_load_presets,
                    inputs=[],
                    outputs=[ui_component.text_preset_dropdown],
                    queue=False
                )                        
                
                create_save_history_tab()
                create_load_history_tab()
                create_session_management_tab()
                create_device_setting_tab()

    # Language Change Event
    def on_settings_language_change(selected_lang: str):
        lang_code = get_language_code(selected_lang)
        translation_manager.set_language(lang_code)
        return [
            gr.update(value=f"## {_('settings_title')}"),
            gr.update(label=_('language_select'), info=_('language_info'))
        ]

    language_dropdown.change(
        fn=on_settings_language_change,
        inputs=[language_dropdown],
        outputs=[page_header.title, language_dropdown]
    )

if __name__ == "__main__":
    demo.launch()
