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

with gr.Blocks() as demo:
    gr.Markdown("# Settings")
    
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

if __name__ == "__main__":
    demo.launch()
