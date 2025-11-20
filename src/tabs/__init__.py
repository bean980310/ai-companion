import gradio as gr

from ..start_app import ui_component

from .cache_tab import create_cache_tab
from .util_tab import create_util_tab
from .setting_tab_custom_model import create_custom_model_tab
from .setting_tab_preset import create_system_preset_management_tab
from .setting_tab_save_history import create_save_history_tab
from .setting_tab_load_history import create_load_history_tab
from .setting_tab_session_manager import create_session_management_tab
from .device_setting import create_device_setting_tab

from ..main.chatbot import Chatbot


chat_bot = Chatbot()

def create_settings_popup(demo):
    with gr.Column(visible=False, elem_classes="settings-popup") as settings_popup:
        with gr.Row(elem_classes="popup-header"):
            gr.Markdown("## Settings")
            with gr.Column():
                close_settings_btn = gr.Button("✕", elem_classes="close-button")
            
        with gr.Tabs():
            create_cache_tab()
            create_util_tab()
        
            with gr.Tab("설정"):
                gr.Markdown("### 설정")

                with gr.Tabs():
                    # 사용자 지정 모델 경로 설정 섹션
                    create_custom_model_tab()
                    create_system_preset_management_tab()
                    # 프리셋 Dropdown 초기화
                    demo.load(
                        fn=chat_bot.initial_load_presets,
                        inputs=[],
                        outputs=[ui_component.preset_dropdown],
                        queue=False
                    )                        
                    create_save_history_tab()
                    create_load_history_tab()
                    setting_session_management_tab, existing_sessions_dropdown, current_session_display=create_session_management_tab()
                    device_tab, device_dropdown=create_device_setting_tab()
                    
        with gr.Row(elem_classes="popup-footer"):
            setting_cancel_btn = gr.Button("Cancel", variant="secondary")
            save_settings_btn = gr.Button("Save Changes", variant="primary")
            
        with gr.Column(visible=False, elem_classes="confirm-dialog") as save_confirm_dialog:
            gr.Markdown("### Save Changes?")
            gr.Markdown("Do you want to save the changes you made?")
            with gr.Row():
                confirm_no_btn = gr.Button("No", variant="secondary")
                confirm_yes_btn = gr.Button("Yes", variant="primary")
                
        with gr.Column(visible=False, elem_classes="confirm-dialog") as discard_confirm_dialog:
            gr.Markdown("### Change triggered.")
            gr.Markdown("Change triggered. Do you want to discard the changes you made?")
            with gr.Row():
                confirm_discard_no_btn = gr.Button("No", variant="secondary")
                confirm_discard_yes_btn = gr.Button("Yes", variant="primary")
                
    return settings_popup, close_settings_btn, setting_session_management_tab, existing_sessions_dropdown, current_session_display, device_tab, device_dropdown, setting_cancel_btn, save_settings_btn, save_confirm_dialog, confirm_no_btn, confirm_yes_btn, discard_confirm_dialog, confirm_discard_no_btn, confirm_discard_yes_btn