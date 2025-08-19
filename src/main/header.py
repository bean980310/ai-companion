import gradio as gr
from ..common.translations import translation_manager, _
from ..common.default_language import default_language
from dataclasses import dataclass
from ..start_app import ui_component

@dataclass
class HeaderUIComponent:
    title: gr.Markdown = None
    settings_button: gr.Button = None
    language_dropdown: gr.Dropdown = None

    @classmethod
    def create_header_container(cls):
        with gr.Row(elem_classes="header-container"):
            with gr.Column(scale=3):
                title = gr.Markdown(f"## {_('main_title')}", elem_classes="title")
                gr.Markdown("### Beta Release")
            with gr.Column(scale=1):
                settings_button = gr.Button("⚙️", elem_classes="settings-button")
            with gr.Column(scale=1):
                language_dropdown = gr.Dropdown(
                    label=_('language_select'),
                    choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
                    value=translation_manager.get_language_display_name(default_language),
                    interactive=True,
                    info=_('language_info'),
                    container=False,
                    elem_classes="custom-dropdown"
                )
        
        ui_component.title = title
        ui_component.settings_button = settings_button
        ui_component.language_dropdown = language_dropdown
        
        return cls(title, settings_button, language_dropdown)
    

