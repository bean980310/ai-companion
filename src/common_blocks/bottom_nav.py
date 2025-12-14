import gradio as gr
# from gradio_i18n import gettext as _


# from translations import i18n as _

from ..common.translations import translation_manager, _
from ..common.default_language import default_language
from dataclasses import dataclass
from ..start_app import ui_component
from .. import __version__

# @dataclass
# class HeaderUIComponent:
#     title: gr.Markdown = None
#     settings_button: gr.Button = None
#     language_dropdown: gr.Dropdown = None
#     navbar: gr.Navbar = None

#     @classmethod
#     def create_header_container(cls, render=False):
#         with gr.Row(elem_classes="header-container", scale=1, render=render) as head:
#             with gr.Column(scale=3,
#                     render=render):
#                 title = gr.Markdown(f"## {_('main_title')}", elem_classes="title",
#                     render=render)
#                 gr.Markdown("### Beta Release",
#                     render=render)
#             with gr.Column(scale=1,
#                     render=render):
#                 settings_button = gr.Button("⚙️", elem_classes="settings-button",
#                     render=render)
#             with gr.Column(scale=1,
#                     render=render):
#                 language_dropdown = gr.Dropdown(
#                     label=_('language_select'),
#                     choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
#                     value=translation_manager.get_language_display_name(default_language),
#                     interactive=True,
#                     info=_('language_info'),
#                     container=False,
#                     elem_classes="language-selector",
#                     render=render
#                 )
        
#         # navbar = gr.Navbar(main_page_name="Chat")
        
#         if ui_component.title is None:
#             ui_component.title = title
#         if ui_component.settings_button is None:
#             ui_component.settings_button = settings_button
#         if ui_component.language_dropdown is None:
#             ui_component.language_dropdown = language_dropdown
        
#         return cls(title=title, settings_button=settings_button)

#     @classmethod
#     def create_navbar(cls):
#         navbar = gr.Navbar(value=[("Chat", "chat"), ("Image Gen", "image"), ("Storyteller", "story"), ("TTS", "tts"), ("Translator", "translate"), ("Download", "download"), ("Settings", "settings")])
#         if ui_component.navbar is None:
#             ui_component.navbar = navbar

@dataclass
class BottomNavUIComponent:
    version: gr.Markdown = None
    language_dropdown: gr.Dropdown = None

    @classmethod
    def create_bottom_bar(cls):
        # with gr.Row(render=render) as bottom:
        version: gr.Markdown = gr.Markdown(f"Version: {__version__}", render=False)
        language_dropdown: gr.Dropdown = gr.Dropdown(
            label=_('language_select'),
            choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
            value=translation_manager.get_language_display_name(default_language),
            interactive=True,
            info=_('language_info'),
            container=False,
            elem_classes="language-selector",
            render=False
        )

        
        ui_component.language_dropdown = language_dropdown
            
        return cls(language_dropdown=language_dropdown)
        # return cls.navbar