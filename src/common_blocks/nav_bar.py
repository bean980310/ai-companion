from dataclasses import dataclass

import gradio as gr
# from gradio_i18n import gettext as _


# from translations import i18n

from ..start_app import ui_component


@dataclass
class NavbarUIComponent:
    navbar: gr.Navbar = gr.Navbar(value=[("Chat", "chat"), ("Image Gen", "image"), ("Storyteller", "story"), ("TTS", "tts"), ("Translator", "translate"), ("Download", "download"), ("Settings", "settings")], render=False)

    @classmethod
    def create_navbar(cls, render=True):
        navbar = cls.navbar.render()

        if navbar.is_rendered and ui_component.navbar is not None:
            pass
        else:
            ui_component.navbar = navbar
