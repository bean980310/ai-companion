import gradio as gr
from src.main.header import HeaderUIComponent
from src.start_app import app_state, ui_component, initialize_speech_manager
from src.common.translations import translation_manager, _
from src.common_blocks import create_page_header, get_language_code

with gr.Blocks() as demo:
    page_header = create_page_header(page_title_key="main_title")
    language_dropdown = page_header.language_dropdown

    def on_header_language_change(selected_lang: str):
        lang_code = get_language_code(selected_lang)
        translation_manager.set_language(lang_code)

        return [
            gr.update(value=f"## {_('main_title')}"),
            gr.update(label=_('language_select'), info=_('language_info'))
        ]

if __name__ == "__main__":
    demo.launch()
