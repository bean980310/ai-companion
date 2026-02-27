import gradio as gr
from src.tabs.download_tab import create_download_tab
from src.common.translations import translation_manager, _
from src.common_blocks import create_page_header, get_language_code

with gr.Blocks() as demo:
    # Page Header with Language Selector
    # page_header = create_page_header(page_title_key="download_title")
    # language_dropdown = page_header.language_dropdown

    create_download_tab()

    # Language Change Event
    def on_download_language_change(selected_lang: str):
        lang_code = get_language_code(selected_lang)
        translation_manager.set_language(lang_code)
        return [
            gr.update(value=f"## {_('download_title')}"),
            gr.update(label=_('language_select'), info=_('language_info'))
        ]

    # language_dropdown.change(
    #     fn=on_download_language_change,
    #     inputs=[language_dropdown],
    #     outputs=[page_header.title, language_dropdown]
    # )

if __name__ == "__main__":
    demo.launch()
