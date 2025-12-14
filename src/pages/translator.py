import gradio as gr
from src.main.translator import create_translate_container
from src.common.translations import translation_manager, _
from src.common_blocks import create_page_header, get_language_code

with gr.Blocks() as demo:
    # Page Header with Language Selector
    page_header = create_page_header(page_title_key="translator_title")
    language_dropdown = page_header.language_dropdown

    with gr.Sidebar():
        gr.Markdown("### Under Construction")

    translate_container = create_translate_container()

    # Language Change Event
    def on_translator_language_change(selected_lang: str):
        lang_code = get_language_code(selected_lang)
        translation_manager.set_language(lang_code)
        return [
            gr.update(value=f"## {_('translator_title')}"),
            gr.update(label=_('language_select'), info=_('language_info'))
        ]

    language_dropdown.change(
        fn=on_translator_language_change,
        inputs=[language_dropdown],
        outputs=[page_header.title, language_dropdown]
    )

if __name__ == "__main__":
    demo.launch()
