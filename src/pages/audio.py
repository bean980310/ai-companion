import gradio as gr
from src.main.tts import create_tts_side, get_tts_models
from src.common.translations import translation_manager, _
from src.common_blocks import create_page_header, get_language_code

with gr.Blocks() as demo:
    # Page Header with Language Selector
    # page_header = create_page_header(page_title_key="tts_title")
    # language_dropdown = page_header.language_dropdown

    get_tts_models()
    with gr.Sidebar():
        tts_side, tts_model_type_dropdown, tts_model_dropdown = create_tts_side()

    with gr.Column(elem_classes='tab-container') as tts_container:
        with gr.Row(elem_classes="chat-interface"):
            gr.Markdown("# Coming Soon!")

    # Language Change Event
    def on_tts_language_change(selected_lang: str):
        lang_code = get_language_code(selected_lang)
        translation_manager.set_language(lang_code)
        return [
            gr.update(value=f"## {_('tts_title')}"),
            gr.update(label=_('language_select'), info=_('language_info'))
        ]

    # language_dropdown.change(
    #     fn=on_tts_language_change,
    #     inputs=[language_dropdown],
    #     outputs=[page_header.title, language_dropdown]
    # )

if __name__ == "__main__":
    demo.launch()
