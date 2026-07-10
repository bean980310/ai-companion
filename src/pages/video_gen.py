import gradio as gr
from src.common.translations import translation_manager, _
from src.common_blocks import create_page_header, get_language_code

with gr.Blocks() as demo:
    with gr.Column(elem_classes="tab-container") as video_gen_container:
        with gr.Row(elem_classes="chat-interface"):
            gr.Markdown("# Coming Soon!")

    # Language Change Event
    # def on_companion_language_change(selected_lang: str):
    #     lang_code = get_language_code(selected_lang)
    #     translation_manager.set_language(lang_code)
    #     return [
    #         gr.update(value=f"## {_('companion_title')}"),
    #         gr.update(label=_("language_select"), info=_("language_info")),
    #     ]


if __name__ == "__main__":
    demo.launch()
