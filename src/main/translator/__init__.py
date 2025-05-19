from .translator import translate_interface
from .upload import upload_handler
import gradio as gr

LANGUAGES = [
    "English", 
    "한국어(Korean)", 
    "日本語(Japanese)", 
    "简体中文(Simp. Chinese)", 
    "Français(French)", 
    "Deutsche(German)", 
    "Español(Spanish)"
]

def create_translate_container():
    with gr.Column(elem_classes='tab-container') as translate_container:
        with gr.Row(elem_classes="model-container"):
            gr.Markdown("### Translate")
        with gr.Row(elem_classes="model-container"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        upload_file_lang = gr.Dropdown(
                            choices=list(dict.fromkeys(LANGUAGES)),
                            value=list(dict.fromkeys(LANGUAGES))[0],
                            label="File Language"
                        )
                        upload_file_btn = gr.Button(
                            "Upload File",
                            variant="primary",
                            elem_classes="send-button-alt"
                        )
                    upload_file_input = gr.File(
                        label="Upload File",
                        file_types=["text", "image"],
                        file_count="single"
                    )
                        
        with gr.Row(elem_classes="chat-interface"):
            with gr.Column():
                with gr.Row():
                    src_lang_dropdown=gr.Dropdown(
                        choices=list(dict.fromkeys(LANGUAGES)), 
                        value=list(dict.fromkeys(LANGUAGES))[0], 
                        label="Source Language"
                    )
                    tgt_lang_dropdown=gr.Dropdown(
                        choices=list(dict.fromkeys(LANGUAGES)), 
                        value=list(dict.fromkeys(LANGUAGES))[1], 
                        label="Target Language"
                    )
                with gr.Row():
                    src_textbox=gr.Textbox(
                        label="Source Text",
                        lines=10,
                        elem_classes='message-input'
                    )
                    translate_result=gr.Textbox(
                        label='Translate result',
                        lines=10,
                        elem_classes='message-output'
                    )
                with gr.Row():
                    translate_btn = gr.Button("Translate", variant="primary", elem_classes="send-button-alt")
                    
    translate_btn.click(
        fn=translate_interface,
        inputs=[src_textbox, src_lang_dropdown, tgt_lang_dropdown],
        outputs=[translate_result]
    )
    
    upload_file_btn.click(
        fn=upload_handler,
        inputs=[upload_file_input, upload_file_lang],
        outputs=src_textbox
    )
                    
    return translate_container


__all__ = ["translate_interface", "upload_handler", "LANGUAGES"]