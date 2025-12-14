import gradio as gr
# from src.main.tts import create_tts_side, create_tts_container
from src.main.tts import create_tts_side, get_tts_models
# Warning: create_tts_container might not exist, src/main/container.py had:
# with gr.Column(elem_classes='tab-container') as tts_container: ... gr.Markdown("# Coming Soon!")
# So I need to verify if create_tts_container exists or if I should implement the placeholder.

# From src/main/container.py:
# with gr.Column(elem_classes='tab-container') as tts_container:
#     with gr.Row(elem_classes="chat-interface"):
#         gr.Markdown("# Coming Soon!")

with gr.Blocks() as demo:
    get_tts_models()
    with gr.Sidebar():
        # From src/main/sidebar.py:
        # tts_side, tts_model_type_dropdown, tts_model_dropdown = create_tts_side()
        tts_side, tts_model_type_dropdown, tts_model_dropdown = create_tts_side()
        
    with gr.Column(elem_classes='tab-container') as tts_container:
        with gr.Row(elem_classes="chat-interface"):
            gr.Markdown("# Coming Soon!")

if __name__ == "__main__":
    demo.launch()
