import gradio as gr
from src.main.translator import create_translate_container

with gr.Blocks() as demo:
    with gr.Sidebar():
        # From src/main/sidebar.py:
        # with gr.Column() as translate_side: ... "Under Construction"
        gr.Markdown("### Under Construction")
        
    translate_container = create_translate_container()

if __name__ == "__main__":
    demo.launch()
