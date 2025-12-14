import gradio as gr
from src.tabs.download_tab import create_download_tab

with gr.Blocks() as demo:    
    # No sidebar for download tab in original code (or empty sidebar)
    create_download_tab()

if __name__ == "__main__":
    demo.launch()
