import gradio as gr
from gradio_gpu_monitor import GPUMonitor


def create_custom_gpu_monitor():
    with gr.Tab("Info"):
        gpu_widget = GPUMonitor(
            update_interval=2500,
            show_last_updated=False,
            label="Hardware Status",
            visible=True,
        )
