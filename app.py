# app.py
import warnings
warnings.filterwarnings("ignore", module="gradio")
warnings.filterwarnings("ignore", module="torchao")
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings('ignore', module='pydantic')

from src import os_name, arch, args, __version__
from src import gradio_ui
from src.start_app import initialize_app

import gradio as gr

with gr.Blocks(css_paths="html/css/style.css", title="AI Companion", fill_height=True) as demo:
    gradio_ui.demo.render()

if __name__=="__main__":
    print(f"AI Companion Version: {__version__}")
    print(f"Detected OS: {os_name}, Architecture: {arch}")
    if os_name == "Darwin" and arch == "x86_64":
        raise EnvironmentError("ERROR: AI Companion for Local Machines no longer supports Intel CPU-based Macs.\nIf you are using an Intel CPU-based Macs, we recommend that you consider migrating to an Apple Silicon Based Macs or a Windows PC or Linux machine with an Nvidia GPU environment. If you have difficulty migrating from an Intel CPU-based Macs, you can use a companion application that supports Intel CPU-based Macs instead.")
    initialize_app()
    demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_port=args.port, width=800, mcp_server=args.mcp_server, pwa=args.pwa)