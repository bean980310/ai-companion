# app.py
import warnings
warnings.filterwarnings("ignore", module="gradio")
warnings.filterwarnings("ignore", module="torchao")
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings('ignore', module='pydantic')

import gradio as gr

from src import os_name, arch, args, __version__
from src.start_app import initialize_app
from src import gradio_ui

with gr.Blocks(title="AI Companion", fill_height=True, fill_width=True, css_paths="html/css/style.css") as demo:
    gradio_ui.demo.render()

if __name__=="__main__":
    print(f"AI Companion Version: {__version__}")
    print(f"Detected OS: {os_name}, Architecture: {arch}")
    if os_name == "Darwin" and arch == "x86_64":
        raise EnvironmentError("ERROR: AI Companion for Local Machines no longer supports Intel CPU-based Macs.\nIf you are using an Intel CPU-based Macs, we recommend that you consider migrating to an Apple Silicon Based Macs or a Windows PC or Linux machine with an Nvidia GPU environment. If you have difficulty migrating from an Intel CPU-based Macs, you can use a companion application that supports Intel CPU-based Macs instead.")
    initialize_app()
    if args.listen:
        host="0.0.0.0"
    else:
        host="127.0.0.1"
    
    demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_name=host, server_port=args.port, mcp_server=args.mcp_server, pwa=args.pwa)