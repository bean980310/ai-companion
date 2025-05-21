# app.py

from src import os_name, arch, args
from src.start_app import initialize_app
from src.gradio_ui import demo

if __name__=="__main__":
    print(f"Detected OS: {os_name}, Architecture: {arch}")
    if os_name == "Darwin" and arch == "x86_64":
        raise EnvironmentError("ERROR: AI Companion for Local Machines no longer supports Intel CPU-based Macs.\nIf you are using an Intel CPU-based Macs, we recommend that you consider migrating to an Apple Silicon Based Macs or a Windows PC or Linux machine with an Nvidia GPU environment. If you have difficulty migrating from an Intel CPU-based Macs, you can use a companion application that supports Intel CPU-based Macs instead.")
    initialize_app()

    demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_port=args.port, width=800)