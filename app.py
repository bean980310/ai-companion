# app.py

from src import os_name, arch, args
from src.start_app import initialize_app
from src.gradio_ui import demo

if __name__=="__main__":
    print(f"Detected OS: {os_name}, Architecture: {arch}")
    if os_name == "Darwin" and arch == "x86_64":
        raise EnvironmentError("ERROR: Support and compatibility for Intel CPU Based Mac is discontinued.")
    initialize_app()

    demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_port=args.port, width=800)