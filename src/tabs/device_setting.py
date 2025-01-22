import torch
import platform
import logging
import gradio as gr
from typing import Tuple

from src.models.models import get_default_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_device(selection):
    """
    Sets the device based on user selection.
    - Auto: Automatically detect the best device.
    - CPU: Force CPU usage.
    - GPU: Detect and use CUDA or MPS based on available hardware.
    """
    if selection == "Auto (Recommended)":
        device = get_default_device()
    elif selection == "CPU":
        device = "cpu"
    elif selection == "GPU":
        if torch.cuda.is_available():
            device = "cuda"
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            device = "mps"
        else:
            return gr.update(value="❌ GPU가 감지되지 않았습니다. CPU로 전환됩니다."), "cpu"
    else:
        device = "cpu"
                
    device_info_message = f"선택된 장치: {device.upper()}"
    logger.info(device_info_message)
    return gr.update(value=device_info_message), device

def create_device_setting_tab(default_device)->Tuple[gr.Tab, gr.Dropdown]:
    device_setting=gr.Tab("장치 설정")
    with device_setting:
        device_dropdown = gr.Dropdown(
            label="사용할 장치 선택",
            choices=["Auto (Recommended)", "CPU", "GPU"],
            value="Auto (Recommended)",
            info="자동 설정을 사용하면 시스템에 따라 최적의 장치를 선택합니다."
        )
        device_info = gr.Textbox(
            label="장치 정보",
            value=f"현재 기본 장치: {default_device.upper()}",
            interactive=False
        )
                        
        device_dropdown.change(
            fn=set_device,
            inputs=[device_dropdown],
            outputs=[device_info, gr.State(default_device)],
            queue=False
        )
        
    return device_setting, device_dropdown