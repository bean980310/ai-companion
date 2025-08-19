import os
import sys
import warnings
import platform

import random

try:
    import mlx.core as mx
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("MLX is not installed. Please install it to use Video features with MLX.", UserWarning)
    else:
        pass

import numpy as np
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool

class WanImage2VideoHandler:
    def __init__(self):
        self.video_path = None

    def process_image(self, image: Image.Image):
        # 이미지 처리 로직
        pass

    def save_video(self, output_path: str):
        if self.video_path:
            save_video(self.video_path, output_path)