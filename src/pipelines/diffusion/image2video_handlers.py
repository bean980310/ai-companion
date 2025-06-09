import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from typing import List
import pandas as pd
import random
import traceback
import os

from src.api.comfy_api import ComfyUIClient

from workflows.load_workflow import load_image2video_wan_workflow

from src import logger

# def image2video_wan(
#     positive_prompt: str,
#     negative_prompt: str,
#     generation_step: int,
#     diffusion_model: str,
#     diffusion_model_type: str,
#     loras: List[str],
#     vae: str,
#     sampler: str,
#     scheduler: str,
#     cfg_scale: float,
#     seed: int,
#     random_seed: bool,
#     image_input: str,
#     denoise_strength: float,
#     lora_text_weights_json: str,
#     lora_unet_weights_json: str,
    
# ):