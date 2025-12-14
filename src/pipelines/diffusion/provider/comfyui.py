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

from PIL import Image, ImageOps, ImageFile, ImageMode

from src.api.comfy_api import ComfyUIClient
from workflows.load_workflow import load_img2img_workflow, load_img2img_sdxl_workflow, load_img2img_sdxl_with_refiner_workflow, load_img2img_workflow_clip_skip, load_img2img_sdxl_workflow_clip_skip, load_img2img_sdxl_with_refiner_workflow_clip_skip

from workflows.load_workflow import load_inpaint_workflow, load_inpaint_sdxl_workflow, load_inpaint_sdxl_with_refiner_workflow, load_inpaint_workflow_clip_skip, load_inpaint_sdxl_workflow_clip_skip, load_inpaint_sdxl_with_refiner_workflow_clip_skip

from workflows.load_workflow import load_txt2img_workflow, load_txt2img_sdxl_workflow, load_txt2img_sdxl_with_refiner_workflow, load_txt2img_workflow_clip_skip, load_txt2img_sdxl_workflow_clip_skip, load_txt2img_sdxl_with_refiner_workflow_clip_skip

from src import logger

class ComfyUIIntegratorBase:
    def __init__(self, positive_prompt: str, negative_prompt: str, width: int, height: int, model: str, model_type: str, **kwargs):
        self.prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.style = str(kwargs.get("style", ""))
        self.step = int(kwargs.get("step", 20))
        self.width = width
        self.height = height
        self.model = model
        self.model_type = model_type
        self.loras = List[str](kwargs.get("loras", []))
        self.vae = str(kwargs.get("vae", ""))
        self.clip_skip = int(kwargs.get("clip_skip", -2))
        self.enable_clip_skip = bool(kwargs.get("enable_clip_skip", False))
        self.clip_g = bool(kwargs.get("clip_g", False))
        self.sampler = str(kwargs.get("sampler", "euler"))
        self.scheduler = str(kwargs.get("scheduler", "normal"))
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.batch_count = int(kwargs.get("batch_count", 1))
        self.cfg_scale = float(kwargs.get("cfg_scale", 7.5))
        self.seed = int(kwargs.get("seed", 42))
        self.random_seed = bool(kwargs.get("random_seed", False))
        self.lora_text_weights_json = str(kwargs.get("lora_text_weights_json", ""))
        self.lora_unet_weights_json = str(kwargs.get("lora_unet_weights_json", ""))

class ComfyUIIntegratorText2Image(ComfyUIIntegratorBase):
    def __init__(self, positive_prompt: str, negative_prompt: str, width: int, height: int, model: str, model_type: str, **kwargs):
        super().__init__(positive_prompt, negative_prompt, width, height, model, model_type, **kwargs)