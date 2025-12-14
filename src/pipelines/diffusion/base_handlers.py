from abc import ABC, abstractmethod
from functools import partial
from typing import Any, List
import os
import json

class BaseModelHandler(ABC):
    def __init__(self, model_id: str, model_type: str, lora_weights: List[str] = None, **kwargs):
        self.model_id = model_id
        self.model_type = model_type
        self.lora_weights = lora_weights
        self.seed = int(kwargs.get('seed', 42))
        self.step = int(kwargs.get('step', 20))
        self.lora_text_weights = json.loads(str(kwargs.get("lora_text_weights_json", "")))
        self.lora_unet_weights = json.loads(str(kwargs.get("lora_unet_weights_json", "[]")))
        self.vae = str(kwargs.get("vae", ""))
        self.clip_skip = int(kwargs.get("clip_skip", 2))
        self.enable_clip_skip = bool(kwargs.get("enable_clip_skip", False))
        self.clip_g = bool(kwargs.get("clip_g", False))
        self.sampler = str(kwargs.get("sampler", ""))
        self.scheduler = str(kwargs.get("scheduler", ""))
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.batch_count = int(kwargs.get("batch_count", 1))
        self.cfg_scale = float(kwargs.get("cfg_scale", 7.5))
        self.random_seed = bool(kwargs.get("random_seed", False))
        self.width = int(kwargs.get("width", 512))
        self.height = int(kwargs.get("height", 512))

    @abstractmethod
    def load_model(self):
        pass

    
