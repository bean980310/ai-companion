from abc import ABC, abstractmethod
from functools import partial
from typing import Any
import os

class BaseTaskHandler(ABC):
    def __init__(self, model_id, model_type, lora_weights, use_comfyui: bool = True, **kwargs):
        self.model_id = model_id
        self.model_type = model_type
        self.lora_weights = lora_weights
        self.use_comfyui = use_comfyui
        self.seed = kwargs.get('seed', 42)
        self.generation_step = kwargs.get('generation_step', 20)

    @abstractmethod
    def handle(self, request: Any) -> Any:
        pass