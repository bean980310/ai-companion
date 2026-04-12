"""
Module providing a Translate Using LLM
"""

import os
import platform
import warnings
from typing import Any
import torch
from PIL import Image, ImageFile
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

try:
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from mlx_vlm.prompt_utils import apply_chat_template
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn(
            "mlx_vlm is not installed. Please install it to use MLX Multimodal Chat.",
            UserWarning,
        )
    else:
        pass

from ai_companion_llm_backend.base_handlers import BaseModelHandler

"""
Class for handling translation using LLM
"""


class LLMTranslateHandler(BaseModelHandler):
    def __init__(
        self,
        model_id: str,
        lora_model_id: str | None = None,
        use_langchain: bool = False,
        image_input: str | list[str] | Image.Image | ImageFile.ImageFile | Any | None = None,
        audio_input: str | list[str] | Any | None = None,
        video_input: str | list[str] | Any | None = None,
        **kwargs,
    ):
        super().__init__(
            model_id,
            lora_model_id,
            use_langchain,
            image_input,
            audio_input,
            video_input,
            **kwargs,
        )
        self.temperature = 0.3
        self.top_p = 0.9
        self.top_k = 40
        self.repetition_penalty = 1.1
        self.load_model()

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, device_map="auto")

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]]]], **kwargs):
        pass

    def _generate(self, prompt: str, src_lang: str, tgt_lang: str, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": src_lang,
                        "target_lang_code": tgt_lang,
                        "text": prompt,
                    }
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True).to(self.model.device, dtype=torch.bfloat16)
        input_len = len(inputs["input_ids"][0])
        self.max_tokens = input_len * 2
        self.generation_config = self.get_settings()

        generation = self.model.generate(**inputs, generation_config=self.generation_config)

        outputs = generation[0][input_len:]
        return self.processor.decode(outputs, skip_special_tokens=True)

    def get_settings(self):
        return GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )

    def load_template(self, messages: list[dict[str, str | list[dict[str, str]]]]):
        pass
