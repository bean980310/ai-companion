import logging
import traceback
import os
from src.common.utils import make_local_dir_name

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

logger = logging.getLogger(__name__)

class MlxVisionHandler:
    def __init__(self, model_id, lora_model_id=None, local_model_path=None, lora_path=None, model_type="mlx"):
        self.model_dir = local_model_path or os.path.join("./models/llm", model_id)
        self.lora_model_dir = lora_path or (os.path.join("./models/llm/loras", lora_model_id) if lora_model_id else None)
        self.processor = None
        self.config = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        self.model, self.processor = load(self.model_dir, adapter_path=self.lora_model_dir)
        self.config = load_config(self.model_dir)
        
    def generate_answer(self, history, *image_inputs, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0):
        # 1) prompt 문자열 생성 대신 history 그대로 사용
        # prompt = self.history_to_prompt(history)  # 주석 처리 혹은 삭제
        images = image_inputs if image_inputs else []
        if image_inputs:
            # 2) 'prompt' 대신 'conversation=history' 형태로 전달
            formatted_prompt = apply_chat_template(
                processor=self.processor,
                config=self.config,
                prompt=history,   # <-- history 자체를 전달
                num_images=len(images)
            )
            output = generate(self.model, self.processor, formatted_prompt, images, verbose=False, repetition_penalty=repetition_penalty, top_p=top_p, temp=temperature)
            return output
        else:
            formatted_prompt = apply_chat_template(
                processor=self.processor,
                config=self.config,
                prompt=history,   # <-- history 자체를 전달
                num_images=0
            )
            output = generate(self.model, self.processor, formatted_prompt, images=None, verbose=False, repetition_penalty=0.8, top_p=0.9, temp=0.7)
            return output