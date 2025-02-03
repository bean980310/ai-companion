import logging
import traceback
import os
from src.common.utils import make_local_dir_name

from mlx_lm import load, generate

logger = logging.getLogger(__name__)

class MlxModelHandler:
    def __init__(self, model_id, local_model_path=None, model_type="mlx"):
        self.model_dir = local_model_path or os.path.join("./models/llm", model_type, make_local_dir_name(model_id))
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        self.model, self.tokenizer = load(self.model_dir, tokenizer_config={"eos_token": "<|im_end|>"})
    
    def generate_answer(self, history):
        text = self.tokenizer.apply_chat_template(
            conversation=history,
            tokenize=False,
            add_generation_prompt=True
        )
        response = generate(self.model, self.tokenizer, prompt=text, verbose=True, top_p=0.8, max_new_tokens=2048)
        
        return response