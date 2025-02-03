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
        response = generate(self.model, self.tokenizer, prompt=text, verbose=True, max_tokens=2048)
        
        return response
    
    def generate_chat_title(self, first_message: str)->str:
        prompt=(
            "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
            f"{first_message}\n\n"
            "Chat Title:"
        )
        logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        
        title_response=generate(self.model, self.tokenizer, prompt=prompt, verbose=True, max_tokens=20)
        
        title=title_response.strip()
        logger.info(f"생성된 채팅 제목: {title}")
        return title