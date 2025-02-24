import traceback
import os

from src import logger

class MlxModelHandler:
    def __init__(self, model_id, lora_model_id=None, local_model_path=None, lora_path=None, model_type="mlx"):
        self.model_dir = local_model_path or os.path.join("./models/llm", model_id)
        self.lora_model_dir = lora_path or (os.path.join("./models/llm/loras", lora_model_id) if lora_model_id else None)
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        from mlx_lm import load
        self.model, self.tokenizer = load(self.model_dir, adapter_path=self.lora_model_dir, tokenizer_config=self.get_eos_token())
    
    def generate_answer(self, history, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        text = self.tokenizer.apply_chat_template(
            conversation=history,
            tokenize=False,
            add_generation_prompt=True
        )
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k
        )
        response = generate(self.model, self.tokenizer, prompt=text, verbose=True, sampler=sampler, logits_processors=make_logits_processors(repetition_penalty=repetition_penalty), max_tokens=2048)
        
        return response
    
    def generate_chat_title(self, first_message: str)->str:
        from mlx_lm import generate
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
    
    def get_eos_token(self):
        if "llama-3" in self.model_dir.lower():
            return {"eos_token": "<|eot_id|>"}
        elif "qwen2" in self.model_dir.lower():
            return {"eos_token": "<|im_end|>"}