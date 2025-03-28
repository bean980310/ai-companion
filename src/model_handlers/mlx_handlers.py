import traceback
import os

from src import logger

from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler

class MlxCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="mlx"):
        super().__init__(model_id, lora_model_id)
        self.load_model()
        
    def load_model(self):
        from mlx_lm import load
        self.model, self.tokenizer = load(self.local_model_path, adapter_path=self.local_lora_model_path, tokenizer_config=self.get_eos_token())
        
    def generate_answer(self, history, **kwargs):
        from mlx_lm import generate
        text = self.load_template(history)
        sampler, logits_processors = self.get_settings(**kwargs)
        response = generate(self.model, self.tokenizer, prompt=text, verbose=True, sampler=sampler, logits_processors=logits_processors, max_tokens=2048)
        
        return response
    
    def get_settings(self, *, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0):
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k
        )
        logits_processors = make_logits_processors(repetition_penalty=repetition_penalty)
        return sampler, logits_processors
    
    def load_template(self, messages):
        return self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
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
        if "llama-3" in self.local_model_path.lower():
            return {"eos_token": "<|eot_id|>"}
        elif "qwen2" in self.local_model_path.lower():
            return {"eos_token": "<|im_end|>"}
        
class MlxVisionModelHandler(BaseVisionModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="mlx"):
        super().__init__(model_id, lora_model_id)
        self.load_model()

    def load_model(self):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self.model, self.processor = load(self.local_model_path, adapter_path=self.local_lora_model_path)
        self.config = load_config(self.local_model_path)

    def generate_answer(self, history, image_input=None, **kwargs):
        from mlx_vlm import generate
        image, formatted_prompt = self.load_template(history, image_input)
        temperature, top_k, top_p, repetition_penalty = self.get_settings(**kwargs)
        response = generate(self.model, self.processor, formatted_prompt, image, verbose=False, repetition_penalty=repetition_penalty, top_p=top_p, temp=temperature)

        return response

    def get_settings(self, *, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0):
        return temperature, top_k, top_p, repetition_penalty

    def load_template(self, messages, image_input):
        from mlx_vlm.prompt_utils import apply_chat_template
        if image_input:
            return image_input, apply_chat_template(
                processor=self.processor,
                config=self.config,
                prompt=messages,
                num_images=1 # <-- history 자체를 전달
            )
        else:
            return None, self.processor.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
    def generate_chat_title(self, first_message: str)->str:
        from mlx_vlm import generate
        prompt=(
            "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
            f"{first_message}\n\n"
            "Chat Title:"
        )
        logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        
        title_response=generate(self.model, self.processor, prompt=prompt, verbose=True, max_tokens=20)
        
        title=title_response.strip()
        logger.info(f"생성된 채팅 제목: {title}")
        return title