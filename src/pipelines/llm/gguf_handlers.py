import llama_cpp
from llama_cpp import Llama # gguf 모델을 로드하기 위한 라이브러리
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

from . import LangchainIntegrator

import os

from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler

from src import logger

class GGUFCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="gguf", device='cpu', use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        self.n_gpu_layers = -1 if device != 'cpu' else 0
        self.sampler = None
        self.logits_processors = None
        
        self.load_model()
        
    def load_model(self):
        if self.use_langchain:
            self.langchain_integrator = LangchainIntegrator(
                backend_type="gguf",
                model_name=self.local_model_path,
                lora_model_name=self.local_lora_model_path,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                n_gpu_layers=self.n_gpu_layers,
                verbose=True
            )
        else:
            self.model = Llama(
                model_path=self.local_model_path,
                lora_path=self.local_lora_model_path,
                n_gpu_layers=self.n_gpu_layers,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
                logits_all=True
            )
        
    def generate_answer(self, history, **kwargs):
        if self.use_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            prompt = [{"role": msg['role'], "content": msg['content']} for msg in history]
            response = self.model.create_chat_completion(
                messages=prompt,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repeat_penalty=self.repetition_penalty
            )
            return response["choices"][0]["message"]["content"]

    def get_settings(self):
        pass
    
    def load_template(self, messages):
        pass
        
    def generate_chat_title(self, first_message: str)->str:
        prompt=(
            "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
            f"{first_message}\n\n"
            "Chat Title:"
        )
        
        logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        title_response=self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20
        )
        
        title = title_response["choices"][0]["message"]["content"]
        logger.info(f"생성된 채팅 제목: {title}")
        return title