from .... import logger
import traceback
from ..base_handlers import BaseAPIClientWrapper

import requests

from ..langchain_integrator import LangchainIntegrator

class PerplexityClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model, api_key="None", use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        if self.use_langchain:
            self.load_model()

    def load_model(self):
        self.langchain_integrator = LangchainIntegrator(
            backend_type="perplexity",
            model_name=self.model,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            verbose=True,
        )

    def generate_answer(self, history, **kwargs):
        if self.use_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] Perplexity API 요청: {messages}")
                
            url = "https://api.perplexity.ai/chat/completions"
            payload = { 
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "frequency_penalty": self.repetition_penalty,
                "presence_penalty": self.repetition_penalty,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.request("POST", url, json=payload, headers=headers)
            answer = response.text
            return answer