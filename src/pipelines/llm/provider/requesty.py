from .... import logger
import traceback
from ..base_handlers import BaseAPIClientWrapper

from huggingface_hub import InferenceClient
import openai

import requests

from ..langchain_integrator import LangchainIntegrator

class OpenRouterClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        if self.use_langchain:
            self.load_model()

    def load_model(self):
        self.langchain_integrator = LangchainIntegrator(
            provider="requesty",
            model_name=self.model,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            verbose=True
        )

    def generate_answer(self, history, **kwargs):
        if self.use_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            client = InferenceClient(
                base_url="https://router.requesty.ai/v1",
                api_key=self.api_key
            )

            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] Requesty API 요청: {messages}")

            chat_completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            answer = chat_completion.choices[0].message.content
            return answer
