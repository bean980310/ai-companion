from .... import logger
import traceback
from ..base_handlers import BaseAPIClientWrapper

from huggingface_hub import InferenceClient, InferenceEndpoint
from openai import OpenAI

import requests

from ..langchain_integrator import LangchainIntegrator

class OpenRouterClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        if self.use_langchain:
            self.load_model()

    def load_model(self):
        self.langchain_integrator = LangchainIntegrator(
            backend_type="openrouter",
            model_name=self.model,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            verbose=True
        )

    def generate_answer(self, history, **kwargs):
        if self.use_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )

            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] OpenRouter API 요청: {messages}")

            chat_completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body={
                    "top_k": self.top_k,
                    "repetition_penalty": self.repetition_penalty
                }
            )

            answer = chat_completion.choices[0].message.content
            return answer
