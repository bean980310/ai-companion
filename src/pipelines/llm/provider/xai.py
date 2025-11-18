from .... import logger
import traceback
from ..base_handlers import BaseAPIClientWrapper

import xai_sdk

from ..langchain_integrator import LangchainIntegrator

class XAIClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        if self.use_langchain:
            self.load_model()

    def load_model(self):
        self.langchain_integrator = LangchainIntegrator(
            provider="xai",
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
            client = xai_sdk.Client(api_key=self.api_key)

            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] XAI API 요청: {messages}")
                
            answer = client.chat.create(
                model=self.model, 
                temperature=self.temperature, 
                max_tokens=self.max_tokens, 
                messages=messages,
                top_p=self.top_p,
                top_logprobs=self.top_k,
                frequency_penalty=self.repetition_penalty,
                # presence_penalty=self.repetition_penalty
            ).sample().content
            return answer
    