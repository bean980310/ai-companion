from .... import logger
import traceback
from ..base_handlers import BaseAPIClientWrapper

import anthropic

from ..langchain_integrator import LangchainIntegrator

class AnthropicClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model, api_key="None", use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)
        
        if self.use_langchain:
            self.load_model()

    def load_model(self):
        self.langchain_integrator = LangchainIntegrator(
            backend_type="anthropic",
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
            client = anthropic.Client(api_key=self.api_key)

            # Anthropic 메시지 형식으로 변환
            messages = []
            for msg in history:
                if msg["role"] == "system":
                    system = msg["content"]
                    continue  # Claude API는 시스템 메시지를 별도로 처리하지 않음
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
                    
            logger.info(f"[*] Anthropic API 요청: {messages}")
                    
            response = client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                # frequency_penalty=repetition_penalty,
                max_tokens=self.max_tokens,
            )
            answer = response.content[0].text
            return answer