from .... import logger
import traceback
from ..base_handlers import BaseAPIClientWrapper

import openai

from ..langchain_integrator import LangchainIntegrator

class OpenAIClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        if self.use_langchain:
            self.load_model()

    def load_model(self):
        self.langchain_integrator = LangchainIntegrator(
            backend_type="openai",
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
            openai.api_key = self.api_key

            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] OpenAI API 요청: {messages}")

            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # top_logprobs=self.top_k,
                frequency_penalty=self.repetition_penalty,
                # presence_penalty=self.repetition_penalty,
                top_p=self.top_p,
            )
            answer = response.choices[0].message["content"]
            return answer

# class OpenAILangChainIntegration:
#     def __init__(self, api_key=None, model="gpt-4o-mini", temperature=0.6, top_p=0.9, top_k=40, repetition_penalty=1.0):
#         self.api_key = api_key
#         if not api_key:
#             logger.error("OpenAI API Key가 missing.")
#             raise "OpenAI API Key가 필요합니다."
        
#         self.model = model
#         self.llm = ChatOpenAI(
#             model=model,
#             temperature=temperature,
#             top_p=top_p,
#             top_logprobs=top_k,
#             frequency_penalty=repetition_penalty,
#             api_key=api_key,
#             max_tokens=2048
#         )