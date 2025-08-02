from .... import logger
import traceback
from ..base_handlers import BaseAPIClientWrapper

from google import genai
from google.genai import types

from . import LangchainIntegrator

class GoogleAIClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model, api_key="None", use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        if self.use_langchain:
            self.load_model()

    def load_model(self):
        self.langchain_integrator = LangchainIntegrator(
            backend_type="google_genai",
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
            client = genai.Client(api_key=self.api_key)

            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            config = types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                frequency_penalty=self.repetition_penalty,
                presence_penalty=self.repetition_penalty,
            )
            logger.info(f"[*] Google API 요청: {messages}")
            response = client.models.generate_content(
                model=self.model,
                contents=messages,
                config=config
            )
            answer = response.text
            return answer