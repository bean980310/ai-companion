import traceback
from perplexity import Perplexity

from .... import logger
from ..base_handlers import BaseAPIClientWrapper
from ..langchain_integrator import LangchainIntegrator

class PerplexityClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        if self.use_langchain:
            self.load_model()

    def load_model(self):
        self.langchain_integrator = LangchainIntegrator(
            provider="perplexity",
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
            client = Perplexity(api_key=self.api_key)

            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] Perplexity API 요청: {messages}")
                
            completion = client.chat.completions.create(
                messages=messages,
                model=self.model,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                frequency_penalty=self.repetition_penalty,
                # presence_penalty=self.repetition_penalty,
                temperature=self.temperature,
            )
            answer = completion.choices[0].message.content
            return answer