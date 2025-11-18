from .... import logger
import traceback
from ..base_handlers import BaseAPIClientWrapper

from google import genai
from google.genai import types

from ..langchain_integrator import LangchainIntegrator

class GoogleAIClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        if self.use_langchain:
            self.load_model()

    def load_model(self):
        self.langchain_integrator = LangchainIntegrator(
            provider="google_genai",
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

            messages = [{"role": msg['role'], "content": msg['content']} for msg in history[:-1]]

            if self.image_input is not None:
                image = client.files.upload(file=self.image_input)
                new_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": history[-1]["content"]["text"]},
                        {"type": "image", "source": { "type": "base64", "mine_type": "image/jpeg", "data": image}}
                    ],
                }
                messages.append(new_message)
            else:
                messages.append([{"role": "user", "content": history[-1]["content"]}])

            config = types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                frequency_penalty=self.repetition_penalty,
                # presence_penalty=self.repetition_penalty,
            )
            logger.info(f"[*] Google API 요청: {messages}")
            response = client.models.generate_content(
                model=self.model,
                contents=messages,
                config=config
            )
            answer = response.text
            return answer