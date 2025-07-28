from .... import logger
import traceback

from google import genai
from google.genai import types

class GoogleAIClientWrapper:
    def __init__(self, selected_model, api_key="None", **kwargs):
        self.model = selected_model
        self.api_key = api_key

        self.max_tokens=2048
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 1.0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

    def generate_answer(self, history, **kwargs):
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