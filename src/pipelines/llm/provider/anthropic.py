import traceback
import anthropic

from .... import logger
from ..base_handlers import BaseAPIClientWrapper
from ..langchain_integrator import LangchainIntegrator

class AnthropicClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)
        
        if self.use_langchain:
            self.load_model()

    def load_model(self):
        self.langchain_integrator = LangchainIntegrator(
            provider="anthropic",
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
            for msg in history[:-1]:
                if msg["role"] == "system":
                    system = msg["content"]
                    continue  # Claude API는 시스템 메시지를 별도로 처리하지 않음
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            if self.image_input is not None:
                image = self.encode_image(self.image_input)
                new_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": history[-1]["content"]["text"]},
                        {"type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": image}}
                    ],
                }
                messages.append(new_message)
            else:
                messages.append([{"role": "user", "content": history[-1]["content"]}])
                    
            logger.info(f"[*] Anthropic API 요청: {messages}")

            if self.enable_streaming is True:
                answer = ""
                with client.messages.stream(
                    model=self.model,
                    system=system,
                    messages=messages,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    # frequency_penalty=repetition_penalty,
                    max_tokens=self.max_tokens,
                ) as stream:
                    for text in stream.text_stream:
                        print(text, end="", flush=True)
                        answer.join(text)

            else:
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