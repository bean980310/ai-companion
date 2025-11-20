# WIP: src/pipelines/llm/api/lmstudio.py

from typing import Any
import lmstudio as lms
from PIL import Image, ImageFile

SERVER_API_HOST = "localhost:1234"

lms.get_default_client(SERVER_API_HOST)

from ..base_handlers import BaseCausalModelHandler, BaseVisionModelHandler, BaseAPIClientWrapper
from ..langchain_integrator import LangchainIntegrator

class LMStudioIntegrator(BaseAPIClientWrapper):
    def __init__(self, model_id: str, selected_model: str = None, api_key: str = "not-needed", lora_model_id: str = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        self.causal = BaseCausalModelHandler(model_id, lora_model_id, use_langchain, **kwargs)
        self.vision = BaseVisionModelHandler(model_id, lora_model_id, use_langchain, image_input, **kwargs)
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        self.model_id: str = model_id
        self.lora_model_id: str | None = lora_model_id

        if selected_model is not None:
            self.local_model_path = selected_model
        else:
            self.local_model_path = self.model_id

        self.local_lora_model_path = self.lora_model_id
        self.image_input = image_input

        self.system_prompt = None
        self.user_message = None
        self.chat_history = None

        if self.max_length > 0:
            self.max_tokens = self.max_length
        else:
            self.max_tokens = 4096

        self.model: lms.LLM | Any | None = None
        self.chat: lms.Chat | Any | None = None
        self.client: lms.Client | Any | None = None
        self.image_handle = lms.FileHandle | Any | None = None

        self.load_model()

    def load_model(self):
        if self.use_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider="lmstudio",
                model_name=self.local_model_path,
                api_key="not-needed",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                verbose=True,
            )
        else:
            self.model = lms.llm(self.local_model_path, config={"seed": self.seed})

    def generate_answer(self, history, **kwargs):
        if self.use_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            for msg in history[:-1]:
                if msg["role"] == "system":
                    self.system_prompt = msg["content"]

            self.chat = lms.Chat(self.system_prompt)

            for msg in history[:-1]:
                if msg["role"] == "user":
                    self.chat.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    self.chat.add_assistant_response(msg["content"])

            self.user_message = history[-1]["content"]

            if self.image_input is not None:
                self.image_handle = lms.prepare_image(self.image_input)
                self.chat.add_user_message(self.user_message, images=[self.image_handle])
            else:
                self.chat.add_user_message(self.user_message)

            if self.enable_streaming is True:
                streamer = self.model.respond_stream(self.chat, config={
                    "temperature": self.temperature,
                    "topPSampling": self.top_p,
                    "topKSampling": self.top_k,
                    "repeatPenalty": self.repetition_penalty,
                    "maxTokens": self.max_tokens,
                })
                answer = ""
                for fragment in streamer:
                    print(fragment.content, end="", flush=True)
                    answer.join(fragment.content)

            else:
                response = self.model.respond(self.chat, config={
                    "temperature": self.temperature,
                    "topPSampling": self.top_p,
                    "topKSampling": self.top_k,
                    "repeatPenalty": self.repetition_penalty,
                    "maxTokens": self.max_tokens,
                })
                answer = response.content

            return answer.strip()