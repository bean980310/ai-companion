import random
import traceback
import os
import platform
import warnings
from typing import Any, Dict, List, Optional, Union, Iterator, Generator

import numpy as np
from PIL import Image, ImageFile

try:
    import mlx.core as mx
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("mlx is not installed. Please install it to use this library.", UserWarning)
    else:
        pass

try:
    from mlx_lm import load as mlx_lm_load
    from mlx_lm import generate as mlx_lm_generate
    from mlx_lm import stream_generate as mlx_lm_stream_generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from mlx_lm.generate import GenerationResponse
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("mlx_lm is not installed. Please install it to use MLX Chat.", UserWarning)
    else:
        pass

try:
    from mlx_vlm import load as mlx_vlm_load
    from mlx_vlm.utils import load_config
    from mlx_vlm import generate as mlx_vlm_generate
    from mlx_vlm import stream_generate as mlx_vlm_stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("mlx_vlm is not installed. Please install it to use MLX Multimodal Chat.", UserWarning)
    else:
        pass

from .langchain_integrator import LangchainIntegrator
from .logging import logger
from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler, BaseModelHandler

class MlxCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="mlx", use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)
        
        self.sampler = None
        self.logits_processors = None
        self.tokenizer_config = self.get_eos_token()

        if self.max_length > 0:
            self.max_tokens = self.max_length
        else:
            if "qwen3" in self.model_id.lower():
                if "instruct" in self.model_id.lower():
                    self.max_tokens = 16384
                else:
                    self.max_tokens = 32768

        self.set_seed(self.seed)
        self.load_model()
        
    def load_model(self):
        if self.use_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider="mlx",
                model_name=self.local_model_path,
                lora_model_name=self.local_lora_model_path,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                verbose=True,
                tokenizer_config=self.tokenizer_config
            )
        else:
            self.model, self.tokenizer = mlx_lm_load(self.local_model_path, adapter_path=self.local_lora_model_path, tokenizer_config=self.tokenizer_config)

    def generate_answer(self, history, **kwargs):
        if self.use_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            text = self.load_template(history)
            self.get_settings()
            response = mlx_lm_generate(self.model, self.tokenizer, prompt=text, verbose=True, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens, max_kv_size=2048)

            if "</think>" in response:
                _, response = response.split("</think>", 1)
            
            return response.strip()

    def get_settings(self):
        self.sampler = make_sampler(
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        )
        self.logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
    
    def load_template(self, messages):
        if "qwen3" in self.model_id.lower() and "instruct" not in self.model_id.lower() and "thinking" not in self.model_id.lower():
            return self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
        else:
            return self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
    def generate_chat_title(self, first_message: str) -> str:
        prompt=(
            "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
            f"{first_message}\n\n"
            "Chat Title:"
        )
        logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        
        title_response=mlx_lm_generate(self.model, self.tokenizer, prompt=prompt, verbose=True, max_tokens=20)
        
        title=title_response.strip()
        logger.info(f"생성된 채팅 제목: {title}")
        return title
        
    def get_eos_token(self):
        if "llama-3" in self.local_model_path.lower():
            return {"eos_token": "<|eot_id|>", "trust_remote_code": True}
        elif any(k in self.local_model_path.lower() for k in ["qwen2", "qwen3"]):
            return {"eos_token": "<|im_end|>", "trust_remote_code": True}
        elif any(k in self.local_model_path.lower() for k in ["mistral", "ministral", "mixtral", "magistral", "devstral"]):
            return {"eos_token": "</s>", "trust_remote_code": True}
        else:
            return {}

    def _generate_streaming(self, prompt_text: str) -> str | Generator[GenerationResponse, None, None]:
        """
        Generate text in chunks to avoid very long single-pass generations.
        Calls mlx_lm_generate repeatedly, appending the continuation each time.
        Stops if EOS or no progress is made.
        """

        generated_text = ""
        temp = ""
        for response in mlx_lm_stream_generate(self.model, self.tokenizer, prompt=prompt_text, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens):
            print(response.text, end='', flush=True)
            if "<think>" in response.text:
                while "</think>" not in response.text:
                    temp += ''.join(response.text)
                else:
                    temp += ''.join(response.text)
                    _, generated_text = temp.split("</think>", 1)
                    yield generated_text.strip()
            else:
                generated_text += ''.join(response.text)
                yield generated_text.strip()


        # return generated_text.strip()
    
    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        mx.random.seed(seed)
        

class MlxVisionModelHandler(BaseVisionModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="mlx", use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, image_input, **kwargs)

        self.sampler = None
        self.logits_processors = None
        self.tokenizer_config = kwargs.get("tokenizer_config", {})

        self.set_seed(self.seed)
        self.load_model()

    def load_model(self):
        if self.image_input:
            self.model, self.processor = mlx_vlm_load(self.local_model_path, adapter_path=self.local_lora_model_path, lazy=True)
            self.config = load_config(self.local_model_path)
        else:
            if self.use_langchain:
                self.langchain_integrator = LangchainIntegrator(
                    provider="mlx",
                    model_name=self.local_model_path,
                    lora_model_name=self.local_lora_model_path,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    verbose=True
                )
            # self.model, self.tokenizer = load(self.local_model_path, adapter_path=self.local_lora_model_path, tokenizer_config=self.tokenizer_config)
            else:
                self.model, self.tokenizer = mlx_lm_load(self.local_model_path, adapter_path=self.local_lora_model_path)

    def generate_answer(self, history, **kwargs):
        if self.image_input:
            formatted_prompt = self.load_template(history)
            response = mlx_vlm_generate(self.model, self.processor, formatted_prompt, self.image_input, verbose=True, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, repetition_penalty=self.repetition_penalty, max_tokens=self.max_tokens, max_kv_size=2048)

            return response.text.strip()
        else:
            if self.use_langchain:
                return self.langchain_integrator.generate_answer(history)
            else:
                text = self.load_template(history)
                self.get_settings()
                response = mlx_lm_generate(self.model, self.tokenizer, prompt=text, verbose=True, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens, max_kv_size=2048)

                return response.strip()

    def get_settings(self):
        self.sampler = make_sampler(
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        )
        self.logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
        # return temperature, top_k, top_p, repetition_penalty

    def load_template(self, messages):
        if self.image_input:
            return apply_chat_template(
                processor=self.processor,
                config=self.config,
                prompt=messages,
                num_images=1, # <-- history 자체를 전달
                add_generation_prompt=True,
            )
        else:
            return self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True
            )

    def generate_chat_title(self, first_message: str, image_input=None) -> str:
        prompt=(
            "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
            f"{first_message}\n\n"
            "Chat Title:"
        )
        logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        if self.image_input:
            title_response = mlx_vlm_generate(self.model, self.processor, prompt=prompt, verbose=True, max_tokens=20)
            title = title_response.text.strip()
        else:
            title_response = mlx_lm_generate(self.model, self.tokenizer, prompt=prompt, verbose=True, max_tokens=20)
            title = title_response.strip()

        logger.info(f"생성된 채팅 제목: {title}")
        return title
    
    def _generate_streaming_vision(self, prompt_text: str) -> str | Generator[str, None, None]:
        generated_text = ""
        temp = ""

        for response in mlx_vlm_stream_generate(self.model, self.processor, prompt_text, self.image_input, verbose=True, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, repetition_penalty=self.repetition_penalty, max_tokens=self.max_tokens):
            print(response, end='', flush=True)
            if "<think>" in response:
                while "</think>" not in response:
                    temp += ''.join(response)
                else:
                    temp += ''.join(response)
                    _, generated_text = temp.split("</think>", 1)
                    yield generated_text.strip()
            else:
                generated_text += ''.join(response)
                yield generated_text.strip()

    def _generate_streaming(self, prompt_text: str) -> str | Generator[GenerationResponse, None, None]:
        """
        Generate text in chunks to avoid very long single-pass generations.
        Calls mlx_lm_generate repeatedly, appending the continuation each time.
        Stops if EOS or no progress is made.
        """

        generated_text = ""
        temp = ""
        for response in mlx_lm_stream_generate(self.model, self.tokenizer, prompt=prompt_text, verbose=True, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens):
            print(response.text, end='', flush=True)
            if "<think>" in response.text:
                while "</think>" not in response.text:
                    temp += ''.join(response.text)
                else:
                    temp += ''.join(response.text)
                    _, generated_text = temp.split("</think>", 1)
                    yield generated_text.strip()
            else:
                generated_text += ''.join(response.text)
                yield generated_text.strip()

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        mx.random.seed(seed)
    
# class MlxLlama4ModelHandler(BaseModelHandler):
#     def __init__(self, model_id, lora_model_id=None, model_type="mlx", image_input=None, use_langchain: bool = True, **kwargs):
#         super().__init__(model_id, lora_model_id, use_langchain, **kwargs)
#         self.tokenizer = None
#         self.processor = None
#         self.model = None
#         self.image_input = image_input

#         self.sampler = None
#         self.logits_processors = None

#         self.load_model()
        
#     def load_model(self):
#         if self.image_input:
#             from mlx_vlm import load
#             from mlx_vlm.utils import load_config
#             self.model, self.processor = load(self.local_model_path, adapter_path=self.local_lora_model_path)
#             self.config = load_config(self.local_model_path)
#         else:
#             from mlx_lm import load
#             self.model, self.tokenizer = load(self.local_model_path, adapter_path=self.local_lora_model_path, tokenizer_config={"eos_token": "<|eot_id|>"})
            
#     def generate_answer(self, history, **kwargs):
#         image, formatted_prompt = self.load_template(history, image_input=self.image_input)
#         self.get_settings()

#         if image:
#             from mlx_vlm import generate
            
#             # temperature, top_k, top_p, repetition_penalty = MlxVisionModelHandler.get_settings(**kwargs)
#             response = generate(self.model, self.processor, formatted_prompt, image, verbose=False, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, repetition_penalty=self.repetition_penalty, max_tokens=self.max_tokens)

#             response = response[0].strip()

#         else:
#             from mlx_lm import generate
            
#             # sampler, logits_processors = self.get_settings()
#             response = generate(self.model, self.tokenizer, prompt=formatted_prompt, verbose=True, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens)

#             response = response.strip()
            
#         return response
            
#     def get_settings(self):
#         if self.image_input is not None:
#             from mlx_vlm.sample_utils import make_sampler, make_logits_processors
#         else:
#             from mlx_lm.sample_utils import make_sampler, make_logits_processors

#         self.sampler = make_sampler(
#             temp=self.temperature,
#             top_p=self.top_p,
#             top_k=self.top_k
#         )
#         self.logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
    
#     def load_template(self, messages, image_input=None):
#         if image_input:
#             from mlx_vlm.prompt_utils import apply_chat_template
#             return image_input, apply_chat_template(
#                 processor=self.processor,
#                 config=self.config,
#                 prompt=messages,
#                 num_images=1 # <-- history 자체를 전달
#             )
#         else:
#             return None, self.tokenizer.apply_chat_template(
#                 conversation=messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
            
#     def generate_chat_title(self, first_message: str, image_input=None)->str:
#         prompt=(
#             "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
#             f"{first_message}\n\n"
#             "Chat Title:"
#         )
#         logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        
#         if image_input:
#             from mlx_vlm import generate
#             title_response=generate(self.model, self.processor, prompt=prompt, verbose=True, max_tokens=20)
#         else:
#             from mlx_lm import generate
#             title_response=generate(self.model, self.tokenizer, prompt=prompt, verbose=True, max_tokens=20)
        
#         title=title_response.strip()
#         logger.info(f"생성된 채팅 제목: {title}")
#         return title
    
# class MlxQwen3ModelHandler(BaseCausalModelHandler):
#     def __init__(self, model_id, lora_model_id=None, model_type="mlx", use_langchain: bool = True, **kwargs):
#         super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

#         self.max_tokens=32768

#         self.enable_thinking = kwargs.get("enable_thinking", True)

#         self.sampler = None
#         self.logits_processors = None

#         self.load_model()
        
#     def load_model(self):
#         from mlx_lm import load
#         self.model, self.tokenizer = load(self.local_model_path, adapter_path=self.local_lora_model_path)
        
#     def generate_answer(self, history, **kwargs):
#         from mlx_lm import generate
#         text = self.load_template(history)
#         self.get_settings()
#         generated = generate(self.model, self.tokenizer, prompt=text, verbose=True, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens)
        
#         if "</think>" in generated:
#             _, response = generated.split("</think>", 1)
#         else:
#             response = generated  # Assign the entire generated text if no </think> tag is found
            
#         return response.strip()
    
#     def get_settings(self):
#         from mlx_lm.sample_utils import make_sampler, make_logits_processors
#         self.sampler = make_sampler(
#             temp=self.temperature,
#             top_p=self.top_p,
#             top_k=self.top_k
#         )
#         self.logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
    
#     def load_template(self, messages):
#         return self.tokenizer.apply_chat_template(
#             conversation=messages,
#             tokenize=False,
#             add_generation_prompt=True,
#             enable_thinking=self.enable_thinking
#         )
        
#     def generate_chat_title(self, first_message: str)->str:
#         from mlx_lm import generate
#         prompt=(
#             "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
#             f"{first_message}\n\n"
#             "Chat Title:"
#         )
#         logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        
#         title_response=generate(self.model, self.tokenizer, prompt=prompt, verbose=True, max_tokens=20)
        
#         title=title_response.strip()
#         logger.info(f"생성된 채팅 제목: {title}")
#         return title