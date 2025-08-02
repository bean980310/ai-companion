import traceback
import os

from typing import Any, Dict, List, Optional, Union, Iterator

from . import LangchainIntegrator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser, JsonOutputParser, XMLOutputParser, PydanticOutputParser
from langchain.output_parsers import RetryOutputParser, RetryWithErrorOutputParser
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory, SQLChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

from src import logger

from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler, BaseModelHandler

class MlxCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="mlx", use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)
        
        self.sampler = None
        self.logits_processors = None

        self.load_model()
        
    def load_model(self):
        from mlx_lm import load
        # from mlx_lm import load
        if self.use_langchain:
            self.langchain_integrator = LangchainIntegrator(
                backend_type="mlx",
                model_name=self.local_model_path,
                lora_model_name=self.local_lora_model_path,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                verbose=True
            )

        else:
            self.model, self.tokenizer = load(self.local_model_path, adapter_path=self.local_lora_model_path)
        
    def generate_answer(self, history, **kwargs):
        from mlx_lm import generate
        if self.use_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            text = self.load_template(history)
            self.get_settings()
            response = generate(self.model, self.tokenizer, prompt=text, verbose=True, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens)
            
            return response.strip()

    def get_settings(self):
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        self.sampler = make_sampler(
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        )
        self.logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
    
    def load_template(self, messages):
        return self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
    def generate_chat_title(self, first_message: str)->str:
        from mlx_lm import generate
        prompt=(
            "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
            f"{first_message}\n\n"
            "Chat Title:"
        )
        logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        
        title_response=generate(self.model, self.tokenizer, prompt=prompt, verbose=True, max_tokens=20)
        
        title=title_response.strip()
        logger.info(f"생성된 채팅 제목: {title}")
        return title
        
    def get_eos_token(self):
        if "llama-3" in self.local_model_path.lower():
            return {"eos_token": "<|eot_id|>"}
        elif "qwen2" in self.local_model_path.lower() or "qwen3" in self.local_model_path.lower():
            return {"eos_token": "<|im_end|>"}
        elif "mistral" or "ministral" or "mixtral" in self.local_model_path.lower():
            return {"eos_token": "</s>"}
        else:
            return {"eos_token": self.tokenizer._eos_token_ids }
        
class MlxVisionModelHandler(BaseVisionModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="mlx", use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        self.sampler = None
        self.logits_processors = None

        self.load_model()

    def load_model(self):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self.model, self.processor = load(self.local_model_path, adapter_path=self.local_lora_model_path)
        self.config = load_config(self.local_model_path)

    def generate_answer(self, history, image_input=None, **kwargs):
        from mlx_vlm import load, generate, stream_generate
        image, formatted_prompt = self.load_template(history, image_input)
        self.get_settings()
        response = stream_generate(self.model, self.processor, formatted_prompt, image, verbose=False, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, repetition_penalty=self.repetition_penalty, max_tokens=self.max_tokens)

        return response[0].strip()

    def get_settings(self):
        from mlx_vlm.sample_utils import make_sampler, make_logits_processors
        self.sampler = make_sampler(
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        )
        self.logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
        # return temperature, top_k, top_p, repetition_penalty

    def load_template(self, messages, image_input):
        from mlx_vlm.prompt_utils import apply_chat_template
        if image_input:
            return image_input, apply_chat_template(
                processor=self.processor,
                config=self.config,
                prompt=messages,
                num_images=1, # <-- history 자체를 전달
                add_generation_prompt=True,
            )
        else:
            return None, self.processor.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True
            )

    def generate_chat_title(self, first_message: str, image_input=None)->str:
        from mlx_vlm import generate
        prompt=(
            "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
            f"{first_message}\n\n"
            "Chat Title:"
        )
        logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        
        title_response=generate(self.model, self.processor, prompt=prompt, verbose=True, max_tokens=20)
        
        title=title_response.strip()
        logger.info(f"생성된 채팅 제목: {title}")
        return title
    
class MlxLlama4ModelHandler(BaseModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="mlx", image_input=None, use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.image_input = image_input

        self.sampler = None
        self.logits_processors = None

        self.load_model()
        
    def load_model(self):
        if self.image_input:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            self.model, self.processor = load(self.local_model_path, adapter_path=self.local_lora_model_path)
            self.config = load_config(self.local_model_path)
        else:
            from mlx_lm import load
            self.model, self.tokenizer = load(self.local_model_path, adapter_path=self.local_lora_model_path, tokenizer_config={"eos_token": "<|eot_id|>"})
            
    def generate_answer(self, history, **kwargs):
        image, formatted_prompt = self.load_template(history, image_input=self.image_input)
        self.get_settings()

        if image:
            from mlx_vlm import generate
            
            # temperature, top_k, top_p, repetition_penalty = MlxVisionModelHandler.get_settings(**kwargs)
            response = generate(self.model, self.processor, formatted_prompt, image, verbose=False, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, repetition_penalty=self.repetition_penalty, max_tokens=self.max_tokens)

            response = response[0].strip()

        else:
            from mlx_lm import generate
            
            # sampler, logits_processors = self.get_settings()
            response = generate(self.model, self.tokenizer, prompt=formatted_prompt, verbose=True, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens)

            response = response.strip()
            
        return response
            
    def get_settings(self):
        if self.image_input is not None:
            from mlx_vlm.sample_utils import make_sampler, make_logits_processors
        else:
            from mlx_lm.sample_utils import make_sampler, make_logits_processors

        self.sampler = make_sampler(
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        )
        self.logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
    
    def load_template(self, messages, image_input=None):
        if image_input:
            from mlx_vlm.prompt_utils import apply_chat_template
            return image_input, apply_chat_template(
                processor=self.processor,
                config=self.config,
                prompt=messages,
                num_images=1 # <-- history 자체를 전달
            )
        else:
            return None, self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
    def generate_chat_title(self, first_message: str, image_input=None)->str:
        prompt=(
            "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
            f"{first_message}\n\n"
            "Chat Title:"
        )
        logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        
        if image_input:
            from mlx_vlm import generate
            title_response=generate(self.model, self.processor, prompt=prompt, verbose=True, max_tokens=20)
        else:
            from mlx_lm import generate
            title_response=generate(self.model, self.tokenizer, prompt=prompt, verbose=True, max_tokens=20)
        
        title=title_response.strip()
        logger.info(f"생성된 채팅 제목: {title}")
        return title
    
class MlxQwen3ModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="mlx", use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        self.max_tokens=32768

        self.enable_thinking = kwargs.get("enable_thinking", False)

        self.sampler = None
        self.logits_processors = None

        self.load_model()
        
    def load_model(self):
        from mlx_lm import load
        self.model, self.tokenizer = load(self.local_model_path, adapter_path=self.local_lora_model_path)
        
    def generate_answer(self, history, **kwargs):
        from mlx_lm import generate
        text = self.load_template(history)
        self.get_settings()
        generated = generate(self.model, self.tokenizer, prompt=text, verbose=True, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens)
        
        if "</think>" in generated:
            _, response = generated.split("</think>", 1)
        else:
            response = generated  # Assign the entire generated text if no </think> tag is found
            response = response.strip()
            
        return response
    
    def get_settings(self):
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        self.sampler = make_sampler(
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        )
        self.logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
    
    def load_template(self, messages):
        return self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        
    def generate_chat_title(self, first_message: str)->str:
        from mlx_lm import generate
        prompt=(
            "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
            f"{first_message}\n\n"
            "Chat Title:"
        )
        logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        
        title_response=generate(self.model, self.tokenizer, prompt=prompt, verbose=True, max_tokens=20)
        
        title=title_response.strip()
        logger.info(f"생성된 채팅 제목: {title}")
        return title