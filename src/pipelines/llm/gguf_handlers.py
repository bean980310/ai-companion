import llama_cpp
from llama_cpp import Llama # gguf 모델을 로드하기 위한 라이브러리
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

from langchain_community.chat_models.llamacpp import ChatLlamaCpp
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

import os

from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler

from src import logger

class GGUFCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="gguf", device='cpu', use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        self.n_gpu_layers = -1 if device != 'cpu' else 0
        self.sampler = None
        self.logits_processors = None
        
        self.load_model()
        
    def load_model(self):
        if self.use_langchain:
            self.llm = ChatLlamaCpp(
                model_path=self.local_model_path,
                lora_path=self.local_lora_model_path,
                n_gpu_layers=self.n_gpu_layers,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repeat_penalty=self.repetition_penalty,
                verbose=True
            )
            self.chat = self.llm
        else:
            self.model = Llama(
                model_path=self.local_model_path,
                lora_path=self.local_lora_model_path,
                n_gpu_layers=self.n_gpu_layers,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
                logits_all=True
            )
        
    def generate_answer(self, history, **kwargs):
        if self.use_langchain:
            self.load_template_with_langchain(history)
            self.chain = self.prompt | self.chat | StrOutputParser()
            if not self.chat_history.messages:
                response = self.chain.invoke({"input": self.user_message.content})
            else:
                chain_with_history = RunnableWithMessageHistory(
                    self.chain,
                    lambda session_id: self.chat_history,
                    input_messages_key="input",
                    history_messages_key="chat_history"
                )
                response = chain_with_history.invoke({"input": self.user_message.content}, {"configurable": {"session_id": "unused"}})

            return response
        else:
            prompt = [{"role": msg['role'], "content": msg['content']} for msg in history]
            response = self.model.create_chat_completion(
                messages=prompt,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repeat_penalty=self.repetition_penalty
            )
            return response["choices"][0]["message"]["content"]

    def get_settings(self):
        pass
    
    def load_template(self, messages):
        pass

    def load_template_with_langchain(self, messages):
        self.chat_history = ChatMessageHistory()
        for msg in messages[:-1]:
            if msg["role"] == "system":
                system_message = SystemMessage(content=msg["content"])
            if msg["role"] == "user":
                self.chat_history.add_user_message(msg["content"])
            if msg["role"] == "assistant":
                self.chat_history.add_ai_message(msg["content"])
        self.user_message = HumanMessage(content=messages[-1]["content"])
        # logger.info(len(self.chat_history.messages))
        if not self.chat_history.messages:
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_message.content),
                    ("user", "{input}")
                ]
            )
        else:
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_message.content),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "{input}")
                ]
            )
        
    def generate_chat_title(self, first_message: str)->str:
        prompt=(
            "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
            f"{first_message}\n\n"
            "Chat Title:"
        )
        
        logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        title_response=self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20
        )
        
        title = title_response["choices"][0]["message"]["content"]
        logger.info(f"생성된 채팅 제목: {title}")
        return title