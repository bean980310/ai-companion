from .... import logger
import traceback
from ..base_handlers import BaseAPIClientWrapper

import requests

from langchain_perplexity import ChatPerplexity
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

class PerplexityClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model, api_key="None", use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        if self.use_langchain:
            self.load_model()

    def load_model(self):
        self.llm = ChatPerplexity(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            model_kwargs={
                "top_p": self.top_p,
                "top_k": self.top_k,
                "frequency_penalty": self.repetition_penalty,
                "presence_penalty": self.repetition_penalty,
            },
            verbose=True,
        )
        self.chat = self.llm

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
            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] Perplexity API 요청: {messages}")
                
            url = "https://api.perplexity.ai/chat/completions"
            payload = { 
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "frequency_penalty": self.repetition_penalty,
                "presence_penalty": self.repetition_penalty,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.request("POST", url, json=payload, headers=headers)
            answer = response.text
            return answer
    
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