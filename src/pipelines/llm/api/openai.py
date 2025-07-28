from .... import logger
import traceback

import openai

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.chains.llm import LLMChain

from langchain_community.chat_message_histories import ChatMessageHistory
class OpenAIClientWrapper:
    def __init__(self, selected_model, api_key="None", use_langchain: bool = True, **kwargs):
        self.model = selected_model
        self.api_key = api_key

        self.use_langchain = use_langchain

        self.max_tokens=2048
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 1.0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        self.llm = None
        self.chat = None
        self.memory = None
        self.prompt = None
        self.user_message = None
        self.chat_history = None
        self.chain = None

        if self.use_langchain:
            self.load_model()

    def load_model(self):
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            top_logprobs=self.top_k,
            frequency_penalty=self.repetition_penalty,
            presence_penalty=self.repetition_penalty,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
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
            openai.api_key = self.api_key

            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] OpenAI API 요청: {messages}")

            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_logprobs=self.top_k,
                frequency_penalty=self.repetition_penalty,
                presence_penalty=self.repetition_penalty,
                top_p=self.top_p,
            )
            answer = response.choices[0].message["content"]
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

class OpenAILangChainIntegration:
    def __init__(self, api_key=None, model="gpt-4o-mini", temperature=0.6, top_p=0.9, top_k=40, repetition_penalty=1.0):
        self.api_key = api_key
        if not api_key:
            logger.error("OpenAI API Key가 missing.")
            raise "OpenAI API Key가 필요합니다."
        
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_logprobs=top_k,
            frequency_penalty=repetition_penalty,
            api_key=api_key,
            max_tokens=2048
        )