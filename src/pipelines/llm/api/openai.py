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
from langchain.chains.llm import LLMChain

class OpenAIClientWrapper:
    def __init__(self, selected_model, api_key="None", **kwargs):
        self.model = selected_model
        self.api_key = api_key

        self.max_tokens=2048
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 1.0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

    def generate_answer(self, history, **kwargs):
        openai.api_key = self.api_key

        messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
        logger.info(f"[*] OpenAI API 요청: {messages}")

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_logprobs=self.top_k,
                frequency_penalty=self.repetition_penalty,
                top_p=self.top_p,
            )
            answer = response.choices[0].message["content"]
            logger.info(f"[*] OpenAI 응답: {answer}")
            return answer
        except Exception as e:
            logger.error(f"OpenAI API 오류: {str(e)}\n\n{traceback.format_exc()}")
            return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
from src import logger

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