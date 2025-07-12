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