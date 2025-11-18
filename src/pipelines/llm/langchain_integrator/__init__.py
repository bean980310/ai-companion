import os
import warnings
import platform
from pathlib import Path
from functools import partial

import torch

try:
    import mlx.nn

except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("langchain_mlx is not installed. Please install it to use MLX features.", UserWarning)
    else:
        pass

try:
    from mlx_lm.tokenizer_utils import TokenizerWrapper, SPMStreamingDetokenizer, BPEStreamingDetokenizer, NaiveStreamingDetokenizer
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("langchain_mlx is not installed. Please install it to use MLX features.", UserWarning)
    else:
        pass

from transformers import pipeline, PreTrainedModel, GenerationMixin, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers import AutoProcessor, ProcessorMixin
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
from peft import PeftModel
from llama_cpp import Llama

# import langchain.globals

from langchain_core.globals import set_llm_cache, set_verbose, set_debug
from langchain_core.caches import InMemoryCache
from langchain_community.cache import RedisCache, SQLAlchemyCache, SQLiteCache, SQLAlchemyMd5Cache, GPTCache

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser, JsonOutputParser, XMLOutputParser, PydanticOutputParser, MarkdownListOutputParser
from langchain_classic.output_parsers import RetryOutputParser, RetryWithErrorOutputParser
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnablePassthrough, RunnableWithMessageHistory, RunnableSerializable, Runnable, RunnableSequence
from langchain_community.chat_message_histories import ChatMessageHistory, SQLChatMessageHistory
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
# from langchain_classic.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PDFPlumberLoader
from langchain_community.document_loaders import CSVLoader, UnstructuredCSVLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_unstructured import UnstructuredLoader
# Back‑end specific chat/LLM wrappers
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface_hijack.chat_models import ChatHuggingFaceEnhanced
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
# The following wrappers are placeholders; implement or replace with your actual provider modules.
from langchain_perplexity import ChatPerplexity        # Perplexity AI
from langchain_xai import ChatXAI                      # xAI Grok

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from typing import Any, Generator, List, LiteralString, Literal

from .state import State

set_debug(True)
set_verbose(True)

try:
    from langchain_mlx.llms import MLXPipeline
    from langchain_mlx.chat_models import ChatMLX
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("langchain_mlx is not installed. Please install it to use MLX features.", UserWarning)
    else:
        pass

class LangchainIntegrator:
    def __init__(self, provider: str, model_name: str = None, lora_model_name: str = None, model: torch.nn.Module | mlx.nn.Module | PreTrainedModel | GenerationMixin | AutoModelForCausalLM | AutoModelForImageTextToText | AutoModel | PeftModel | Llama | Any | None = None, tokenizer: AutoTokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | PreTrainedTokenizerBase | TokenizerWrapper | type[SPMStreamingDetokenizer] | partial[SPMStreamingDetokenizer] | type[BPEStreamingDetokenizer] | type[NaiveStreamingDetokenizer] | Any | None = None, processor: AutoProcessor | ProcessorMixin | Any | None = None, **kwargs):
        """
        Parameters
        ----------
        provider : str
            One of: ``transformers`` | ``gguf`` | ``mlx`` | ``openai`` |
            ``anthropic`` | ``google_genai`` | ``perplexity`` | ``xai`` | ``openrouter`` | ``hf_endpoint`` | ``lmstudio`` | ``ollama`` .
        model_name : str
            HF repo id, local model file, or provider‑specific model id.
        lora_model_name : str
            Optional LoRA model name or path.
        model : AutoModelForCausalLM | AutoModelForImageTextToText | Qwen3ForCausalLM | Qwen3MoeForCausalLM | Llama4ForCausalLM | Llama4ForConditionalGeneration | Mistral3ForConditionalGeneration | Qwen2VLForConditionalGeneration | Qwen2_5_VLForConditionalGeneration
            Pre‑loaded model instance (if applicable).
        tokenizer : AutoTokenizer
            Pre‑loaded tokenizer instance (if applicable).
        processor : AutoProcessor
            Pre‑loaded processor instance (if applicable).
        max_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top K sampling parameter.
        top_p : float
            Top P sampling parameter.
        repetition_penalty : float
            Repetition penalty for text generation.
        api_key : str
            API key for the model provider (if applicable).
        **kwargs : Any
            Extra args forwarded to the underlying LangChain chat/LLM class.
        """

        self.provider = provider.lower()
        self.model_name = model_name
        self.model = model
        self.lora_model_name = lora_model_name
        self.max_tokens = int(kwargs.get("max_tokens", 1024))
        self.seed = int(kwargs.get("seed", 42))
        self.temperature = float(kwargs.get("temperature", 1.0))
        self.top_k = int(kwargs.get("top_k", 50))
        self.top_p = float(kwargs.get("top_p", 1.0))
        self.repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
        self.tokenizer_config = kwargs.get("tokenizer_config", {})
        self.enable_thinking = bool(kwargs.get("enable_thinking", False))

        self.chunk_size = int(kwargs.get("chunk_size", 2048))

        self.tokenizer = tokenizer
        self.processor = processor

        self.api_key = str(kwargs.get("api_key", None))

        self.n_gpu_layers = int(kwargs.get("n_gpu_layers", 1))
        self.verbose = bool(kwargs.get("verbose", True))

        # Lazily initialise attributes
        self.prompt: ChatPromptTemplate = None
        self.user_message = None
        self.chat_history: ChatMessageHistory = None
        self.chain = None

        # Build the chat/LLM instance based on provider
        self.chat: BaseChatModel | ChatHuggingFace | ChatLlamaCpp | ChatMLX | ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI | ChatPerplexity | ChatXAI = self._init_llm(provider.lower())

        self.workflow = StateGraph(state_schema=State)

        # Kick off the first generation pass
        # self.generate_answer(history)

    def _init_llm(self, provider: str) -> BaseChatModel:
        """Factory that returns a LangChain‑compatible chat/LLM object."""
        if provider == "transformers":
            # Uses HuggingFace Inference Endpoint or Hub inference API
            pipeline_kwargs={"max_new_tokens": self.max_tokens, "temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k, "repetition_penalty": self.repetition_penalty}
            pipe = pipeline(model=self.model, tokenizer=self.tokenizer, task="text-generation")
            llm = HuggingFacePipeline(pipeline=pipe, pipeline_kwargs=pipeline_kwargs, verbose=self.verbose)
            return ChatHuggingFaceEnhanced(llm=llm, verbose=self.verbose, max_tokens=self.max_tokens, model_kwargs={})
        
        elif provider == "gguf":
            # Local GGUF (llama.cpp) model
            return ChatLlamaCpp(
                model_path=self.model_name,
                lora_path=self.lora_model_name,
                n_gpu_layers=self.n_gpu_layers,
                max_tokens=self.max_tokens,
                seed=self.seed,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repeat_penalty=self.repetition_penalty,
                verbose=self.verbose,
                n_ctx=2048,  # Ensure context length is set
                n_batch=512,
            )
        elif provider == "mlx":
            # apple/mlx backend via llama.cpp; requires backend='mlx'
            pipeline_kwargs = {"max_tokens": self.max_tokens, "temp": self.temperature, "top_p": self.top_p, "top_k": self.top_k, "repetition_penalty": self.repetition_penalty}
            llm = MLXPipeline.from_model_id(model_id=self.model_name, adapter_file=self.lora_model_name, pipeline_kwargs=pipeline_kwargs, tokenizer_config=self.tokenizer_config)
            return ChatMLX(llm=llm, verbose=self.verbose)
        
        elif provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                top_logprobs=self.top_k,
                frequency_penalty=self.repetition_penalty,
                presence_penalty=self.repetition_penalty,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
            )
        elif provider == "google_genai":
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                api_key=self.api_key,
                max_output_tokens=self.max_tokens,
                verbose=self.verbose,
                model_kwargs={
                    "frequency_penalty": self.repetition_penalty,
                    "presence_penalty": self.repetition_penalty,
                }
            )
        elif provider == "perplexity":
            return ChatPerplexity(
                model=self.model_name,
                temperature=self.temperature,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                model_kwargs={
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "frequency_penalty": self.repetition_penalty,
                    "presence_penalty": self.repetition_penalty,
                },
                verbose=self.verbose,
            )
        elif provider == "xai":
            return ChatXAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_logprobs=self.top_k,
                frequency_penalty=self.repetition_penalty,
                presence_penalty=self.repetition_penalty,
                verbose=self.verbose,
            )
        elif provider == "openrouter":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                model_kwargs={
                    "top_k": self.top_k,
                    "repetition_penalty": self.repetition_penalty
                },
                verbose=self.verbose,
            )
        elif provider == "hf_endpoint":
            return HuggingFaceEndpoint(
                repo_id=self.model_name,
                temperature=self.temperature,
                huggingfacehub_api_token=self.api_key,
                provider="auto",
                max_new_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                verbose=self.verbose,
            )
        else:
            raise ValueError(f"Unsupported backend type: {provider}")

    def generate_answer(self, history):
        # chunks = []
        # response = ""
        # for chunk in self.chat.stream(history):
        #     for block in chunk.content_blocks:
        #         if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
        #             print(f"<think>{reasoning}</think>", end="", flush=True)
        #         elif block["type"] == "tool_call_chunk":
        #             tool_name = block.get("tool_name", "unknown_tool")
        #             tool_input = block.get("tool_input", "")
        #             print(f"<tool>{tool_name}({tool_input})</tool>", end="", flush=True)
        #             # chunks.append(f"<tool>{tool_name}({tool_input})</tool>")
        #             # response += f"<tool>{tool_name}({tool_input})</tool>"
        #         elif block["type"] == "text":
        #             print(block["text"], end="", flush=True)
        #             chunks.append(block["text"])
        #             response += block["text"]

        self.load_template_with_langchain(history)
        if any(n in ["transformers", "mlx"] for n in self.provider.lower()):
            if "qwen3" in self.model_name.lower() and "instruct" not in self.model_name.lower() and "thinking" not in self.model_name.lower():
                self.chat._to_chat_prompt_thinking(messages=[history, self.user_message], enable_thinking=self.enable_thinking)
            else:
                self.chat._to_chat_prompt(messages=[history, self.user_message])
        self.chain = self.prompt | self.chat | StrOutputParser()
        if not self.chat_history.messages:
            # chunks = []
            # response = ""
            # for chunk in self.chain.stream({"input": self.user_message.content}):
            #     chunks.append(chunk)
            #     print(chunk, end="", flush=True)
            # for i in range(len(chunks)):
            #     response += "".join(chunks[i])
            response = self.chain.invoke({"input": self.user_message.content})
        else:
            chain_with_history = RunnableWithMessageHistory(
                self.chain,
                lambda session_id: self.chat_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            # chunks = []
            # response = ""
            # for chunk in chain_with_history.stream({"input": self.user_message.content}, {"configurable": {"session_id": "unused"}}):
            #     chunks.append(chunk)
            #     print(chunk, end="", flush=True)
            # for i in range(len(chunks)):
            #     response += "".join(chunks[i])
            response = chain_with_history.invoke({"input": self.user_message.content}, {"configurable": {"session_id": "unused"}})

        if "</think>" in response:
            _, response = response.split("</think>", 1)
            # else:
            # response = chain_with_history.invoke({"input": self.user_message.content}, {"configurable": {"session_id": "unused"}})

        return response.strip()
    
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

    @staticmethod
    def process_doc(src: str | Path | List[str] | List[Path]):
        loader = UnstructuredLoader(src)

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200)

        chunks = []
        doc_splitted = []
        for doc in docs:
            chunk = splitter.split_documents([doc])
            chunks.extend(chunk)
            doc_splitted.append(chunks)
            chunks = []

        return doc_splitted