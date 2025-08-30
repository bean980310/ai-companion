from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLLM, BaseChatModel
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
# Back‑end specific chat/LLM wrappers
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
# The following wrappers are placeholders; implement or replace with your actual provider modules.
from langchain_perplexity import ChatPerplexity        # Perplexity AI
from langchain_xai import ChatXAI                      # xAI Grok

import warnings
import platform

from transformers import pipeline, AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText, Qwen3ForCausalLM, Qwen3MoeForCausalLM, Llama4ForCausalLM, Llama4ForConditionalGeneration, Mistral3ForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

from typing import Any

try:
    from langchain_mlx.llms import MLXPipeline
    from langchain_mlx.chat_models import ChatMLX
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("langchain_mlx is not installed. Please install it to use MLX features.", UserWarning)
    else:
        pass

class LangchainIntegrator:
    def __init__(self, backend_type: str, model_name: str = None, lora_model_name: str = None, model: AutoModelForCausalLM | AutoModelForImageTextToText | Qwen3ForCausalLM | Qwen3MoeForCausalLM | Llama4ForCausalLM | Llama4ForConditionalGeneration | Mistral3ForConditionalGeneration | Qwen2VLForConditionalGeneration | Qwen2_5_VLForConditionalGeneration = None, tokenizer: AutoTokenizer = None, processor: AutoProcessor = None, max_tokens: int = 512, temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0, repetition_penalty: float = 1.0, api_key: str = None, **kwargs):
        """
        Parameters
        ----------
        backend_type : str
            One of: ``transformers`` | ``gguf`` | ``mlx`` | ``openai`` |
            ``anthropic`` | ``google_genai`` | ``perplexity`` | ``xai`` | ``openrouter`` | ``hf_endpoint``.
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

        self.model_name = model_name
        self.model = model
        self.lora_model_name = lora_model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self.tokenizer = tokenizer
        self.processor = processor

        self.api_key = api_key

        self.n_gpu_layers = kwargs.get("n_gpu_layers", 1)
        self.verbose: bool = kwargs.get("verbose", True)

        # Lazily initialise attributes
        self.prompt = None
        self.user_message = None
        self.chat_history = None
        self.chain = None

        # Build the chat/LLM instance based on backend_type
        self.chat: BaseChatModel = self._init_llm(backend_type.lower())

        # Kick off the first generation pass
        # self.generate_answer(history)

    def _init_llm(self, backend_type: str) -> BaseChatModel:
        """Factory that returns a LangChain‑compatible chat/LLM object."""
        if backend_type == "transformers":
            # Uses HuggingFace Inference Endpoint or Hub inference API
            pipeline_kwargs={"max_new_tokens": self.max_tokens, "temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k, "repetition_penalty": self.repetition_penalty}
            pipe = pipeline(model=self.model, tokenizer=self.tokenizer, task="text-generation")
            llm = HuggingFacePipeline(pipeline=pipe, pipeline_kwargs=pipeline_kwargs, verbose=self.verbose)
            return ChatHuggingFace(llm=llm, verbose=self.verbose, max_tokens=self.max_tokens, model_kwargs={})
        
        elif backend_type == "gguf":
            # Local GGUF (llama.cpp) model
            return ChatLlamaCpp(
                model_path=self.model_name,
                lora_path=self.lora_model_name,
                n_gpu_layers=self.n_gpu_layers,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repeat_penalty=self.repetition_penalty,
                verbose=self.verbose,
                n_ctx=2048,  # Ensure context length is set
            )
        elif backend_type == "mlx":
            # apple/mlx backend via llama.cpp; requires backend='mlx'
            pipeline_kwargs = {"max_tokens": self.max_tokens, "temp": self.temperature, "top_p": self.top_p, "top_k": self.top_k, "repetition_penalty": self.repetition_penalty}
            llm = MLXPipeline.from_model_id(model_id=self.model_name, adapter_file=self.lora_model_name, pipeline_kwargs=pipeline_kwargs)
            return ChatMLX(llm=llm, verbose=self.verbose)
        
        elif backend_type == "openai":
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
        elif backend_type == "anthropic":
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
            )
        elif backend_type == "google_genai":
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
        elif backend_type == "perplexity":
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
        elif backend_type == "xai":
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
        elif backend_type == "openrouter":
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
        elif backend_type == "hf_endpoint":
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
            raise ValueError(f"Unsupported backend type: {backend_type}")

    def generate_answer(self, history):
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