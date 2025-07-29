from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForCausalLM, GenerationConfig, Llama4ForConditionalGeneration, TextStreamer, TextIteratorStreamer, Qwen3ForCausalLM, Qwen3MoeForCausalLM, pipeline

from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.chat_models import ChatHuggingFace
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

from peft import PeftModel
import os
import traceback
import threading

from src import logger

from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler, BaseModelHandler

class TransformersCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        self.max_new_tokens = self.max_tokens

        self.kwargs = self.get_settings_with_langchain()
        self.device = device

        self.pipe = None
        self.load_model()
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
        
        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)

        if self.use_langchain:
            self.pipe = pipeline(model=self.model, tokenizer=self.tokenizer, task="text-generation", temperature=self.temperature, max_new_tokens=self.max_new_tokens, repetition_penalty=self.repetition_penalty, top_k=self.top_k, top_p=self.top_p)
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            self.chat = ChatHuggingFace(llm=self.llm, verbose=True)
        
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
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            # If kwargs are provided, update the settings
            self.config = self.get_settings()
            self.config = self.get_settings()

            input_ids = self.load_template(prompt_messages)
            
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            # outputs = self.model.generate(
            #     input_ids,
            #     generation_config=self.config,
            #     max_new_tokens=1024,
            #     do_sample=True,
            # )
            
            _ = self.model.generate(
                input_ids,
                generation_config=self.config,
                streamer=streamer,
            )
            
            # generated_text = self.tokenizer.decode(
            #     outputs[0][input_ids.shape[-1]:],
            #     skip_special_tokens=True
            # )
            
            generated_text = ""
            
            for text in streamer:
                generated_text += text
                
            return generated_text.strip()
        
    def get_settings(self):
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )

    def get_settings_with_langchain(self):
        return {"max_new_tokens": self.max_new_tokens, "temp": self.temperature, "top_p": self.top_p, "top_k": self.top_k, "repetition_penalty": self.repetition_penalty}

    def load_template(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False
        )
    
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
        
class TransformersVisionModelHandler(BaseVisionModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        self.max_new_tokens = self.max_tokens

        self.device = device
        self.load_model()

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.local_model_path)
        self.model = AutoModel.from_pretrained(self.local_model_path)

        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)

    def generate_answer(self, history, image_input=None, **kwargs):
        try:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]

            self.config = self.get_settings()

            inputs = self.load_template(prompt_messages, image_input)
            
            streamer = TextStreamer(self.processor, skip_prompt=True)

            # outputs = self.model.generate(
            #     **inputs,
            #     max_new_tokens=1024,
            #     do_sample=True,
            # )
            
            _ = self.model.generate(
                **inputs,
                generation_config=self.config,
                streamer=streamer
            )

            # generated_text = self.processor.decode(
            #     outputs[0],
            #     skip_special_tokens=True
            # )
            
            generated_text = ""
            
            for text in streamer:
                generated_text += text

            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}")
            return f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}"

    def get_settings(self):
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )

    def load_template(self, messages, image_input):
        if image_input:
            return self.processor(
                image_input,
                messages,
                add_special_tokens=False,
                return_tensors="pt"
            )
        else:
            return self.processor(
                messages,
                add_special_tokens=False,
                return_tensors="pt"
            )
            
class TransformersLlama4ModelHandler(BaseModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        self.max_new_tokens = self.max_tokens

        self.tokenizer = None
        self.processor = None
        self.model = None

        self.load_model()
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.local_model_path)
        self.model = Llama4ForConditionalGeneration.from_pretrained(self.local_model_path)

        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)
            
    def generate_answer(self, history, image_input=None, **kwargs):
        try:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]

            self.config = self.get_settings()

            inputs = self.load_template(prompt_messages, image_input)
            streamer = TextStreamer(self.processor, skip_prompt=True)

            # outputs = self.model.generate(
            #     **inputs,
            #     max_new_tokens=1024,
            #     do_sample=True,
            # )
            
            _ = self.model.generate(
                **inputs,
                generation_config=self.config,
                streamer=streamer
            )
            
            # input_ids = inputs['input_ids']
            
            # if image_input:
            #     generated_text = self.processor.decode(
            #     outputs[0][input_ids.shape[-1]:],
            #     skip_special_tokens=True
            # )
            # else:
            #     generated_text = self.tokenizer.decode(
            #         outputs[0][input_ids.shape[-1]:],
            #         skip_special_tokens=True
            #     )
                
            generated_text = ""
            
            for text in streamer:
                generated_text += text
                
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}")
            return f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}"

    def get_settings(self):
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )

    def load_template(self, messages, image_input):
        if image_input:
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                return_tensors="pt",
                return_dict=True
            )
        else:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
                return_dict=True
            )
            
class TransformersQwen3ModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        self.max_new_tokens = kwargs.get("max_new_tokens", 32768)

        self.device = device
        self.load_model()
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = Qwen3ForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
        
        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)
        
    def generate_answer(self, history, **kwargs):
        try:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            
            self.config = self.get_settings(**kwargs)

            input_ids = self.load_template(prompt_messages)
            
            model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.model.device)
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            outputs = self.model.generate(
                **model_inputs,
                generation_config=self.config,
            )
            
            generated_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()
            
            # _ = self.model.generate(
            #     input_ids,
            #     max_new_tokens=32768,
            #     do_sample=True,
            #     streamer=streamer,
            # )
            
            # generated_text = self.tokenizer.decode(
            #     outputs[0][input_ids.shape[-1]:],
            #     skip_special_tokens=True
            # )
            
            generated_stream = ""
            
            for ids in streamer:
                generated_ids += ids
                
            try:
                index=len(generated_ids)-generated_ids[::-1].index(151668)
            except:
                index=0
                
            generated_thinking = self.tokenizer.decode(generated_ids[:index], skip_special_tokens=True)
            generated_text = self.tokenizer.decode(generated_ids[index:], skip_special_tokens=True)
                
            return generated_text.strip()
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}")
            return f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}"
        
    def get_settings(self):
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )
        
    def load_template(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False,
            enable_thinking=True
        )
    
class TransformersQwen3MoeModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        self.max_new_tokens = kwargs.get("max_new_tokens", 32768)
        
        self.device = device
        self.load_model()
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = Qwen3MoeForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
        
        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)
        
    def generate_answer(self, history, **kwargs):
        try:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            
            self.config = self.get_settings(**kwargs)
            self.config = self.get_settings(**kwargs)

            input_ids = self.load_template(prompt_messages)
            
            model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.model.device)
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            outputs = self.model.generate(
                **model_inputs,
                generation_config=self.config
            )
            
            generated_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()
            
            # _ = self.model.generate(
            #     input_ids,
            #     max_new_tokens=32768,
            #     do_sample=True,
            #     streamer=streamer,
            # )
            
            # generated_text = self.tokenizer.decode(
            #     outputs[0][input_ids.shape[-1]:],
            #     skip_special_tokens=True
            # )
            
            generated_stream = ""
            
            for ids in streamer:
                generated_ids += ids
                
            try:
                index=len(generated_ids)-generated_ids[::-1].index(151668)
            except:
                index=0
                
            generated_thinking = self.tokenizer.decode(generated_ids[:index], skip_special_tokens=True)
            generated_text = self.tokenizer.decode(generated_ids[index:], skip_special_tokens=True)
                
            return generated_text.strip()
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}")
            return f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}"
        
    def get_settings(self):
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )
        
    def load_template(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False,
            enable_thinking=True
        )