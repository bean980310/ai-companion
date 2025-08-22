from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForImageTextToText, AutoModelForCausalLM, GenerationConfig, Llama4ForConditionalGeneration, TextStreamer, TextIteratorStreamer, Qwen3ForCausalLM, Qwen3MoeForCausalLM, Mistral3ForConditionalGeneration, MistralForCausalLM, Llama4Processor, LlamaTokenizer

from transformers.tokenization_mistral_common import MistralCommonTokenizer

from .langchain_integrator import LangchainIntegrator

from peft import PeftModel
import os
import traceback
import threading

from src import logger

from typing import Any, List

from PIL.Image import Image

from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler, BaseModelHandler

class TransformersCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', use_langchain: bool = True, **kwargs):
        super().__init__(self, model_id, lora_model_id, use_langchain, **kwargs)

        if "qwen3" in self.model_id.lower():
            if "instruct" in self.model_id.lower():
                self.max_tokens = 16384
            else:
                self.max_tokens = 32768

        self.max_new_tokens = self.max_tokens
        self.enable_thinking = kwargs.get("enable_thinking", True)

        self.device = device

        self.load_model()
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
        
        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)

        if self.use_langchain:
            self.langchain_integrator = LangchainIntegrator(
                backend_type="transformers",
                model=self.model,
                tokenizer=self.tokenizer,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                verbose=True
            )
        
    def generate_answer(self, history, **kwargs):
        if self.use_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            # If kwargs are provided, update the settings
            self.config = self.get_settings()

            input_ids = self.load_template(prompt_messages)
            
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            # outputs = self.model.generate(
            #     input_ids,
            #     generation_config=self.config,
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

            try:
                index=len(generated_text)-generated_text[::-1].index(151668)
            except:
                index=0
                
            generated_thinking = generated_text[:index]
            generated_text = generated_text[index:]
                
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



    def load_template(self, messages):
        if self.enable_thinking:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
                enable_thinking=True
            )
        else:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False
            )
        
class TransformersVisionModelHandler(BaseVisionModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', use_langchain: bool = True, image_input: str | Image | Any | None = None, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, image_input, **kwargs)

        self.image_input = image_input

        self.max_new_tokens = self.max_tokens

        self.device = device
        self.load_model()

    def load_model(self):
        if self.image_input:
            self.processor = AutoProcessor.from_pretrained(self.local_model_path, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(self.local_model_path, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True)

        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)

    def generate_answer(self, history, **kwargs):
        try:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]

            self.config = self.get_settings()

            inputs = self.load_template(prompt_messages)
            
            streamer = TextStreamer(self.processor, skip_prompt=True)

            # outputs = self.model.generate(
            #     **inputs,
            #     generation_config=self.config,
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

    def load_template(self, messages):
        if self.image_input:
            return self.processor(
                images=[self.image_input[i] for i in range(len(self.image_input))] if isinstance(self.image_input, list) else self.image_input,
                text=messages,
                add_special_tokens=False,
                return_tensors="pt"
            )
        else:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False
            )
            
# class TransformersLlama4ModelHandler(BaseModelHandler):
#     def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', use_langchain: bool = True, **kwargs):
#         super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

#         self.max_new_tokens = self.max_tokens

#         self.tokenizer = None
#         self.processor = None
#         self.model = None

#         self.load_model()
        
#     def load_model(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
#         self.processor = AutoProcessor.from_pretrained(self.local_model_path)
#         self.model = Llama4ForConditionalGeneration.from_pretrained(self.local_model_path)

#         if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
#             self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)
            
#     def generate_answer(self, history, image_input=None, **kwargs):
#         try:
#             prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]

#             self.config = self.get_settings()

#             inputs = self.load_template(prompt_messages, image_input)
#             streamer = TextStreamer(self.processor, skip_prompt=True)

#             # outputs = self.model.generate(
#             #     **inputs,
#             #     max_new_tokens=1024,
#             #     do_sample=True,
#             # )
            
#             _ = self.model.generate(
#                 **inputs,
#                 generation_config=self.config,
#                 streamer=streamer
#             )
            
#             # input_ids = inputs['input_ids']
            
#             # if image_input:
#             #     generated_text = self.processor.decode(
#             #     outputs[0][input_ids.shape[-1]:],
#             #     skip_special_tokens=True
#             # )
#             # else:
#             #     generated_text = self.tokenizer.decode(
#             #         outputs[0][input_ids.shape[-1]:],
#             #         skip_special_tokens=True
#             #     )
                
#             generated_text = ""
            
#             for text in streamer:
#                 generated_text += text
                
#             return generated_text.strip()
#         except Exception as e:
#             logger.error(f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}")
#             return f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}"

#     def get_settings(self):
#         return GenerationConfig(
#             max_new_tokens=self.max_new_tokens,
#             do_sample=True,
#             temperature=self.temperature,
#             top_k=self.top_k,
#             top_p=self.top_p,
#             repetition_penalty=self.repetition_penalty
#         )

#     def load_template(self, messages, image_input):
#         if image_input:
#             return self.processor.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,
#                 tokenize=False,
#                 return_tensors="pt",
#                 return_dict=True
#             )
#         else:
#             return self.tokenizer.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,
#                 return_tensors="pt",
#                 tokenize=False,
#                 return_dict=True
#             )
        
            
# class TransformersQwen3ModelHandler(BaseCausalModelHandler):
#     def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', use_langchain: bool = True, **kwargs):
#         super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

#         self.max_new_tokens = kwargs.get("max_new_tokens", 32768)

#         self.device = device
#         self.load_model()
        
#     def load_model(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
#         self.model = Qwen3ForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
        
#         if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
#             self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)
        
#     def generate_answer(self, history, **kwargs):
#         try:
#             prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            
#             self.config = self.get_settings(**kwargs)

#             input_ids = self.load_template(prompt_messages)
            
#             model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.model.device)
#             streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
#             outputs = self.model.generate(
#                 **model_inputs,
#                 generation_config=self.config,
#             )
            
#             generated_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()
            
#             # _ = self.model.generate(
#             #     input_ids,
#             #     max_new_tokens=32768,
#             #     do_sample=True,
#             #     streamer=streamer,
#             # )
            
#             # generated_text = self.tokenizer.decode(
#             #     outputs[0][input_ids.shape[-1]:],
#             #     skip_special_tokens=True
#             # )
            
#             generated_stream = ""
            
#             for ids in streamer:
#                 generated_ids += ids
                
#             try:
#                 index=len(generated_ids)-generated_ids[::-1].index(151668)
#             except:
#                 index=0
                
#             generated_thinking = self.tokenizer.decode(generated_ids[:index], skip_special_tokens=True)
#             generated_text = self.tokenizer.decode(generated_ids[index:], skip_special_tokens=True)
                
#             return generated_text.strip()
        
#         except Exception as e:
#             logger.error(f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}")
#             return f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}"
        
#     def get_settings(self):
#         return GenerationConfig(
#             max_new_tokens=self.max_new_tokens,
#             do_sample=True,
#             temperature=self.temperature,
#             top_k=self.top_k,
#             top_p=self.top_p,
#             repetition_penalty=self.repetition_penalty
#         )
        
#     def load_template(self, messages):
#         return self.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt",
#             tokenize=False,
#             enable_thinking=True
#         )
 
    
# class TransformersQwen3MoeModelHandler(BaseCausalModelHandler):
#     def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', use_langchain: bool = True, **kwargs):
#         super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

#         self.max_new_tokens = kwargs.get("max_new_tokens", 32768)
        
#         self.device = device
#         self.load_model()
        
#     def load_model(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
#         self.model = Qwen3MoeForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
        
#         if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
#             self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)
        
#     def generate_answer(self, history, **kwargs):
#         try:
#             prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            
#             self.config = self.get_settings(**kwargs)

#             input_ids = self.load_template(prompt_messages)
            
#             model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.model.device)
#             streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
#             outputs = self.model.generate(
#                 **model_inputs,
#                 generation_config=self.config
#             )
            
#             generated_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()
            
#             # _ = self.model.generate(
#             #     input_ids,
#             #     max_new_tokens=32768,
#             #     do_sample=True,
#             #     streamer=streamer,
#             # )
            
#             # generated_text = self.tokenizer.decode(
#             #     outputs[0][input_ids.shape[-1]:],
#             #     skip_special_tokens=True
#             # )
            
#             # generated_stream = ""
            
#             # for ids in streamer:
#             #     generated_ids += ids
                
#             try:
#                 index=len(generated_ids)-generated_ids[::-1].index(151668)
#             except:
#                 index=0
                
#             generated_thinking = self.tokenizer.decode(generated_ids[:index], skip_special_tokens=True)
#             generated_text = self.tokenizer.decode(generated_ids[index:], skip_special_tokens=True)
                
#             return generated_text.strip()
        
#         except Exception as e:
#             logger.error(f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}")
#             return f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}"
        
#     def get_settings(self):
#         return GenerationConfig(
#             max_new_tokens=self.max_new_tokens,
#             do_sample=True,
#             temperature=self.temperature,
#             top_k=self.top_k,
#             top_p=self.top_p,
#             repetition_penalty=self.repetition_penalty
#         )
        
#     def load_template(self, messages):
#         return self.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt",
#             tokenize=False,
#             enable_thinking=True
#         )
