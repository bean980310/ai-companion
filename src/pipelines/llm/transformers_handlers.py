from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoConfig, AutoModelForCausalLM, GenerationConfig, Llama4ForConditionalGeneration, TextStreamer, TextIteratorStreamer
from peft import PeftModel
import os
import traceback
import threading

from src import logger

from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler, BaseModelHandler

class TransformersCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', **kwargs):
        super().__init__(model_id, lora_model_id)

        self.max_new_tokens = kwargs.get("max_new_tokens", 1024)
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 1.0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        self.device = device
        self.load_model()
        
    def load_model(self):
        self.config = AutoConfig.from_pretrained(self.local_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, config=self.config, trust_remote_code=True, device_map='auto')
        
        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)
        
    def generate_answer(self, history, **kwargs):
        try:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            # If kwargs are provided, update the settings
            self.get_settings()

            input_ids = self.load_template(prompt_messages)
            
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            # outputs = self.model.generate(
            #     input_ids,
            #     max_new_tokens=1024,
            #     do_sample=True,
            # )
            
            _ = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
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
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}")
            return f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}"
        
    def get_settings(self):
        self.model.config.temperature = self.temperature
        self.model.config.top_k = self.top_k
        self.model.config.top_p = self.top_p
        self.model.config.repetition_penalty = self.repetition_penalty

    def load_template(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False
        )
        
class TransformersVisionModelHandler(BaseVisionModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', **kwargs):
        super().__init__(model_id, lora_model_id)

        self.max_new_tokens = kwargs.get("max_new_tokens", 1024)
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 1.0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        self.device = device
        self.load_model()

    def load_model(self):
        self.config = AutoConfig.from_pretrained(self.local_model_path)
        self.processor = AutoProcessor.from_pretrained(self.local_model_path)
        self.model = AutoModel.from_pretrained(self.local_model_path, config=self.config)

        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)

    def generate_answer(self, history, image_input=None, **kwargs):
        try:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]

            self.get_settings()

            inputs = self.load_template(prompt_messages, image_input)
            
            streamer = TextStreamer(self.processor, skip_prompt=True)

            # outputs = self.model.generate(
            #     **inputs,
            #     max_new_tokens=1024,
            #     do_sample=True,
            # )
            
            _ = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
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
        self.model.config.temperature = self.temperature
        self.model.config.top_k = self.top_k
        self.model.config.top_p = self.top_p
        self.model.config.repetition_penalty = self.repetition_penalty

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
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', **kwargs):
        super().__init__(model_id, lora_model_id)

        self.max_new_tokens = kwargs.get("max_new_tokens", 1024)
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 1.0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        self.tokenizer = None
        self.processor = None
        self.model = None

        self.load_model()
        
    def load_model(self):
        self.config = AutoConfig.from_pretrained(self.local_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.local_model_path)
        self.model = Llama4ForConditionalGeneration.from_pretrained(self.local_model_path, config=self.config)

        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)
            
    def generate_answer(self, history, image_input=None, **kwargs):
        try:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]

            self.get_settings()

            inputs = self.load_template(prompt_messages, image_input)
            streamer = TextStreamer(self.processor, skip_prompt=True)

            # outputs = self.model.generate(
            #     **inputs,
            #     max_new_tokens=1024,
            #     do_sample=True,
            # )
            
            _ = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
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
        self.model.config.temperature = self.temperature
        self.model.config.top_k = self.top_k
        self.model.config.top_p = self.top_p
        self.model.config.repetition_penalty = self.repetition_penalty

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
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', **kwargs):
        super().__init__(model_id, lora_model_id)

        self.max_new_tokens = kwargs.get("max_new_tokens", 32768)
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 1.0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        self.device = device
        self.load_model()
        
    def load_model(self):
        self.config = AutoConfig.from_pretrained(self.local_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, config=self.config, trust_remote_code=True, device_map='auto')
        
        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)
        
    def generate_answer(self, history, **kwargs):
        try:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            
            self.get_settings(**kwargs)

            input_ids = self.load_template(prompt_messages)
            
            model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.model.device)
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens
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
        self.model.config.temperature = self.temperature
        self.model.config.top_k = self.top_k
        self.model.config.top_p = self.top_p
        self.model.config.repetition_penalty = self.repetition_penalty
        
    def load_template(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False,
            enable_thinking=True
        )