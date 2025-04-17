from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoConfig, AutoModelForCausalLM, GenerationConfig, Llama4ForConditionalGeneration
from peft import PeftModel
import os
import traceback

from src import logger

from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler, BaseModelHandler

class TransformersCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu'):
        super().__init__(model_id, lora_model_id)
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
            
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                do_sample=True,
            )
            
            generated_text = self.tokenizer.decode(
                outputs[0][input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}")
            return f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}"
        
    def get_settings(self, *, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0):
        self.model.config.temperature = temperature
        self.model.config.top_k = top_k
        self.model.config.top_p = top_p
        self.model.config.repetition_penalty = repetition_penalty
        
    def load_template(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False
        )
        
class TransformersVisionModelHandler(BaseVisionModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu'):
        super().__init__(model_id, lora_model_id)
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

            self.get_settings(**kwargs)

            inputs = self.load_template(prompt_messages, image_input)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
            )

            generated_text = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}")
            return f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}"

    def get_settings(self, *, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0):
        self.model.config.temperature = temperature
        self.model.config.top_k = top_k
        self.model.config.top_p = top_p
        self.model.config.repetition_penalty = repetition_penalty

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
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu'):
        super().__init__(model_id, lora_model_id)
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

            self.get_settings(**kwargs)

            inputs = self.load_template(prompt_messages, image_input)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
            )
            
            input_ids = inputs['input_ids']
            
            if image_input:
                generated_text = self.processor.decode(
                outputs[0][input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            else:
                generated_text = self.tokenizer.decode(
                    outputs[0][input_ids.shape[-1]:],
                    skip_special_tokens=True
                )
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}")
            return f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}"
            
    def get_settings(self, *, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0):
        self.model.config.temperature = temperature
        self.model.config.top_k = top_k
        self.model.config.top_p = top_p
        self.model.config.repetition_penalty = repetition_penalty
            
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