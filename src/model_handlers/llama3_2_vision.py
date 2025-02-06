# model_handlers/vision_model_handler.py
import os
import torch
import logging
import traceback
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from src.common.utils import make_local_dir_name

from peft import PeftModel

logger = logging.getLogger(__name__)

class VisionModelHandler:
    def __init__(self, model_id, lora_model_id=None, local_model_path=None, lora_path=None, model_type="transformers", device='cpu'):
        self.model_dir = local_model_path or os.path.join("./models/llm", model_type, make_local_dir_name(model_id))
        self.lora_model_dir = lora_path or os.path.join("./model/llm/loras", make_local_dir_name(lora_model_id))
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.device = device
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"[*] Loading tokenizer from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            
            logger.info(f"[*] Loading processor from {self.model_dir}")
            self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
            
            logger.info(f"[*] Loading model from {self.model_dir}")
            self.model = AutoModel.from_pretrained(
                self.model_dir,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(self.device)
            if self.lora_model_dir and os.path.exists(self.lora_model_dir):
                logger.info(f"[*] Loading LoRA from {self.lora_model_dir}")
                self.model=PeftModel.from_pretrained(
                    self.model_dir,
                    self.lora_model_dir
                ).to(self.device)
            logger.info(f"[*] Model loaded successfully: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load Vision Model: {str(e)}\n\n{traceback.format_exc()}")
            raise

    def generate_answer(self, history, image_input=None, temperature=0.6, top_k=50, top_p=0.9, repetition_penalty=0.8):
        try:
            prompt_messages = []
            for msg in history:
                if msg['role'] == 'user':
                    if image_input:
                        prompt_messages.append({
                            "role": "user", 
                            "content": "Please see the attached image."
                        })
                        prompt_messages.append({
                            "role": "user",
                            "content": msg['content']
                        })
                    else:
                        prompt_messages.append({"role": "user", "content": msg['content']})
                elif msg['role'] == 'assistant':
                    prompt_messages.append({"role": "assistant", "content": msg['content']})
            
            logger.info(f"[*] Prompt messages: {prompt_messages}")
            
            if image_input:
                inputs = self.processor(
                    image_input,
                    prompt_messages,
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                logger.info("[*] Image input processed successfully")
            else:
                inputs = self.tokenizer(
                    [msg['content'] for msg in prompt_messages if msg['role'] in ['user', 'assistant']],
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                logger.info("[*] Text input processed successfully")
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            terminators = self.get_terminators()
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            logger.info("[*] Model generated the response")
            
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            logger.info(f"[*] Generated text: {generated_text}")
            
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error during answer generation: {str(e)}\n\n{traceback.format_exc()}")
            return f"Error during answer generation: {str(e)}\n\n{traceback.format_exc()}"

    def get_terminators(self):
        return [
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]