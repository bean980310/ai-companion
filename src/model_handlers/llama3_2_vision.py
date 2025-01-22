# model_handlers/vision_model_handler.py
import os
import torch
import logging
import traceback
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from src.common.utils import make_local_dir_name

logger = logging.getLogger(__name__)

class VisionModelHandler:
    def __init__(self, model_id, local_model_path=None, model_type="transformers", device='cpu'):
        self.model_dir = local_model_path or os.path.join("./models", model_type, make_local_dir_name(model_id))
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
            if 'fp8' in self.model_dir:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_8bit=True,)
                self.model = AutoModel.from_pretrained(
                    self.model_dir,
                    quantization_config=bnb_config,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).to(self.device)
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_dir,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).to(self.device)
            logger.info(f"[*] Model loaded successfully: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load Vision Model: {str(e)}\n\n{traceback.format_exc()}")
            raise

    def generate_answer(self, history, image_input=None):
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
                temperature=0.6,
                top_p=0.9
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