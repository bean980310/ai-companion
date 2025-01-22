import os
import torch
import logging
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig

from optimum.quanto import QuantizedModelForCausalLM
from src.common.utils import make_local_dir_name

logger = logging.getLogger(__name__)

class GLM4HfHandler:
    def __init__(self, model_id, local_model_path=None, model_type="transformers", device='cpu'):
        self.model_dir = local_model_path or os.path.join("./models", model_type, make_local_dir_name(model_id))
        self.tokenizer = None
        self.model = None
        self.device = device
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"[*] Loading tokenizer from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir
            )
            logger.info(f"[*] Loading model from {self.model_dir}")
            if "float8" in self.model_dir or "int8" in self.model_dir or "int4" in self.model_dir:
                self.model=QuantizedModelForCausalLM.from_pretrained(
                    self.model_dir
                ).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    torch_dtype=torch.bfloat16,
                ).to(self.device)
            logger.info(f"[*] Model loaded successfully: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load GLM4 Model: {str(e)}\n\n{traceback.format_exc()}")
            raise
    def generate_answer(self, history):
        try:
            # 메시지 처리
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] Prompt messages for GLM: {prompt_messages}")
            
            inputs = self.tokenizer.apply_chat_template(
                conversation=prompt_messages,
                add_generation_prompt=True, 
                tokenize=True, 
                return_tensors="pt",
                return_dict=True
            ).to(self.model.device)
            logger.info("[*] GLM input template applied successfully")
                
            input_len = inputs['input_ids'].shape[1]
                
            # 생성 설정
            generation_config = {
                "input_ids": inputs['input_ids'],
                "attention_mask": inputs['attention_mask'],
                "max_new_tokens": 128,
                "do_sample": False,
            }
                
            # 텍스트 생성
            outputs = self.model.generate(**generation_config)
            logger.info("[*] GLM model generated the response")
                
            # 결과 처리
            generated_text = self.tokenizer.decode(
                outputs[0][input_len:],
                skip_special_tokens=True
            )
            logger.info(f"[*] Generated text: {generated_text}")
                
            return generated_text.strip()
            
        except Exception as e:
            error_msg = f"Error during GLM answer generation: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg