# model_handlers/glm4v_handler.py

import os
import torch
import logging
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from PIL import Image
from src.common.utils import make_local_dir_name

logger = logging.getLogger(__name__)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if input_ids[0][-1] in stop_ids:
                return True
        return False

class GLM4VHandler:
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
                self.model_dir, 
                trust_remote_code=True, 
                encode_special_tokens=True
            )
            
            logger.info(f"[*] Loading model from {self.model_dir}")
            if 'fp8' in self.model_dir:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).to(self.device).eval()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).to(self.device).eval()
            logger.info(f"[*] Model loaded successfully: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load GLM4V Model: {str(e)}\n\n{traceback.format_exc()}")
            raise

    def get_stopping_criteria(self):
        """GLM 모델의 실제 stopping 토큰 ID들을 사용"""
        stop_token_ids = [
            [self.tokenizer.eos_token_id],  # EOS 토큰
            [2],  # ChatGLM의 일반적인 종료 토큰
            [self.tokenizer.pad_token_id] if self.tokenizer.pad_token_id is not None else [],  # PAD 토큰
        ]
        return StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    def generate_answer(self, history, image_input=None):
        try:
            # 이미지 처리가 필요한 경우 여기에 추가
            if image_input:
                image = Image.open(image_input).convert('RGB')
            else:
                image = None
            # 메시지 처리
            prompt_messages = [{"role": msg['role'], "image": msg['image'], "content": msg['content']} for msg in history]
            logger.info(f"[*] Prompt messages for GLM: {prompt_messages}")
            
            # 입력 처리
            inputs = self.tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True, 
                tokenize=True, 
                return_tensors="pt",
                return_dict=True
            ).to(self.model.device)
            logger.info("[*] GLM input template applied successfully")
            
            # 생성 설정
            generation_config = {
                "max_new_tokens": 1024,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.8,
                "repetition_penalty": 1.2,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "stopping_criteria": self.get_stopping_criteria()
            }
            
            # 텍스트 생성
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
            logger.info("[*] GLM model generated the response")
            
            # 결과 처리
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            logger.info(f"[*] Generated text: {generated_text}")
            
            return generated_text.strip()
            
        except Exception as e:
            error_msg = f"Error during GLM answer generation: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg