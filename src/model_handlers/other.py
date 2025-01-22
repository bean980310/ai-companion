# common.py 상단에 추가
import os
import torch
import logging
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.common.utils import get_terminators, make_local_dir_name

logger = logging.getLogger(__name__)

class OtherModelHandler:
    def __init__(self, model_id, local_model_path=None, model_type="transformers", device='cpu'):
        self.model_dir = local_model_path or os.path.join("./models", model_type, make_local_dir_name(model_id))
        self.tokenizer = None
        self.model = None
        self.device = device
        self.load_model()
    def load_model(self):
        try:
            logger.info(f"[*] Loading tokenizer from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            logger.info(f"[*] Loading model from {self.model_dir}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(self.device)
            logger.info(f"[*] Model loaded successfully: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load GLM4 Model: {str(e)}\n\n{traceback.format_exc()}")
            raise
    def generate_answer(self, history):
        prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
        logger.info(f"[*] Prompt messages for other models: {prompt_messages}")
        
        terminators = get_terminators(self.tokenizer)
        try:
            input_ids = self.tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            logger.info("[*] 입력 템플릿 적용 완료")
        except Exception as e:
            logger.error(f"입력 템플릿 적용 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return f"입력 템플릿 적용 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}"

        try:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
            logger.info("[*] 모델 생성 완료")
        except Exception as e:
            logger.error(f"모델 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return f"모델 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}"

        try:
            generated_text = self.tokenizer.decode(
                outputs[0][input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            logger.info(f"[*] 생성된 텍스트: {generated_text}")
        except Exception as e:
            logger.error(f"출력 디코딩 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return f"출력 디코딩 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}"

        return generated_text.strip()