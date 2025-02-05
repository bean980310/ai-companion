import os
import torch
import logging
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from optimum.quanto import QuantizedModelForCausalLM
from src.common.utils import make_local_dir_name

logger = logging.getLogger(__name__)

class QwenHandler:
    def __init__(self, model_id, lora_model_id=None, local_model_path=None, lora_path=None, model_type="transformers", device='cpu'):
        self.model_dir = local_model_path or os.path.join("./models/llm", model_type, make_local_dir_name(model_id))
        self.lora_model_dir = lora_path or os.path.join("./model/llm/loras", make_local_dir_name(lora_model_id))
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
            ).to(self.device)
            if self.lora_model_dir and os.path.exists(self.lora_model_dir):
                logger.info(f"[*] Loading LoRA from {self.lora_model_dir}")
                self.model=PeftModel.from_pretrained(
                    self.model_dir,
                    self.lora_model_dir,
                ).to(self.device)
            logger.info(f"[*] Model loaded successfully: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load Qwen Model: {str(e)}\n\n{traceback.format_exc()}")
            raise
    def generate_answer(self, history):
        prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
        logger.info(f"[*] Prompt messages for other models: {prompt_messages}")
        
        try:
            text = self.tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=False
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            logger.info("[*] 입력 템플릿 적용 완료")
        except Exception as e:
            logger.error(f"입력 템플릿 적용 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return f"입력 템플릿 적용 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}"

        try:
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=0.8
            )
            logger.info("[*] 모델 생성 완료")
        except Exception as e:
            logger.error(f"모델 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return f"모델 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}"

        try:
            outputs=[
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
            ]
            generated_text = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0]
            logger.info(f"[*] 생성된 텍스트: {generated_text}")
        except Exception as e:
            logger.error(f"출력 디코딩 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return f"출력 디코딩 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}"

        return generated_text.strip()