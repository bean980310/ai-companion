# model_handlers/minicpm_llama3_v2_5.py
import os
import torch
from transformers import AutoTokenizer, AutoModel, ProcessorMixin
import traceback
import logging
from PIL import Image
from src.common.utils import make_local_dir_name

logger = logging.getLogger(__name__)

class MiniCPMLlama3V25Handler:
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
            self.model = AutoModel.from_pretrained(
                self.model_dir,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(self.device).eval()
            logger.info(f"[*] Model loaded successfully: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load MiniCPM-Llama3-V-2_5 model: {str(e)}\n\n{traceback.format_exc()}")
            raise

    def generate_answer(self, history, image_input=None):
        try:
            # 이미지 처리
            if image_input is not None:
                try:
                    # 이미지가 이미 PIL Image 객체인 경우
                    if isinstance(image_input, Image.Image):
                        image = image_input
                    # 이미지 경로나 파일인 경우
                    else:
                        image = Image.open(image_input).convert('RGB')
                    logger.info("[*] Image processed successfully")
                except Exception as img_error:
                    logger.error(f"Error processing image: {str(img_error)}")
                    return f"Error processing image: {str(img_error)}"
            else:
                logger.info("[*] No image provided")
                return "이미지가 필요합니다. 이미지를 업로드해주세요."

            # 메시지 처리
            messages = []
            for msg in history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    messages.append({
                        'role': msg['role'],
                        'content': str(msg['content'])  # 내용을 문자열로 변환
                    })
                
            logger.info("[*] Generating response...")
            logger.info(f"Messages: {messages}")
            
            outputs = self.model.chat(
                image,  # PIL Image 직접 전달
                messages,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=0.7
            )
            
            # 결과 텍스트 생성
            generated_text = ""
            try:
                for new_text in outputs:
                    generated_text += new_text
            except TypeError:
                generated_text = outputs
                
            logger.info(f"[*] Generated text: {generated_text}")
            return generated_text
            
        except Exception as e:
            error_msg = f"Error during answer generation: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg