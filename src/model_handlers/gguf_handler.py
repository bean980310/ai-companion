# model_handlers/gguf_handler.py
import faulthandler
faulthandler.enable()
import logging
import llama_cpp
from llama_cpp import Llama # gguf 모델을 로드하기 위한 라이브러리
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
import os

logger = logging.getLogger(__name__)

class GGUFModelHandler:
    def __init__(self, model_id, local_model_path=None, model_type="gguf"):
        """
        GGUF 모델 핸들러 초기화
        """
        self.model_id = model_id
        self.model_type = model_type
        self.local_model_path = local_model_path or os.path.join("./models/llm", model_id)
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        GGUF 모델 로드
        """
        logging.info(f"GGUF 모델 로드 시작: {self.local_model_path}")
        try:
            self.model = Llama(
                model_path=self.local_model_path,
                n_gpu_layers=-1,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
                logits_all=True
                # 필요에 따라 추가 매개변수 설정
            )
            logging.info("GGUF 모델 로드 성공")
        except Exception as e:
            logging.error(f"GGUF 모델 로드 실패: {str(e)}")
            raise e
    
    def generate_answer(self, history, temperature, top_k, top_p, repetition_penalty):
        """
        사용자 히스토리를 기반으로 답변 생성
        """
        prompt = [{"role": msg['role'], "content": msg['content']} for msg in history]
        try:
            response = self.model.create_chat_completion(
                messages=prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repetition_penalty
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"GGUF 모델 추론 오류: {str(e)}")
            return f"오류 발생: {str(e)}"
        
    def generate_chat_title(self, first_message: str)->str:
        prompt=(
            "Summarize the following message in one sentence and create an appropriate chat title:\n\n"
            f"{first_message}\n\n"
            "Chat Title:"
        )
        
        logger.info(f"채팅 제목 생성 프롬프트: {prompt}")
        title_response=self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20
        )
        
        title = title_response["choices"][0]["message"]["content"]
        logger.info(f"생성된 채팅 제목: {title}")
        return title