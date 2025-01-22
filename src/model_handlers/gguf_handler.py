# model_handlers/gguf_handler.py

import logging
from llama_cpp import Llama # gguf 모델을 로드하기 위한 라이브러리
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
import os

class GGUFModelHandler:
    def __init__(self, model_id, quantization_bit="qint8", local_model_path=None, model_type="gguf"):
        """
        GGUF 모델 핸들러 초기화
        """
        self.model_id = model_id
        self.quantization_bit = quantization_bit
        self.model_type = model_type
        self.local_model_path = local_model_path or os.path.join("./models", model_type, self.make_local_dir_name(model_id, quantization_bit))
        self.llm = None
        self.load_model()
    
    def make_local_dir_name(self, model_id, quantization_bit):
        """
        모델 ID와 양자화 비트를 기반으로 로컬 디렉토리 이름 생성
        """
        return f"{model_id.replace('/', '_')}_{quantization_bit}"
    
    def load_model(self):
        """
        GGUF 모델 로드
        """
        logging.info(f"GGUF 모델 로드 시작: {self.local_model_path}")
        try:
            self.llm = Llama(
                model_path=self.local_model_path,
                n_ctx=2048,
                n_threads=4,
                # 필요에 따라 추가 매개변수 설정
            )
            logging.info("GGUF 모델 로드 성공")
        except Exception as e:
            logging.error(f"GGUF 모델 로드 실패: {str(e)}")
            raise e
    
    def generate_answer(self, history):
        """
        사용자 히스토리를 기반으로 답변 생성
        """
        prompt = self.history_to_prompt(history)
        try:
            response = self.llm(prompt, max_tokens=128)
            return response
        except Exception as e:
            logging.error(f"GGUF 모델 추론 오류: {str(e)}")
            return f"오류 발생: {str(e)}"
    
    def history_to_prompt(self, history):
        """
        대화 히스토리를 프롬프트로 변환
        """
        prompt = ""
        for message in history:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant:"
        return prompt