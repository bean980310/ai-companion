"""
사용자 정의 OpenAI 호환 Provider 핸들러

사용자가 설정한 base_url + api_key 를 사용하여
OpenAI 호환 API 엔드포인트에 연결합니다.
"""

from typing import Any, Optional

from openai import OpenAI

from ai_companion_core import logger
from ai_companion_llm_backend.base_handlers import BaseAPIClientWrapper


class CustomOpenAIClientWrapper(BaseAPIClientWrapper):
    """
    사용자 정의 OpenAI 호환 API 서버용 클라이언트 래퍼.

    Args:
        selected_model: 사용할 모델 이름
        api_key: API 키
        base_url: OpenAI 호환 base_url (예: https://api.example.com/v1)
        use_langchain: LangchainIntegrator 사용 여부 (기본 False)
    """

    def __init__(
        self,
        selected_model: str,
        api_key: str = "not-needed",
        use_langchain: bool = False,
        base_url: Optional[str] = None,
        image_input: Any = None,
        **kwargs,
    ):
        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        self.base_url = base_url or "http://localhost:8000/v1"
        if self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

        if self.max_length > 0:
            self.max_tokens = self.max_length

        self.client: Optional[OpenAI] = None
        self.load_model()

    def load_model(self):
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"사용자 정의 OpenAI 호환 클라이언트 초기화: {self.base_url}")

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs) -> str:
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]

        if self.enable_streaming:
            return self._generate_streaming(messages)
        return self._generate_non_streaming(messages)

    def _generate_non_streaming(self, messages: list[dict]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed if self.seed != -1 else None,
            )
            answer = response.choices[0].message.content
            return answer.strip() if answer else ""
        except Exception as e:
            logger.error(f"사용자 정의 provider 생성 오류: {e}")
            raise

    def _generate_streaming(self, messages: list[dict]) -> str:
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed if self.seed != -1 else None,
                stream=True,
            )
            answer = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    answer += content
            print()
            return answer.strip()
        except Exception as e:
            logger.error(f"사용자 정의 provider 스트리밍 오류: {e}")
            raise


def get_custom_provider_models(base_url: str, api_key: str) -> list[str]:
    """
    사용자 정의 OpenAI 호환 엔드포인트에서 모델 목록을 가져옵니다.
    모델 목록 조회에 실패하면 오류 메시지를 포함한 목록을 반환합니다.
    """
    model_list = []

    if not base_url:
        model_list.append("Base URL이 필요합니다.")
        return model_list

    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)

    try:
        models = client.models.list()
        for m in models.data:
            model_list.append(m.id)
        logger.info(f"사용자 정의 provider 모델 목록: {len(model_list)}개")
        return model_list
    except Exception as e:
        logger.error(f"사용자 정의 provider 모델 목록 조회 오류: {e}")
        model_list.append(f"모델 목록 조회 실패: {e}")
        return model_list