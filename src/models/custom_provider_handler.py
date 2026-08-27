"""
사용자 정의 OpenAI 호환 Provider 핸들러

사용자가 설정한 base_url + api_key 를 사용하여
OpenAI 호환 API 엔드포인트에 연결합니다.
"""

import os
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
        self.base_url = base_url or "http://localhost:8000/v1"
        if self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

        # mem0의 기본 embedder가 OPENAI_API_KEY / OPENAI_BASE_URL 환경변수를 읽으므로,
        # custom provider의 자격증명을 주입해 초기화 실패(인증 오류)를 방지한다.
        os.environ.setdefault("OPENAI_API_KEY", api_key or "not-needed")
        os.environ.setdefault("OPENAI_BASE_URL", self.base_url)

        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        if self.max_length > 0:
            self.max_tokens = self.max_length

        self.client: Optional[OpenAI] = None
        self.load_model()

    def load_model(self):
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"사용자 정의 OpenAI 호환 클라이언트 초기화: {self.base_url}")

    @staticmethod
    def _normalize_content(content: Any) -> Any:
        """OpenAI 호환 API가 요구하는 content 형식으로 정규화한다.

        - 문자열 content → `[{"type": "text", "text": ...}]`
        - 리스트 내부의 원시 문자열 → `{"type": "text", "text": ...}` 로 변환
        - 이미 dict 콘텐츠 파트인 경우 그대로 유지
        """
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            normalized = []
            for part in content:
                if isinstance(part, str):
                    normalized.append({"type": "text", "text": part})
                else:
                    normalized.append(part)
            return normalized
        return content

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs) -> str:
        messages = [
            {"role": msg["role"], "content": self._normalize_content(msg["content"])}
            for msg in history
        ]

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