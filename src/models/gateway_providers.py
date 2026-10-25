"""ai-companion 앱의 gateway 프로바이더 등록 헬퍼.

``~/.ai-companion/custom_providers.yaml`` 의 OpenAI 호환 프로필을
``ai-companion-gateway`` 의 클라이언트 빌더 + OpenAI 엔드포인트로 등록한다.

모델 로드(``src.models.models``)와 MCP tool-calling(``src.mcp.agent``)이
같은 등록 로직을 공유하기 위한 모듈이다.
"""

from __future__ import annotations

from typing import Any, Dict

from ai_companion_core import logger
import ai_companion_gateway as gateway

from src.common.custom_providers import CUSTOM_PREFIX, get_custom_provider, get_custom_provider_names
from src.models.custom_provider_handler import CustomOpenAIClientWrapper


def _build_custom_client(profile: Dict[str, Any]):
    """프로필 하나에 대한 gateway 클라이언트 빌더를 만든다."""

    def _build(selected_model: str, api_key: str | None = None, _profile: Dict[str, Any] = profile, **kwargs: Any):
        return CustomOpenAIClientWrapper(
            selected_model,
            api_key=api_key or _profile.get("api_key") or "not-needed",
            base_url=_profile.get("base_url"),
            **kwargs,
        )

    return _build


def register_custom_gateway_providers(*, refresh: bool = False) -> list[str]:
    """yaml 프로필을 gateway에 등록하고 등록된 provider id 목록을 반환한다.

    Args:
        refresh: True면 기존 등록을 해제하고 현재 yaml 내용으로 다시 등록한다.
            False면 아직 등록되지 않은 프로필만 추가한다.
    """
    profiles: Dict[str, Dict[str, Any]] = {}
    for name in get_custom_provider_names():
        profile = get_custom_provider(name)
        if profile:
            profiles[f"{CUSTOM_PREFIX}{name}"] = profile

    # yaml에서 삭제된 프로필은 gateway에서도 제거한다.
    for provider_id in gateway.supported_providers():
        if provider_id.startswith(CUSTOM_PREFIX) and provider_id not in profiles:
            gateway.unregister_client_builder(provider_id)
    for provider_id in gateway.list_openai_endpoints():
        if provider_id.startswith(CUSTOM_PREFIX) and provider_id not in profiles:
            gateway.unregister_openai_endpoint(provider_id)

    for provider_id, profile in profiles.items():
        if refresh:
            gateway.unregister_client_builder(provider_id)
            gateway.unregister_openai_endpoint(provider_id)

        if provider_id not in gateway.supported_providers():
            gateway.register_client_builder(provider_id, _build_custom_client(profile), requires_api_key=False)
            logger.info(f"gateway에 사용자 정의 provider 등록: {provider_id}")

        if not gateway.supports_openai_tools(provider_id):
            gateway.register_openai_endpoint(
                provider_id,
                profile.get("base_url"),
                requires_api_key=False,
                default_key=profile.get("api_key") or "not-needed",
            )

    return list(profiles)


def ensure_gateway_provider(provider: str) -> None:
    """``custom:<name>`` 프로바이더가 gateway에 등록돼 있는지 보장한다."""
    if provider.startswith(CUSTOM_PREFIX):
        register_custom_gateway_providers()


__all__ = ["ensure_gateway_provider", "register_custom_gateway_providers"]
