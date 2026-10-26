"""gateway 기반 전체 모델 카탈로그.

provider 드롭다운에서 "All"을 선택했을 때, gateway에 등록된 모든
프로바이더(및 ``custom:*`` 프로필)의 모델 목록을 합쳐서 보여주기 위한 모듈.

- 목록을 만들면서 모델명 -> provider 매핑을 캐시한다.
- 모델을 선택하면 이 매핑으로 provider 드롭다운을 자동 전환한다.
- provider가 여전히 "All"인 채로 생성이 시작되면 매핑으로 실제 provider를
  해석해 기존 생성 경로(load_model)가 그대로 동작하게 한다.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import ai_companion_gateway as gateway
from ai_companion_core import logger

from src.models.gateway_providers import register_custom_gateway_providers

# 모델명 -> provider (먼저 등록된 provider가 우선)
_model_provider_map: Dict[str, str] = {}
_catalog: Dict[str, List[str]] = {}


def refresh_all_models() -> Dict[str, List[str]]:
    """모든 gateway provider의 모델 목록을 다시 조회한다."""
    register_custom_gateway_providers()

    catalog = gateway.list_all_models()

    _catalog.clear()
    _catalog.update(catalog)

    _model_provider_map.clear()
    for provider, models in catalog.items():
        for model in models:
            _model_provider_map.setdefault(model, provider)

    total = sum(len(models) for models in catalog.values())
    logger.info(f"gateway 전체 모델 목록: {len(catalog)}개 provider, {total}개 모델")
    return dict(catalog)


def get_model_provider_map() -> Dict[str, str]:
    if not _model_provider_map:
        refresh_all_models()
    return dict(_model_provider_map)


def list_all_models() -> List[str]:
    """모든 provider의 모델을 중복 제거해 정렬된 목록으로 반환한다."""
    catalog = refresh_all_models()
    models = []
    for provider_models in catalog.values():
        models.extend(provider_models)
    return sorted(set(models))


def resolve_model_provider(model: str) -> Optional[str]:
    """모델명으로 provider를 찾는다 (없으면 None)."""
    if not _model_provider_map:
        refresh_all_models()
    return _model_provider_map.get(model)


def resolve_effective_provider(provider: str, selected_model: str) -> str:
    """``provider``가 "All"이면 선택된 모델의 실제 provider로 해석한다."""
    from src.models import ALL_PROVIDERS

    if provider != ALL_PROVIDERS:
        return provider
    return resolve_model_provider(selected_model) or provider


__all__ = [
    "get_model_provider_map",
    "list_all_models",
    "refresh_all_models",
    "resolve_effective_provider",
    "resolve_model_provider",
]
