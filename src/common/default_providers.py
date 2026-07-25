"""
기본 Provider 설정 파일 관리 모듈

~/.ai-companion/default_providers.yaml 파일을 통해
LLM과 이미지 생성의 기본 provider를 관리합니다.
"""

from pathlib import Path
from typing import TypedDict

from src.common.apppath import APPDATA_PATH

from ai_companion_core import logger


CONFIG_FILE = APPDATA_PATH / "default_providers.yaml"

DEFAULT_CONFIG = {
    "llm": {
        "default_provider": "self-provided",
    },
    "image": {
        "default_provider": "self-provided",
    },
}


class ProviderConfig(TypedDict):
    default_provider: str


class DefaultProvidersConfig(TypedDict):
    llm: ProviderConfig
    image: ProviderConfig


def _ensure_yaml():
    """PyYAML이 설치되어 있는지 확인"""
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        logger.warning("PyYAML이 설치되지 않았습니다. 기본값을 사용합니다.")
        return False


def load_default_providers() -> DefaultProvidersConfig:
    """
    설정 파일에서 기본 provider 설정을 읽어옵니다.
    파일이 없으면 기본값으로 자동 생성합니다.

    Returns:
        DefaultProvidersConfig: LLM과 Image의 기본 provider 설정
    """
    if not _ensure_yaml():
        return DEFAULT_CONFIG

    import yaml

    if not CONFIG_FILE.exists():
        save_default_providers(DEFAULT_CONFIG)
        logger.info(f"기본 provider 설정 파일 생성: {CONFIG_FILE}")
        return DEFAULT_CONFIG

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        # 누락된 키에 기본값 채우기
        result = DEFAULT_CONFIG.copy()
        if "llm" in config and isinstance(config["llm"], dict):
            result["llm"] = {**DEFAULT_CONFIG["llm"], **config["llm"]}
        if "image" in config and isinstance(config["image"], dict):
            result["image"] = {**DEFAULT_CONFIG["image"], **config["image"]}

        return result

    except Exception as e:
        logger.error(f"설정 파일 읽기 실패, 기본값 사용: {e}")
        return DEFAULT_CONFIG


def save_default_providers(config: DefaultProvidersConfig) -> None:
    """
    기본 provider 설정을 파일에 저장합니다.

    Args:
        config: 저장할 설정
    """
    if not _ensure_yaml():
        return

    import yaml

    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write("# AI Companion Default Provider Configuration\n")
            f.write("# 앱 시작 시 이 설정에 지정된 provider의 모델 목록만 초기 로딩합니다.\n")
            f.write("# 나머지 provider는 UI에서 선택할 때 on-demand로 로딩됩니다.\n\n")
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"기본 provider 설정 저장 완료: {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"설정 파일 저장 실패: {e}")


def get_default_llm_provider() -> str:
    """기본 LLM provider를 반환합니다."""
    config = load_default_providers()
    return config["llm"]["default_provider"]


def get_default_image_provider() -> str:
    """기본 Image provider를 반환합니다."""
    config = load_default_providers()
    return config["image"]["default_provider"]
