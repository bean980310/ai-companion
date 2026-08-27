"""
사용자 정의 Provider 프로필 관리 모듈

~/.ai-companion/custom_providers.yaml 파일을 통해
OpenAI 호환 API 엔드포인트 프로필을 관리합니다.

각 프로필은 이름, base_url, api_key를 가집니다.
드롭다운에서는 "custom:<이름>" 형태로 표시되어 기존 provider와 구분됩니다.
"""

from typing import Optional, TypedDict

from src.common.apppath import APPDATA_PATH

from ai_companion_core import logger


CONFIG_FILE = APPDATA_PATH / "custom_providers.yaml"

# 드롭다운/내부 식별용 접두사
CUSTOM_PREFIX = "custom:"


class CustomProvider(TypedDict):
    name: str
    base_url: str
    api_key: str


def _ensure_yaml():
    """PyYAML이 설치되어 있는지 확인"""
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        logger.warning("PyYAML이 설치되지 않았습니다.")
        return False


def _normalize_base_url(base_url: str) -> str:
    """base_url 후행 슬래시 제거 및 /v1 자동 보정."""
    url = base_url.strip().rstrip("/")
    if not url:
        return url
    # 이미 /v1 로 끝나면 그대로, 아니면 OpenAI 호환 기본 경로를 붙임
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def load_custom_providers() -> list[CustomProvider]:
    """
    설정 파일에서 사용자 정의 provider 목록을 읽어옵니다.
    파일이 없으면 빈 목록을 반환합니다.
    """
    if not _ensure_yaml():
        return []

    import yaml

    if not CONFIG_FILE.exists():
        return []

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return []

        providers = data.get("providers", [])
        if not isinstance(providers, list):
            return []

        result: list[CustomProvider] = []
        for item in providers:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            result.append(
                {
                    "name": str(item["name"]).strip(),
                    "base_url": str(item.get("base_url", "")).strip(),
                    "api_key": str(item.get("api_key", "")).strip(),
                }
            )
        return result

    except Exception as e:
        logger.error(f"사용자 정의 provider 설정 읽기 실패: {e}")
        return []


def save_custom_providers(providers: list[CustomProvider]) -> None:
    """사용자 정의 provider 목록을 파일에 저장합니다."""
    if not _ensure_yaml():
        return

    import yaml

    try:
        data = {"providers": providers}
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write("# AI Companion Custom Provider Configuration\n")
            f.write("# 각 항목은 OpenAI 호환 API 엔드포인트를 나타냅니다.\n")
            f.write("# 드롭다운에서는 'custom:<이름>' 형태로 표시됩니다.\n\n")
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"사용자 정의 provider 설정 저장 완료: {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"사용자 정의 provider 설정 저장 실패: {e}")


def get_custom_provider(name: str) -> Optional[CustomProvider]:
    """이름으로 사용자 정의 provider를 찾습니다."""
    for provider in load_custom_providers():
        if provider["name"] == name:
            return provider
    return None


def get_custom_provider_names() -> list[str]:
    """사용자 정의 provider 이름 목록을 반환합니다."""
    return [p["name"] for p in load_custom_providers()]


def get_custom_provider_choices() -> list[str]:
    """드롭다운 선택지(custom:<이름>) 목록을 반환합니다."""
    return [f"{CUSTOM_PREFIX}{name}" for name in get_custom_provider_names()]


def add_custom_provider(name: str, base_url: str, api_key: str) -> tuple[bool, str]:
    """새 사용자 정의 provider를 추가합니다."""
    name = name.strip()
    base_url = _normalize_base_url(base_url)
    api_key = api_key.strip()

    if not name:
        return False, "이름을 입력해주세요."
    if not base_url:
        return False, "Base URL을 입력해주세요."

    providers = load_custom_providers()
    if any(p["name"] == name for p in providers):
        return False, f"이미 존재하는 프로필 이름입니다: {name}"

    providers.append({"name": name, "base_url": base_url, "api_key": api_key})
    save_custom_providers(providers)
    return True, f"프로필 '{name}' 추가 완료."


def update_custom_provider(name: str, base_url: str, api_key: str) -> tuple[bool, str]:
    """기존 사용자 정의 provider를 수정합니다."""
    name = name.strip()
    base_url = _normalize_base_url(base_url)
    api_key = api_key.strip()

    providers = load_custom_providers()
    for provider in providers:
        if provider["name"] == name:
            provider["base_url"] = base_url
            provider["api_key"] = api_key
            save_custom_providers(providers)
            return True, f"프로필 '{name}' 수정 완료."
    return False, f"프로필을 찾을 수 없습니다: {name}"


def delete_custom_provider(name: str) -> tuple[bool, str]:
    """사용자 정의 provider를 삭제합니다."""
    providers = load_custom_providers()
    remaining = [p for p in providers if p["name"] != name]
    if len(remaining) == len(providers):
        return False, f"프로필을 찾을 수 없습니다: {name}"
    save_custom_providers(remaining)
    return True, f"프로필 '{name}' 삭제 완료."