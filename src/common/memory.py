"""
mem0 기반 장기기억(Long-Term Memory) 관리 모듈.

채팅 대화에서 사용자에 대한 지속적인 사실/선호/맥락을 추출해 저장하고,
이후 대화에서 관련 메모리를 검색해 시스템 프롬프트에 주입함으로써
세션을 넘어서도 사용자를 기억하는 기능을 제공한다.

백엔드는 두 가지를 지원한다:

1) Ollama (기본, API 키 불필요)
    - embedder : Ollama (기본 `nomic-embed-text`)
    - llm      : Ollama (메모리 추출/요약용, 기본 `gemma4:12b`)
    - vector store : ChromaDB (영구 로컬 저장)

2) 커스텀 OpenAI 호환 provider (base_url)
    - vLLM / LM Studio / Ollama OpenAI-호환 엔드포인트 등 임의의 base_url 사용
    - `MEMORY_BASE_URL` + `MEMORY_API_KEY` + `MEMORY_MODEL` 지정

구성은 `~/.ai-companion/.env` 의 환경변수로 변경할 수 있다:

    - MEMORY_ENABLED          : 장기기억 활성화 (기본 "true")

프로바이더는 LLM(메모리 추출용)과 embedder(임베딩용)를 분리해 지정할 수 있다:

    - MEMORY_PROVIDER         : 양쪽의 기본값 ("ollama" 또는 "custom"/"openai", 기본 "ollama")
    - MEMORY_LLM_PROVIDER     : 메모리 추출 LLM 프로바이더 (미지정 시 MEMORY_PROVIDER)
        지원: ollama, openai, vllm, lmstudio, deepseek, anthropic, gemini, groq, together, litellm, xai 등
    - MEMORY_EMBEDDER_PROVIDER: 임베딩 프로바이더 (미지정 시 MEMORY_PROVIDER)
        지원: ollama, openai, lmstudio, huggingface, gemini, together, fastembed 등

    - MEMORY_LLM_MODEL        : 메모리 추출 LLM (ollama 기본 "gemma4:12b")
    - MEMORY_LLM_BASE_URL     : LLM base_url (미지정 시 MEMORY_BASE_URL, 그 후 프로바이더별 기본값)
    - MEMORY_LLM_API_KEY      : LLM API 키 (미지정 시 MEMORY_API_KEY)
    - MEMORY_EMBEDDER_MODEL   : 임베딩 모델 (ollama 기본 "nomic-embed-text")
    - MEMORY_EMBEDDER_BASE_URL: embedder base_url (미지정 시 MEMORY_BASE_URL)
    - MEMORY_EMBEDDER_API_KEY : embedder API 키 (미지정 시 MEMORY_API_KEY)
    - MEMORY_EMBEDDER_DIMS    : 임베딩 차원 수 (프로바이더 기본값과 다른 모델 사용 시 지정)
    - OLLAMA_BASE_URL         : (하위 호환) Ollama 서버 주소 — 현재는 사용하지 않고
                                MEMORY_LLM_BASE_URL / MEMORY_EMBEDDER_BASE_URL 을 사용한다

    - MEMORY_VECTOR_STORE     : 벡터스토어 (기본 "chroma", 대안 "faiss")
    - MEMORY_COLLECTION       : 벡터스토어 컬렉션명 (기본 "mem0")
    - MEMORY_STORE_PATH       : 벡터스토어 저장 경로 (기본 "<appdata>/mem0")
    - MEMORY_MAX_TOKENS       : 메모리 주입 시 최대 토큰 수 (기본 2000)

예시:
    # 전부 로컬 Ollama (기본)
    MEMORY_PROVIDER=ollama

    # 임베딩은 로컬 Ollama, 메모리 추출 LLM은 vLLM 서버
    MEMORY_EMBEDDER_PROVIDER=ollama
    MEMORY_LLM_PROVIDER=vllm
    MEMORY_LLM_BASE_URL=http://localhost:8000/v1
    MEMORY_LLM_MODEL=Qwen/Qwen3-32B

    # 전부 LM Studio
    MEMORY_PROVIDER=lmstudio

참고: `ai_companion_core` 와 동일한 `~/.ai-companion/.env` 를 읽는다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from ai_companion_core import logger
from ai_companion_core.environ_manager import load_env_variables
from ai_companion_core.appdata import APPDATA_PATH

# mem0 는 import 시점에 초기화하지 않고, 설정이 활성화된 경우에만 지연 로딩한다.
# (기본 OpenAI embedder 가 API 키 없이 실패하는 것을 방지하기 위함)
_memory: Optional[Any] = None
_memory_enabled: bool = False
_ui_toggle: Optional[bool] = None


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env(key: str, default: str) -> str:
    # 프로세스 환경변수를 우선시하고, 없으면 ~/.ai-companion/.env 를 읽는다.
    value = os.environ.get(key)
    if value:
        return value
    value = load_env_variables(key)
    return value if value else default


def _resolve_provider(side: str) -> str:
    """LLM/embedder 별 프로바이더를 결정한다.

    우선순위: MEMORY_<SIDE>_PROVIDER > MEMORY_PROVIDER > "ollama"
    "custom" 은 "openai" (OpenAI 호환 base_url) 의 별칭으로 처리한다.
    """
    provider = _env(f"MEMORY_{side}_PROVIDER", "") or _env("MEMORY_PROVIDER", "ollama")
    provider = provider.strip().lower()
    if provider == "custom":
        provider = "openai"
    return provider or "ollama"


def _llm_config_class(provider: str):
    from mem0.utils.factory import LlmFactory

    entry = LlmFactory.provider_to_class.get(provider)
    if entry is None:
        raise ValueError(f"지원하지 않는 MEMORY_LLM_PROVIDER: {provider} (지원 목록: {', '.join(LlmFactory.provider_to_class)})")
    return entry[1]


def _llm_base_url_key(provider: str) -> str:
    return {
        "ollama": "ollama_base_url",
        "vllm": "vllm_base_url",
        "lmstudio": "lmstudio_base_url",
        "deepseek": "deepseek_base_url",
    }.get(provider, "openai_base_url")


def _build_llm_config(provider: str, max_tokens: int) -> dict:
    """mem0 LLM 설정을 구성한다. 프로바이더 config 클래스가 받는 키만 전달한다."""
    import inspect

    config_class = _llm_config_class(provider)
    accepted = set(inspect.signature(config_class.__init__).parameters.keys())

    # base_url 기본값: 프로바이더별 로컬 기본 포트
    default_base_urls = {
        "ollama": "http://localhost:11434",
        "vllm": "http://localhost:8000/v1",
        "lmstudio": "http://localhost:1234/v1",
    }
    base_url = _env("MEMORY_LLM_BASE_URL", "") or _env("MEMORY_BASE_URL", "") or default_base_urls.get(provider, "")
    api_key = _env("MEMORY_LLM_API_KEY", "") or _env("MEMORY_API_KEY", "")
    model = _env("MEMORY_LLM_MODEL", "") or ("gemma4:12b" if provider == "ollama" else "")

    kwargs: dict[str, Any] = {
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if model:
        kwargs["model"] = model
    if base_url:
        kwargs[_llm_base_url_key(provider)] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    elif provider == "ollama":
        kwargs["api_key"] = "not-needed"

    return {k: v for k, v in kwargs.items() if k in accepted}


def _build_embedder_config(provider: str) -> dict:
    """mem0 embedder 설정을 구성한다.

    embedder 는 BaseEmbedderConfig 하나로 수렴하므로 프로바이더별 base_url 키만
    올바르게 매핑하면 된다 (ollama / lmstudio / 그 외 OpenAI 호환).
    """
    from mem0.utils.factory import EmbedderFactory

    if provider not in EmbedderFactory.provider_to_class:
        raise ValueError(f"지원하지 않는 MEMORY_EMBEDDER_PROVIDER: {provider} (지원 목록: {', '.join(EmbedderFactory.provider_to_class)})")

    base_url_key = {
        "ollama": "ollama_base_url",
        "lmstudio": "lmstudio_base_url",
    }.get(provider, "openai_base_url")

    default_base_urls = {
        "ollama": "http://localhost:11434",
        "lmstudio": "http://localhost:1234/v1",
    }
    base_url = _env("MEMORY_EMBEDDER_BASE_URL", "") or _env("MEMORY_BASE_URL", "") or default_base_urls.get(provider, "")
    api_key = _env("MEMORY_EMBEDDER_API_KEY", "") or _env("MEMORY_API_KEY", "")
    model = _env("MEMORY_EMBEDDER_MODEL", "") or ("nomic-embed-text" if provider == "ollama" else "")

    kwargs: dict[str, Any] = {}
    if model:
        kwargs["model"] = model
    if base_url:
        kwargs[base_url_key] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    dims = _env("MEMORY_EMBEDDER_DIMS", "")
    if dims:
        kwargs["embedding_dims"] = int(dims)

    return kwargs


def _build_config() -> dict:
    """mem0 MemoryConfig 를 dict 형태로 구성한다.

    LLM 과 embedder 의 프로바이더를 분리해 지정할 수 있다:
        - MEMORY_LLM_PROVIDER       (기본: MEMORY_PROVIDER, 최종 기본 "ollama")
        - MEMORY_EMBEDDER_PROVIDER  (기본: MEMORY_PROVIDER, 최종 기본 "ollama")
    각각의 base_url / api_key / 모델은 MEMORY_LLM_* / MEMORY_EMBEDDER_* 변수로
    개별 지정하며, 지정하지 않으면 공통 MEMORY_BASE_URL / MEMORY_API_KEY 에
    fallback 한다.
    """
    llm_provider = _resolve_provider("LLM")
    embedder_provider = _resolve_provider("EMBEDDER")
    vector_store = _env("MEMORY_VECTOR_STORE", "chroma").lower()
    collection = _env("MEMORY_COLLECTION", "mem0")
    store_path = _env("MEMORY_STORE_PATH", str(APPDATA_PATH / "mem0"))
    max_tokens = int(_env("MEMORY_MAX_TOKENS", "2000"))

    Path(store_path).mkdir(parents=True, exist_ok=True)

    llm_config = _build_llm_config(llm_provider, max_tokens)
    embedder_config = _build_embedder_config(embedder_provider)

    if vector_store == "faiss":
        vector_store_config = {
            "provider": "faiss",
            "config": {
                "collection_name": collection,
                "path": str(Path(store_path) / "faiss"),
            },
        }
    else:  # 기본 chroma
        vector_store_config = {
            "provider": "chroma",
            "config": {
                "collection_name": collection,
                "path": str(Path(store_path) / "chroma"),
            },
        }

    return {
        "vector_store": vector_store_config,
        "llm": {"provider": llm_provider, "config": llm_config},
        "embedder": {"provider": embedder_provider, "config": embedder_config},
    }


def is_memory_enabled() -> bool:
    """장기기억 기능 활성화 여부. UI 토글 상태를 우선하며, 없으면 환경변수를 따른다."""
    if _ui_toggle is not None:
        return _ui_toggle
    return _as_bool(load_env_variables("MEMORY_ENABLED"), default=True)


def set_memory_enabled(enabled: bool) -> None:
    """UI에서 장기기억 사용 여부를 런타임에 전환한다."""
    global _ui_toggle
    _ui_toggle = bool(enabled)
    logger.info("장기기억(mem0) %s", "활성화" if enabled else "비활성화")


def get_memory() -> Optional[Any]:
    """전역 mem0 Memory 인스턴스를 반환한다 (지연 초기화).

    장기기억이 비활성화되었거나 초기화에 실패하면 None 을 반환한다.
    """
    global _memory, _memory_enabled

    if _memory is not None:
        return _memory

    if _memory_enabled:
        return None

    if not is_memory_enabled():
        _memory_enabled = True
        logger.info("장기기억(mem0)이 비활성화되어 있습니다.")
        return None

    try:
        from mem0 import Memory

        config = _build_config()
        _memory = Memory.from_config(config)
        logger.info("장기기억(mem0) 초기화 완료: %s / %s", config["embedder"]["provider"], config["vector_store"]["provider"])
        return _memory
    except Exception as e:
        _memory_enabled = True  # 재시도 방지
        logger.error(f"장기기억(mem0) 초기화 실패: {e}")
        return None


def reset_memory() -> None:
    """전역 메모리 인스턴스를 초기화한다 (설정 변경 시 재구성용)."""
    global _memory, _memory_enabled
    _memory = None
    _memory_enabled = False


def _memory_namespace(character: Optional[str]) -> Optional[str]:
    """mem0 user_id 로 사용할 안전한 식별자로 변환한다.

    mem0 는 user_id 에 공백을 포함한 식별자를 허용하지 않으므로
    (Invalid user_id: cannot contain whitespace), 공백을 언더스코어로 치환한다.
    add/search/get_all/clear 모두 이 헬퍼를 거치므로 네임스페이스가 일관되게 유지된다.
    """
    if not character:
        return None
    sanitized = "_".join(str(character).split())
    return sanitized or None


def add_memory(
    user_message: str,
    assistant_message: str,
    character: Optional[str] = None,
    session_id: Optional[str] = None,
) -> bool:
    """대화 한 턴을 장기기억에 저장한다.

    Args:
        user_message: 사용자 메시지 (텍스트)
        assistant_message: 봇 응답 (텍스트)
        character: 캐릭터(페르소나) 이름. 메모리를 캐릭터별로 분리하는 네임스페이스.
        session_id: 채팅 세션 ID (메타데이터로 저장)

    Returns:
        성공 여부
    """
    mem = get_memory()
    if mem is None:
        return False

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]
    metadata = {}
    if session_id:
        metadata["session_id"] = session_id

    try:
        mem.add(messages, user_id=_memory_namespace(character), metadata=metadata)
        return True
    except Exception as e:
        logger.error(f"장기기억 저장 실패: {e}")
        return False


def search_memory(
    query: str,
    character: Optional[str] = None,
    top_k: int = 10,
    threshold: float = 0.1,
) -> list[str]:
    """질의와 관련된 장기기억을 검색한다.

    Args:
        query: 검색 질의 (현재 사용자 메시지 등)
        character: 캐릭터 네임스페이스. 지정 시 해당 캐릭터 메모리만 검색.
        top_k: 반환할 메모리 개수
        threshold: 유사도 임계값

    Returns:
        관련 메모리 텍스트 목록
    """
    mem = get_memory()
    if mem is None:
        return []

    try:
        filters = {"user_id": _memory_namespace(character)} if character else None
        raw = mem.search(query, filters=filters, top_k=top_k, threshold=threshold)
        results = raw.get("results", raw) if isinstance(raw, dict) else raw
        memories = []
        for r in results:
            text = r.get("memory") if isinstance(r, dict) else r
            if text:
                memories.append(text)
        return memories
    except Exception as e:
        logger.error(f"장기기억 검색 실패: {e}")
        return []


def get_all_memories(character: Optional[str] = None) -> list[str]:
    """저장된 장기기억 전체를 반환한다."""
    mem = get_memory()
    if mem is None:
        return []

    try:
        filters = {"user_id": _memory_namespace(character)} if character else None
        raw = mem.get_all(filters=filters)
        results = raw.get("results", raw) if isinstance(raw, dict) else raw
        return [r.get("memory") if isinstance(r, dict) else r for r in results if r]
    except Exception as e:
        logger.error(f"장기기억 조회 실패: {e}")
        return []


def delete_memory(memory_id: str) -> bool:
    """특정 메모리를 삭제한다."""
    mem = get_memory()
    if mem is None:
        return False

    try:
        mem.delete(memory_id)
        return True
    except Exception as e:
        logger.error(f"장기기억 삭제 실패: {e}")
        return False


def clear_character_memories(character: Optional[str] = None) -> int:
    """캐릭터(또는 전체)의 장기기억을 모두 삭제하고 삭제된 개수를 반환한다."""
    mem = get_memory()
    if mem is None:
        return 0

    try:
        filters = {"user_id": _memory_namespace(character)} if character else None
        raw = mem.get_all(filters=filters, top_k=1000)
        results = raw.get("results", raw) if isinstance(raw, dict) else raw
        deleted = 0
        for r in results:
            memory_id = r.get("id") if isinstance(r, dict) else None
            if memory_id:
                try:
                    mem.delete(memory_id)
                    deleted += 1
                except Exception as e:
                    logger.warning(f"장기기억 개별 삭제 실패 ({memory_id}): {e}")
        logger.info("장기기억 %d개 삭제 완료 (character=%s)", deleted, character or "전체")
        return deleted
    except Exception as e:
        logger.error(f"장기기억 전체 삭제 실패: {e}")
        return 0


def build_memory_context(
    query: str,
    character: Optional[str] = None,
    top_k: int = 10,
    max_chars: int = 3000,
) -> str:
    """검색된 장기기억을 시스템 프롬프트에 주입할 텍스트 블록으로 구성한다.

    메모리가 없으면 빈 문자열을 반환한다.
    """
    memories = search_memory(query, character=character, top_k=top_k)
    if not memories:
        return ""

    lines = []
    used = 0
    for m in memories:
        if used + len(m) > max_chars:
            break
        lines.append(f"- {m}")
        used += len(m)

    block = "\n".join(lines)
    return f"\n\n[장기기억 (이전 대화에서 기억하는 사용자 정보)]\n{block}\n[장기기억 끝 - 이 정보를 대화에 자연스럽게 반영하되, 사용자가 언급하지 않는 한 장황하게 나열하지 마세요.]"
