# lorebook.py
"""SillyTavern 호환 로어북(월드 인포) 스키마 정규화 및 활성화 엔진.

지원 규격:
- 캐릭터 카드 V2/V3의 ``data.extensions.character_book`` (entries: list)
- SillyTavern World Info JSON (entries: {id: entry} 객체 형태, key/keysecondary 필드)

엔트리 의미 (내부 표준):
- ``keys``: 트리거 키워드. 최근 대화 텍스트에서 발견되면 활성화
- ``secondary_keys``: ``selective``가 True일 때 추가로 필요한 키워드 (AND ANY)
- ``constant``: True면 키워드와 무관하게 항상 활성화
- ``enabled``: False면 스캔에서 제외
- ``insertion_order``: 낮을수록 먼저(앞에) 삽입
- ``priority``: 높을수록 중요. 토큰 예산 초과 시 낮은 것부터 탈락
- ``position``: "before_char"(캐릭터 정의 앞) | "after_char"(캐릭터 정의 뒤)
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ai_companion_core import logger

from src.characters.character_card import apply_macros

# 북 기본값
DEFAULT_SCAN_DEPTH = 2          # 활성화 스캔에 사용할 최근 메시지 수
DEFAULT_TOKEN_BUDGET = 0        # 0 = 무제한
DEFAULT_RECURSIVE = True
MAX_RECURSION_PASSES = 3        # 재귀 활성화 최대 추가 패스 수

# 엔트리 기본값
DEFAULT_PRIORITY = 10
DEFAULT_INSERTION_ORDER = 100
VALID_POSITIONS = ("before_char", "after_char")

# 토큰 추정 계수: 대략 토큰 수 ≈ 글자 수 / TOKEN_CHARS_PER_TOKEN (CJK+영문 혼합 근사)
TOKEN_CHARS_PER_TOKEN = 3.0

# 북 레벨 키 별칭 (world info export ↔ card character_book)
BOOK_KEY_ALIASES = {
    "scan_depth": ("scan_depth",),
    "token_budget": ("token_budget", "budget"),
    "recursive_scanning": ("recursive_scanning", "recursive"),
}


class LorebookError(Exception):
    """로어북 파싱/처리 관련 예외"""

    pass


# ---------------------------------------------------------------------------
# 정규화
# ---------------------------------------------------------------------------

def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str_list(value: Any) -> List[str]:
    """키워드 리스트 정규화. 문자열이면 쉼표로 분리."""
    if value is None:
        return []
    if isinstance(value, str):
        return [k.strip() for k in value.split(",") if k.strip()]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
            elif isinstance(item, (int, float)):
                result.append(str(item))
        return result
    return []


def _normalize_position(value: Any) -> str:
    """position 정규화. world info 숫자 코드(0=before, 1=after)도 허용."""
    if isinstance(value, (int, float)):
        return "after_char" if int(value) == 1 else "before_char"
    text = str(value or "").strip().lower()
    if text in VALID_POSITIONS:
        return text
    # ST 확장 문자열 형태 방어
    if "after" in text:
        return "after_char"
    return "before_char"


def normalize_entry(raw: Any, entry_id: int) -> Optional[Dict[str, Any]]:
    """엔트리를 내부 표준 스키마로 정규화합니다.

    카드 character_book(keys/enabled)과 world info(key/keysecondary/disable)
    양쪽 형식을 모두 허용합니다. content가 없는 엔트리는 None 반환.
    """
    if not isinstance(raw, dict):
        return None

    content = str(raw.get("content") or "")
    if not content.strip():
        return None

    keys = _as_str_list(raw.get("keys") if raw.get("keys") is not None else raw.get("key"))
    secondary_keys = _as_str_list(
        raw.get("secondary_keys") if raw.get("secondary_keys") is not None else raw.get("keysecondary")
    )

    # enabled: 카드는 enabled, world info는 disable 사용
    if "enabled" in raw:
        enabled = _as_bool(raw.get("enabled"), True)
    else:
        enabled = not _as_bool(raw.get("disable"), False)

    selective = _as_bool(raw.get("selective"), bool(secondary_keys))

    return {
        "id": _as_int(raw.get("id"), entry_id),
        "keys": keys,
        "secondary_keys": secondary_keys,
        "content": content,
        "comment": str(raw.get("comment") or raw.get("name") or ""),
        "constant": _as_bool(raw.get("constant"), False),
        "selective": selective,
        "enabled": enabled,
        "insertion_order": _as_int(
            raw.get("insertion_order") if raw.get("insertion_order") is not None else raw.get("order"),
            DEFAULT_INSERTION_ORDER,
        ),
        "position": _normalize_position(raw.get("position")),
        "case_sensitive": _as_bool(raw.get("case_sensitive"), False),
        "priority": _as_int(raw.get("priority"), DEFAULT_PRIORITY),
        "extensions": raw.get("extensions") if isinstance(raw.get("extensions"), dict) else {},
    }


def _iter_entries(raw_entries: Any) -> List[Any]:
    """entries 필드를 리스트로 통일 (list 또는 {id: entry} 객체 모두 허용)."""
    if isinstance(raw_entries, list):
        return raw_entries
    if isinstance(raw_entries, dict):
        return list(raw_entries.values())
    return []


def _book_value(raw: Dict[str, Any], key: str, default: Any) -> Any:
    for alias in BOOK_KEY_ALIASES[key]:
        if alias in raw and raw[alias] is not None:
            return raw[alias]
    return default


def normalize_lorebook(raw: Any, name: str = "") -> Dict[str, Any]:
    """로어북(카드 character_book 또는 world info JSON)을 내부 표준 스키마로 정규화합니다."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LorebookError(f"로어북 JSON을 파싱할 수 없습니다: {e}")
    if not isinstance(raw, dict):
        return empty_lorebook(name)

    book_name = str(raw.get("name") or name or "")
    entries: List[Dict[str, Any]] = []
    for idx, raw_entry in enumerate(_iter_entries(raw.get("entries"))):
        entry = normalize_entry(raw_entry, idx)
        if entry:
            entries.append(entry)

    # id 재부여 (중복/누락 방지)
    for idx, entry in enumerate(entries):
        entry["id"] = idx

    return {
        "name": book_name,
        "scan_depth": max(1, _as_int(_book_value(raw, "scan_depth", DEFAULT_SCAN_DEPTH), DEFAULT_SCAN_DEPTH)),
        "token_budget": max(0, _as_int(_book_value(raw, "token_budget", DEFAULT_TOKEN_BUDGET), DEFAULT_TOKEN_BUDGET)),
        "recursive_scanning": _as_bool(_book_value(raw, "recursive_scanning", DEFAULT_RECURSIVE), DEFAULT_RECURSIVE),
        "entries": entries,
        "extensions": raw.get("extensions") if isinstance(raw.get("extensions"), dict) else {},
    }


def empty_lorebook(name: str = "") -> Dict[str, Any]:
    """빈 로어북을 생성합니다."""
    return {
        "name": name,
        "scan_depth": DEFAULT_SCAN_DEPTH,
        "token_budget": DEFAULT_TOKEN_BUDGET,
        "recursive_scanning": DEFAULT_RECURSIVE,
        "entries": [],
        "extensions": {},
    }


def lorebook_to_st(book: Dict[str, Any]) -> Dict[str, Any]:
    """내부 로어북을 SillyTavern 호환 형식으로 변환합니다.

    카드 character_book 스키마(keys/enabled/insertion_order)를 따르며,
    entries를 id 키 객체로 직렬화해 world info 임포트와도 호환됩니다.
    """
    entries: Dict[str, Any] = {}
    for entry in book.get("entries", []):
        entries[str(entry.get("id", len(entries)))] = {
            "id": entry.get("id", len(entries)),
            "keys": list(entry.get("keys", [])),
            "secondary_keys": list(entry.get("secondary_keys", [])),
            "content": entry.get("content", ""),
            "comment": entry.get("comment", ""),
            "name": entry.get("comment", ""),
            "constant": bool(entry.get("constant", False)),
            "selective": bool(entry.get("selective", False)),
            "enabled": bool(entry.get("enabled", True)),
            "insertion_order": int(entry.get("insertion_order", DEFAULT_INSERTION_ORDER)),
            "position": entry.get("position", "before_char"),
            "case_sensitive": bool(entry.get("case_sensitive", False)),
            "priority": int(entry.get("priority", DEFAULT_PRIORITY)),
            "extensions": entry.get("extensions", {}),
        }
    return {
        "name": book.get("name", ""),
        "scan_depth": int(book.get("scan_depth", DEFAULT_SCAN_DEPTH)),
        "token_budget": int(book.get("token_budget", DEFAULT_TOKEN_BUDGET)),
        "recursive_scanning": bool(book.get("recursive_scanning", DEFAULT_RECURSIVE)),
        "entries": entries,
        "extensions": book.get("extensions", {}),
    }


def export_lorebook_json(book: Dict[str, Any], out_path: str | Path) -> Path:
    """로어북을 SillyTavern 호환 JSON 파일로 저장합니다."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lorebook_to_st(book), f, ensure_ascii=False, indent=2)
    logger.info(f"로어북 JSON 익스포트 완료: {out_path}")
    return out_path


def load_lorebook_json(path: str | Path, name: str = "") -> Dict[str, Any]:
    """SillyTavern world info JSON 파일을 로드해 정규화합니다."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise LorebookError(f"로어북 JSON 파일을 읽을 수 없습니다: {e}")
    fallback_name = name or Path(path).stem
    return normalize_lorebook(raw, name=fallback_name)


# ---------------------------------------------------------------------------
# 활성화 엔진
# ---------------------------------------------------------------------------

def _match_keys(text_pool: str, keys: List[str], case_sensitive: bool) -> bool:
    """키워드 중 하나라도 포함되면 True (부분 문자열 매칭, ST 기본 동작)."""
    if not keys:
        return False
    hay = text_pool if case_sensitive else text_pool.lower()
    for key in keys:
        needle = key if case_sensitive else key.lower()
        if needle and needle in hay:
            return True
    return False


def _entry_matches(entry: Dict[str, Any], pool: str) -> bool:
    """단일 엔트리의 활성화 조건을 평가합니다."""
    if entry["constant"]:
        return True
    if not _match_keys(pool, entry["keys"], entry["case_sensitive"]):
        return False
    # selective: 보조 키워드 중 하나라도 함께 있어야 함 (AND ANY)
    if entry["selective"] and entry["secondary_keys"]:
        if not _match_keys(pool, entry["secondary_keys"], entry["case_sensitive"]):
            return False
    return True


def estimate_tokens(text: str) -> int:
    """대략적인 토큰 수를 추정합니다 (글자 수 기반 근사)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / TOKEN_CHARS_PER_TOKEN))


def _apply_budget(entries: List[Dict[str, Any]], token_budget: int) -> List[Dict[str, Any]]:
    """토큰 예산을 초과하면 priority가 낮은 엔트리부터 탈락시킵니다.

    동률일 때는 insertion_order가 큰(나중에 삽입되는) 엔트리부터 제거합니다.
    """
    total = sum(estimate_tokens(e["content"]) for e in entries)
    if total <= token_budget:
        return entries

    remaining = list(entries)
    while remaining and total > token_budget:
        victim = min(remaining, key=lambda e: (e["priority"], -e["insertion_order"]))
        remaining.remove(victim)
        total -= estimate_tokens(victim["content"])
        logger.debug(f"로어북 예산 초과로 엔트리 탈락: {victim.get('comment') or victim['id']}")
    return remaining


def activate_entries(
    entries: Any,
    scan_text: str,
    *,
    token_budget: int = 0,
    recursive: bool = True,
    max_recursion: int = MAX_RECURSION_PASSES,
) -> List[Dict[str, Any]]:
    """스캔 텍스트에서 활성화될 로어북 엔트리 목록을 반환합니다.

    Args:
        entries: 엔트리 리스트 또는 {id: entry} 객체 (원시 형식 허용)
        scan_text: 키워드 스캔 대상 텍스트 (최근 대화)
        token_budget: 0 이하면 무제한. 초과 시 priority 낮은 순으로 탈락
        recursive: True면 활성화된 엔트리 내용을 스캔 풀에 추가해 재스캔
        max_recursion: 재귀 활성화 최대 추가 패스 수

    Returns:
        insertion_order 오름차순으로 정렬된 정규화 엔트리 리스트
    """
    normalized: List[Dict[str, Any]] = []
    for idx, raw_entry in enumerate(_iter_entries(entries)):
        entry = normalize_entry(raw_entry, idx)
        if entry:
            normalized.append(entry)

    activated: Dict[int, Dict[str, Any]] = {}
    pool = scan_text or ""

    for _ in range(max_recursion + 1):
        newly: List[Dict[str, Any]] = []
        for entry in normalized:
            if entry["id"] in activated or not entry["enabled"]:
                continue
            if _entry_matches(entry, pool):
                activated[entry["id"]] = entry
                newly.append(entry)
        if not newly or not recursive:
            break
        # 재귀: 방금 활성화된 엔트리의 내용도 스캔 대상에 추가
        pool = pool + "\n" + "\n".join(e["content"] for e in newly)

    result = sorted(activated.values(), key=lambda e: (e["insertion_order"], e["id"]))
    if token_budget and token_budget > 0:
        result = _apply_budget(result, token_budget)
    return result


def build_lorebook_context(
    book: Optional[Dict[str, Any]],
    message_texts: List[str],
    char_name: str = "",
    user_name: str = "",
) -> str:
    """로어북을 최근 대화에 대해 활성화해 시스템 메시지에 붙일 컨텍스트를 빌드합니다.

    Args:
        book: 정규화된 로어북 dict (None/빈 북이면 빈 문자열)
        message_texts: 최근 메시지 텍스트 리스트 (오래된 순). scan_depth만큼 잘라 사용
        char_name: {{char}} 매크로 치환용 캐릭터 이름
        user_name: {{user}} 매크로 치환용 유저 이름

    Returns:
        활성화된 엔트리가 있으면 포맷된 컨텍스트 문자열, 없으면 빈 문자열
    """
    if not book:
        return ""
    entries = book.get("entries") or []
    if not entries:
        return ""

    scan_depth = max(1, _as_int(book.get("scan_depth"), DEFAULT_SCAN_DEPTH))
    recent = [t for t in message_texts[-scan_depth:] if t]
    scan_text = "\n".join(recent)
    if not scan_text.strip() and not any(_as_bool(e.get("constant"), False) for e in _iter_entries(entries)):
        return ""

    activated = activate_entries(
        entries,
        scan_text,
        token_budget=_as_int(book.get("token_budget"), DEFAULT_TOKEN_BUDGET),
        recursive=_as_bool(book.get("recursive_scanning"), DEFAULT_RECURSIVE),
    )
    if not activated:
        return ""

    lines = ["", "### 월드 인포 (World Info) ###"]
    for entry in activated:
        content = apply_macros(str(entry["content"]), char_name=char_name, user_name=user_name).strip()
        title = str(entry.get("comment") or "").strip()
        lines.append(f"[{title}]\n{content}" if title else content)

    logger.info(f"로어북 활성화: {len(activated)}개 엔트리 주입 ({[e.get('comment') or e['id'] for e in activated]})")
    return "\n".join(lines) + "\n"