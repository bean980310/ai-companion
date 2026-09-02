# character_card.py
"""SillyTavern 캐릭터 카드(V1/V2/V3) 파싱, 프롬프트 빌딩, 익스포트 유틸리티.

지원 규격:
- V3 (chara_card_v3): PNG tEXt 청크 키 ``ccv3`` (base64 JSON)
- V2 (chara_card_v2): PNG tEXt 청크 키 ``chara`` (base64 JSON, ``data`` 필드)
- V1 (legacy): PNG tEXt 청크 키 ``chara`` (base64 JSON, 플랫 필드)
- JSON 파일: {"spec":..., "data": {...}} / V1 플랫 / 구형 ST export (char_name 등)
"""

import base64
import binascii
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from ai_companion_core import logger

# 내부 표준 필드 (V2 data 기준)
CARD_FIELDS = [
    "name",
    "description",
    "personality",
    "scenario",
    "first_mes",
    "mes_example",
    "creator_notes",
    "system_prompt",
    "post_history_instructions",
    "alternate_greetings",
    "tags",
    "creator",
    "character_version",
    "extensions",
]

V1_FIELD_ALIASES = {
    "char_name": "name",
    "name": "name",
    "char_persona": "personality",
    "personality": "personality",
    "description": "description",
    "world_scenario": "scenario",
    "scenario": "scenario",
    "char_greeting": "first_mes",
    "first_mes": "first_mes",
    "example_dialogue": "mes_example",
    "mes_example": "mes_example",
}


class CharacterCardError(Exception):
    """캐릭터 카드 파싱/처리 관련 예외"""

    pass


def normalize_card_data(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """카드 JSON을 내부 표준 형식(V2 data 스타일 dict)으로 정규화합니다.

    Returns:
        정규화된 card_data dict, 유효하지 않으면 None
    """
    if not isinstance(raw, dict):
        return None

    data: Dict[str, Any] = {}
    if isinstance(raw.get("data"), dict):
        source = raw["data"]
    else:
        # V1 플랫 또는 구형 ST export 형식
        source = {}
        for key, value in raw.items():
            mapped = V1_FIELD_ALIASES.get(key)
            if mapped and mapped not in source:
                source[mapped] = value

    if not source.get("name"):
        return None

    for field in CARD_FIELDS:
        value = source.get(field)
        if value is None:
            value = "" if field not in ("alternate_greetings", "tags", "extensions") else ([] if field != "extensions" else {})
        data[field] = value

    # extensions 가 dict 이 아닌 경우 방어
    if not isinstance(data["extensions"], dict):
        data["extensions"] = {}
    if isinstance(data["tags"], str):
        data["tags"] = [t.strip() for t in data["tags"].split(",") if t.strip()]
    if isinstance(data["alternate_greetings"], str):
        data["alternate_greetings"] = [data["alternate_greetings"]] if data["alternate_greetings"] else []

    # spec 정보 보존
    data["spec"] = raw.get("spec", "chara_card_v1" if not isinstance(raw.get("data"), dict) else "chara_card_v2")
    data["spec_version"] = raw.get("spec_version", "")
    return data


def _decode_chara_payload(raw: str) -> Optional[Dict[str, Any]]:
    """tEXt 청크의 base64 JSON payload 를 디코딩합니다."""
    try:
        decoded = base64.b64decode(raw.strip())
        return json.loads(decoded.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError) as e:
        logger.warning(f"카드 payload 디코딩 실패: {e}")
        return None


def read_png_card(path: str | Path) -> Optional[Dict[str, Any]]:
    """PNG 파일의 tEXt/iTXt/zTXt 청크에서 캐릭터 카드를 읽어 정규화합니다."""
    try:
        with Image.open(path) as im:
            texts = dict(getattr(im, "text", {}) or {})
            if not texts:
                texts = {k: v for k, v in (im.info or {}).items() if isinstance(v, str)}
    except Exception as e:
        raise CharacterCardError(f"PNG 파일을 열 수 없습니다: {e}")

    # V3 우선, 그 다음 V2/V1
    for key in ("ccv3", "chara"):
        payload = texts.get(key)
        if not payload:
            continue
        raw = _decode_chara_payload(payload)
        if raw is None:
            continue
        card = normalize_card_data(raw)
        if card:
            return card
    return None


def read_json_card(path: str | Path) -> Optional[Dict[str, Any]]:
    """JSON 파일에서 캐릭터 카드를 읽어 정규화합니다."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise CharacterCardError(f"JSON 파일을 읽을 수 없습니다: {e}")

    if isinstance(raw, dict) and isinstance(raw.get("data"), dict):
        return normalize_card_data(raw)
    return normalize_card_data(raw)


def parse_character_card(path: str | Path) -> Optional[Dict[str, Any]]:
    """PNG 또는 JSON 캐릭터 카드 파일을 파싱합니다.

    Returns:
        정규화된 card_data dict, 카드 데이터가 없으면 None
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".png":
        return read_png_card(path)
    elif suffix == ".json":
        return read_json_card(path)
    raise CharacterCardError(f"지원하지 않는 카드 형식입니다: {suffix} (.png, .json 만 지원)")


def apply_macros(text: str, char_name: str = "", user_name: str = "") -> str:
    """SillyTavern 매크로({{char}}, {{user}}, <BOT>, <USER>)를 치환합니다."""
    if not isinstance(text, str):
        return text
    if char_name:
        text = text.replace("{{char}}", char_name).replace("<BOT>", char_name)
    if user_name:
        text = text.replace("{{user}}", user_name).replace("<USER>", user_name)
    return text


def build_system_prompt(card_data: Dict[str, Any], user_name: str = "", include_greeting: bool = False) -> str:
    """정규화된 카드 데이터를 시스템 프롬프트 문자열로 빌드합니다.

    Args:
        card_data: 정규화된 카드 데이터
        user_name: {{user}} 매크로 치환에 사용할 유저 이름 (빈 값이면 치환하지 않음)
        include_greeting: 첫 인사(first_mes)를 프롬프트에 포함할지 여부

    Returns:
        빌드된 시스템 프롬프트 문자열
    """
    char_name = card_data.get("name", "")

    def section(title: str, content: str) -> str:
        content = (content or "").strip()
        if not content:
            return ""
        return f"### {title} ###\n{content}\n"

    parts: List[str] = []

    system_prompt = (card_data.get("system_prompt") or "").strip()
    if system_prompt:
        parts.append(system_prompt + "\n")

    description = (card_data.get("description") or "").strip()
    if description:
        parts.append(f"{char_name}의 설정:\n{description}\n")

    for title, field in [
        ("성격", "personality"),
        ("시나리오", "scenario"),
    ]:
        content = (card_data.get(field) or "").strip()
        if content:
            parts.append(f"### {title} ###\n{content}\n")

    mes_example = (card_data.get("mes_example") or "").strip()
    if mes_example:
        parts.append(f"### 대화 예시 ###\n{mes_example}\n")

    if include_greeting:
        first_mes = (card_data.get("first_mes") or "").strip()
        if first_mes:
            parts.append(f"### 첫 인사 (첫 메시지는 이 인사로 시작해도 좋습니다) ###\n{first_mes}\n")

    prompt = "\n".join(parts).strip()
    return apply_macros(prompt, char_name=char_name, user_name=user_name)


def card_to_v2(card_data: Dict[str, Any]) -> Dict[str, Any]:
    """내부 카드 데이터를 SillyTavern V2 카드 JSON으로 변환합니다."""
    data = {field: card_data.get(field, "" if field not in ("alternate_greetings", "tags", "extensions") else ([] if field != "extensions" else {})) for field in CARD_FIELDS}
    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": data,
        "name": data["name"],
        "description": data["description"],
        "personality": data["personality"],
        "scenario": data["scenario"],
        "first_mes": data["first_mes"],
        "mes_example": data["mes_example"],
    }


def export_card_json(card_data: Dict[str, Any], out_path: str | Path) -> Path:
    """카드 데이터를 SillyTavern V2 JSON 파일로 저장합니다."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(card_to_v2(card_data), f, ensure_ascii=False, indent=2)
    logger.info(f"카드 JSON 익스포트 완료: {out_path}")
    return out_path


def export_card_png(card_data: Dict[str, Any], avatar_path: str | Path | None, out_path: str | Path) -> Path:
    """카드 데이터를 아바타 이미지에 embed 하여 SillyTavern V2 PNG 카드로 저장합니다."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(card_to_v2(card_data), ensure_ascii=False).encode("utf-8")
    encoded = base64.b64encode(payload).decode("utf-8")

    if avatar_path and Path(avatar_path).is_file():
        with Image.open(avatar_path) as im:
            im = im.convert("RGBA")
    else:
        im = Image.new("RGBA", (400, 600), (40, 44, 60, 255))

    from PIL.PngImagePlugin import PngInfo

    pnginfo = PngInfo()
    pnginfo.add_text("chara", encoded)
    im.save(out_path, format="PNG", pnginfo=pnginfo)
    logger.info(f"카드 PNG 익스포트 완료: {out_path}")
    return out_path