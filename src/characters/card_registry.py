# card_registry.py
"""SillyTavern 형식 캐릭터 카드의 임포트/등록/삭제 및 레거시 프리셋 마이그레이션.

- character_cards 테이블: ST 형식 카드 데이터의 영구 저장소 (소스 오브 트루스)
- 임포트된 카드는 런타임 characters 레지스트리 + system_presets 에 등록되어
  기존 캐릭터 선택 플로우 그대로 사용할 수 있다.
- 레거시 프리셋(presets/*.json)은 마크다운 섹션을 파싱해 ST V2 필드로 변환한다.
"""

import json
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_companion_core import logger

from src.common.character_info import characters
from src.common.database import get_db_connection, add_system_preset, delete_system_preset
from src.common.translations import translation_manager
from src.common.default_language import default_language
from src.characters.character_card import parse_character_card, build_system_prompt, CharacterCardError
from src.characters.lorebook import normalize_lorebook, build_lorebook_context
from src.characters.preset_images import PRESET_IMAGES

IMPORTED_AVATAR_DIR = Path("assets/imported_cards")

# 임포트된 카드 캐릭터의 말투 변환 비활성화 플래그
CARD_DEFAULT_TONE = "none"

# 레거시 프리셋 → character_settings 키 매핑 (유틸리티 프리셋은 제외)
LEGACY_CHARACTER_KEYS = {
    "미나미 아스카 (南飛鳥, みなみあすか, Minami Asuka)": "minami_asuka",
    "마코토노 아오이 (真琴乃葵, まことのあおい, Makotono Aoi)": "makotono_aoi",
    "아이노 코이토 (愛野小糸, あいのこいと, Aino Koito)": "aino_koito",
    "아리아 프린세스 페이트 (アリア·プリンセス·フェイト, Aria Princess Fate)": "aria_princess_fate",
    "아리아 프린스 페이트 (アリア·プリンス·フェイト, Aria Prince Fate)": "aria_prince_fate",
    "왕 메이린 (王美玲, ワン·メイリン, Wang Mei-Ling)": "wang_mei_ling",
    "미스티 레인 (ミスティ·レーン, Misty Lane)": "misty_lane",
    "릴리 엠프레스 (リリー·エンプレス, Lily Empress)": "lily_empress",
    "최유나 (崔有娜, チェ·ユナ, Choi Yuna)": "choi_yuna",
    "최유리 (崔有莉, チェ·ユリ, Choi Yuri)": "choi_yuri",
}

# 레거시 마크다운 섹션 제목 → ST 필드 매핑
SECTION_FIELD_MAP = {
    "프로필": "description",
    "profile": "description",
    "プロフィール": "description",
    "성격": "personality",
    "personality": "personality",
    "性格": "personality",
}


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS character_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            card_json TEXT NOT NULL,
            avatar_path TEXT,
            source TEXT NOT NULL DEFAULT 'import',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def save_card(name: str, card_bundle: Dict[str, Any], avatar_path: Optional[str], source: str = "import") -> None:
    """카드 번들을 DB에 저장합니다.

    card_bundle 형식:
        {"spec": ..., "data": {...}, "per_language": {lang: card_data, ...} (선택)}
    """
    with get_db_connection() as conn:
        _ensure_table(conn)
        conn.execute(
            """
            INSERT INTO character_cards (name, card_json, avatar_path, source)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                card_json = excluded.card_json,
                avatar_path = excluded.avatar_path,
                source = excluded.source,
                updated_at = CURRENT_TIMESTAMP
            """,
            (name, json.dumps(card_bundle, ensure_ascii=False), avatar_path, source),
        )
        conn.commit()


def list_cards() -> List[Dict[str, Any]]:
    """저장된 모든 카드를 반환합니다."""
    try:
        with get_db_connection() as conn:
            _ensure_table(conn)
            rows = conn.execute("SELECT id, name, card_json, avatar_path, source FROM character_cards ORDER BY name ASC").fetchall()
            cards = []
            for row in rows:
                try:
                    bundle = json.loads(row[2])
                except json.JSONDecodeError:
                    logger.warning(f"손상된 카드 데이터 건너뜀: {row[1]}")
                    continue
                cards.append({"id": row[0], "name": row[1], "bundle": bundle, "avatar_path": row[3], "source": row[4]})
            return cards
    except sqlite3.Error as e:
        logger.error(f"카드 목록 조회 실패: {e}")
        return []


def get_card_bundle(name: str) -> Optional[Dict[str, Any]]:
    """이름으로 카드 번들을 조회합니다."""
    try:
        with get_db_connection() as conn:
            _ensure_table(conn)
            row = conn.execute("SELECT card_json FROM character_cards WHERE name = ?", (name,)).fetchone()
            return json.loads(row[0]) if row else None
    except (sqlite3.Error, json.JSONDecodeError) as e:
        logger.error(f"카드 조회 실패 (name={name}): {e}")
        return None


def delete_card(name: str) -> Tuple[bool, str]:
    """임포트된 카드를 DB, system_presets, 런타임 레지스트리에서 제거합니다.

    레거시 마이그레이션 카드(내장 캐릭터)는 삭제할 수 없습니다.
    """
    with get_db_connection() as conn:
        _ensure_table(conn)
        row = conn.execute("SELECT source FROM character_cards WHERE name = ?", (name,)).fetchone()
        if not row:
            return False, f"❌ '{name}' 카드를 찾을 수 없습니다."
        if row[0] != "import":
            return False, f"⚠️ '{name}'은(는) 내장 캐릭터입니다. 임포트된 카드만 삭제할 수 있습니다."
        conn.execute("DELETE FROM character_cards WHERE name = ?", (name,))
        conn.commit()

    # system_presets 에서 제거 (기본 프리셋은 delete_system_preset 이 자체적으로 거부)
    for lang in translation_manager.get_available_languages():
        delete_system_preset(name, lang)

    characters.pop(name, None)
    PRESET_IMAGES.pop(name, None)
    logger.info(f"카드 삭제됨: {name}")
    return True, f"✅ '{name}' 카드가 삭제되었습니다."


def register_card_runtime(name: str, bundle: Dict[str, Any], avatar_path: Optional[str]) -> None:
    """카드 번들을 런타임 characters 레지스트리와 system_presets 에 등록합니다."""
    per_language: Dict[str, Dict[str, Any]] = bundle.get("per_language") or {}
    base_data = bundle.get("data") or {}

    # 언어별 시스템 프리셋 등록 (카드가 단일 언어면 모든 언어에 동일 프롬프트)
    for lang in translation_manager.get_available_languages():
        card_data = per_language.get(lang, base_data)
        if not card_data:
            continue
        prompt = build_system_prompt(card_data)
        add_system_preset(name, lang, prompt, overwrite=True)

    languages = sorted(per_language.keys()) or [default_language]
    characters[name] = {
        "default_tone": CARD_DEFAULT_TONE,
        "languages": languages,
        "default_language": default_language if default_language in languages else languages[0],
        "preset_name": name,
        "profile_image": avatar_path or "",
        "is_card": True,
    }
    if avatar_path:
        PRESET_IMAGES[name] = avatar_path
    logger.info(f"카드 런타임 등록 완료: {name}")


def import_card_file(path: str) -> Tuple[bool, str, Optional[str]]:
    """SillyTavern 카드 파일(PNG/JSON)을 임포트하여 캐릭터로 등록합니다.

    Returns:
        (성공 여부, 메시지, 등록된 캐릭터 이름)
    """
    try:
        card_data = parse_character_card(path)
    except CharacterCardError as e:
        return False, f"❌ {e}", None

    if not card_data:
        return False, "❌ 파일에서 캐릭터 카드 데이터를 찾을 수 없습니다. (tEXt 'chara'/'ccv3' 청크 또는 유효한 JSON 필요)", None

    name = card_data.get("name", "").strip()
    if not name:
        return False, "❌ 카드에 이름(name) 필드가 없습니다.", None

    # 아바타: PNG 카드면 원본 이미지를 그대로 복사
    avatar_path = None
    src = Path(path)
    if src.suffix.lower() == ".png":
        IMPORTED_AVATAR_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", name)
        dest = IMPORTED_AVATAR_DIR / f"{safe_name}.png"
        counter = 1
        while dest.exists():
            dest = IMPORTED_AVATAR_DIR / f"{safe_name}_{counter}.png"
            counter += 1
        try:
            shutil.copyfile(src, dest)
            avatar_path = str(dest)
        except OSError as e:
            logger.warning(f"아바타 복사 실패: {e}")

    bundle = {
        "spec": card_data.get("spec", "chara_card_v2"),
        "spec_version": card_data.get("spec_version", ""),
        "data": card_data,
    }

    # 카드에 character_book(로어북)이 포함되어 있으면 내부 표준 스키마로 정규화
    extensions = card_data.get("extensions")
    if isinstance(extensions, dict) and isinstance(extensions.get("character_book"), dict) and extensions["character_book"]:
        try:
            extensions["character_book"] = normalize_lorebook(extensions["character_book"], name=name)
            card_data["extensions"] = extensions
            logger.info(f"카드 로어북 정규화 완료: {name} ({len(extensions['character_book'].get('entries', []))}개 엔트리)")
        except Exception as e:
            logger.warning(f"카드 로어북 정규화 실패 (원본 유지): {e}")

    save_card(name, bundle, avatar_path, source="import")
    register_card_runtime(name, bundle, avatar_path)
    logger.info(f"SillyTavern 카드 임포트 완료: {name} (spec={bundle['spec']})")
    return True, f"✅ '{name}' 캐릭터가 임포트되었습니다. (spec: {bundle['spec']})", name


# ---------------------------------------------------------------------------
# 캐릭터 로어북 (카드 extensions.character_book)
# ---------------------------------------------------------------------------

def get_character_lorebook(name: str) -> Optional[Dict[str, Any]]:
    """캐릭터 카드에 포함된 로어북을 정규화된 형태로 조회합니다. 없으면 None."""
    bundle = get_card_bundle(name)
    if not bundle:
        return None
    extensions = ((bundle.get("data") or {}).get("extensions")) or {}
    raw_book = extensions.get("character_book")
    if not isinstance(raw_book, dict) or not raw_book:
        return None
    try:
        return normalize_lorebook(raw_book, name=name)
    except Exception as e:
        logger.warning(f"캐릭터 로어북 정규화 실패 (name={name}): {e}")
        return None


def update_character_lorebook(name: str, book: Dict[str, Any]) -> Tuple[bool, str]:
    """캐릭터 카드의 extensions.character_book 을 갱신합니다."""
    bundle = get_card_bundle(name)
    if not bundle:
        return False, f"❌ '{name}' 카드를 찾을 수 없습니다."

    try:
        normalized = normalize_lorebook(book, name=name)
    except Exception as e:
        return False, f"❌ 로어북 정규화 실패: {e}"

    data = bundle.get("data") or {}
    extensions = data.get("extensions") if isinstance(data.get("extensions"), dict) else {}
    extensions["character_book"] = normalized
    data["extensions"] = extensions
    bundle["data"] = data

    with get_db_connection() as conn:
        _ensure_table(conn)
        row = conn.execute("SELECT avatar_path, source FROM character_cards WHERE name = ?", (name,)).fetchone()
    if not row:
        return False, f"❌ '{name}' 카드를 찾을 수 없습니다."

    save_card(name, bundle, row[0], source=row[1])
    logger.info(f"캐릭터 로어북 갱신 완료: {name} ({len(normalized.get('entries', []))}개 엔트리)")
    return True, f"✅ '{name}' 캐릭터의 로어북이 저장되었습니다. ({len(normalized.get('entries', []))}개 엔트리)"


def get_character_lorebook_context(
    character_name: Optional[str],
    message_texts: List[str],
    user_name: str = "",
) -> str:
    """활성 캐릭터의 로어북을 대화 텍스트에 대해 활성화해 컨텍스트 문자열을 반환합니다.

    로어북이 없거나 활성화된 엔트리가 없으면 빈 문자열. 채팅 플로우에서 호출됩니다.
    """
    if not character_name:
        return ""
    book = get_character_lorebook(character_name)
    if not book:
        return ""
    try:
        return build_lorebook_context(book, message_texts, char_name=character_name, user_name=user_name)
    except Exception as e:
        logger.warning(f"로어북 컨텍스트 생성 실패 (character={character_name}): {e}")
        return ""


# ---------------------------------------------------------------------------
# 레거시 프리셋 → ST 카드 마이그레이션
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{2,4}\s*(.+?)\s*#{0,4}\s*$")


def parse_legacy_prompt(text: str) -> Dict[str, str]:
    """레거시 마크다운 프롬프트를 (intro, {section_title: content}) 로 파싱합니다."""
    intro_lines: List[str] = []
    sections: Dict[str, List[str]] = {}
    current_title: Optional[str] = None

    for line in text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            current_title = match.group(1).strip()
            sections.setdefault(current_title, [])
        elif current_title is None:
            intro_lines.append(line)
        else:
            sections[current_title].append(line)

    intro = "\n".join(intro_lines).strip()
    return {"intro": intro, "sections": {title: "\n".join(lines).strip() for title, lines in sections.items()}}


def legacy_prompt_to_card(name: str, prompt_text: str) -> Dict[str, Any]:
    """레거시 프롬프트 텍스트를 ST V2 card_data 로 변환합니다."""
    parsed = parse_legacy_prompt(prompt_text)

    description_parts: List[str] = []
    personality_parts: List[str] = []

    for title, content in parsed["sections"].items():
        field = SECTION_FIELD_MAP.get(title.lower())
        content_block = f"### {title} ###\n{content}" if content else ""
        if not content_block:
            continue
        if field == "personality":
            personality_parts.append(content_block)
        elif field == "description":
            description_parts.append(content_block)
        else:
            description_parts.append(content_block)

    return {
        "name": name,
        "description": "\n\n".join(description_parts).strip(),
        "personality": "\n\n".join(personality_parts).strip(),
        "scenario": "",
        "first_mes": "",
        "mes_example": "",
        "creator_notes": "Migrated from ai-companion legacy preset",
        "system_prompt": parsed["intro"],
        "post_history_instructions": "",
        "alternate_greetings": [],
        "tags": [],
        "creator": "",
        "character_version": "",
        "extensions": {"legacy_prompt": prompt_text, "migrated": True},
        "spec": "chara_card_v2",
        "spec_version": "2.0",
    }


def migrate_legacy_presets() -> int:
    """레거시 캐릭터 프리셋을 ST 카드 형식으로 변환하여 등록합니다.

    - presets/*.json (언어별 플랫 프롬프트)을 파싱해 ST V2 필드로 변환
    - character_cards 테이블에 per_language 카드로 저장
    - system_presets 는 변환된 카드로부터 재빌드된 ST 스타일 프롬프트로 갱신
    - 원본 프롬프트는 extensions.legacy_prompt 에 보존

    Returns:
        마이그레이션된 캐릭터 수
    """
    migrated = 0
    for display_name, character_key in LEGACY_CHARACTER_KEYS.items():
        preset = translation_manager.character_settings.get(character_key)
        if not preset:
            logger.warning(f"레거시 프리셋을 찾을 수 없어 건너뜀: {character_key}")
            continue

        per_language: Dict[str, Dict[str, Any]] = {}
        for lang, prompt_text in preset.items():
            if not prompt_text or not prompt_text.strip():
                continue
            per_language[lang] = legacy_prompt_to_card(display_name, prompt_text)

        if not per_language:
            continue

        default_card = per_language.get(default_language) or next(iter(per_language.values()))
        bundle = {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": default_card,
            "per_language": per_language,
        }

        avatar_path = PRESET_IMAGES.get(display_name)
        save_card(display_name, bundle, avatar_path, source="legacy_migration")

        # system_presets 를 ST 스타일로 재빌드
        for lang, card_data in per_language.items():
            add_system_preset(display_name, lang, build_system_prompt(card_data), overwrite=True)

        # 런타임 레지스트리에 카드 플래그 반영
        if display_name in characters:
            characters[display_name]["is_card"] = True

        migrated += 1
        logger.info(f"레거시 프리셋 ST 카드 마이그레이션 완료: {display_name}")

    return migrated


def register_all_on_startup() -> None:
    """앱 시작 시 카드 시스템을 초기화합니다.

    1. 레거시 프리셋을 ST 카드 형식으로 마이그레이션
    2. 저장된 임포트 카드를 런타임 레지스트리에 재등록 (재시작 후 유지)
    """
    try:
        migrated = migrate_legacy_presets()
        logger.info(f"레거시 ST 카드 마이그레이션: {migrated}개 캐릭터")
    except Exception as e:
        logger.error(f"레거시 ST 카드 마이그레이션 실패: {e}")

    try:
        restored = 0
        for card in list_cards():
            if card["source"] == "import":
                register_card_runtime(card["name"], card["bundle"], card["avatar_path"])
                restored += 1
        if restored:
            logger.info(f"임포트된 카드 {restored}개를 런타임에 재등록했습니다.")
    except Exception as e:
        logger.error(f"임포트 카드 재등록 실패: {e}")