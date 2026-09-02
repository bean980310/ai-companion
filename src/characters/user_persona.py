# user_persona.py
"""유저 페르소나 CRUD 및 챗 플로우 주입 컨텍스트 빌더.

유저 페르소나는 chat_history.db 의 user_personas 테이블에 저장되며,
활성화된 페르소나는 매 응답 생성 시 시스템 메시지에 주입됩니다.
"""

import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ai_companion_core import logger

from src.common.database import get_db_connection

NO_PERSONA_VALUE = "__none__"


@dataclass
class UserPersona:
    id: int
    name: str
    description: str
    avatar_path: Optional[str] = None
    is_active: bool = False


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_personas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            avatar_path TEXT,
            is_active INTEGER NOT NULL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _row_to_persona(row: Tuple) -> UserPersona:
    return UserPersona(id=row[0], name=row[1], description=row[2] or "", avatar_path=row[3], is_active=bool(row[4]))


def list_user_personas() -> List[UserPersona]:
    """모든 유저 페르소나를 목록으로 반환합니다."""
    try:
        with get_db_connection() as conn:
            _ensure_table(conn)
            rows = conn.execute("SELECT id, name, description, avatar_path, is_active FROM user_personas ORDER BY name ASC").fetchall()
            return [_row_to_persona(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"유저 페르소나 목록 조회 실패: {e}")
        return []


def get_persona_choices() -> List[Tuple[str, str]]:
    """드롭다운용 (label, value) 선택지 목록. '없음'이 항상 첫 항목."""
    choices = [(NO_PERSONA_VALUE_LABEL, NO_PERSONA_VALUE)]
    for persona in list_user_personas():
        label = f"{persona.name} ✦" if persona.is_active else persona.name
        choices.append((label, str(persona.id)))
    return choices


NO_PERSONA_VALUE_LABEL = "없음 (No Persona)"


def get_active_persona() -> Optional[UserPersona]:
    """현재 활성화된 유저 페르소나를 반환합니다."""
    try:
        with get_db_connection() as conn:
            _ensure_table(conn)
            row = conn.execute("SELECT id, name, description, avatar_path, is_active FROM user_personas WHERE is_active = 1 LIMIT 1").fetchone()
            return _row_to_persona(row) if row else None
    except sqlite3.Error as e:
        logger.error(f"활성 유저 페르소나 조회 실패: {e}")
        return None


def get_persona_by_name(name: str) -> Optional[UserPersona]:
    """이름으로 유저 페르소나를 조회합니다."""
    try:
        with get_db_connection() as conn:
            _ensure_table(conn)
            row = conn.execute("SELECT id, name, description, avatar_path, is_active FROM user_personas WHERE name = ?", (name,)).fetchone()
            return _row_to_persona(row) if row else None
    except sqlite3.Error as e:
        logger.error(f"유저 페르소나 조회 실패 (name={name}): {e}")
        return None


def get_persona_by_id(persona_id: int | str) -> Optional[UserPersona]:
    """ID로 유저 페르소나를 조회합니다."""
    try:
        with get_db_connection() as conn:
            _ensure_table(conn)
            row = conn.execute("SELECT id, name, description, avatar_path, is_active FROM user_personas WHERE id = ?", (int(persona_id),)).fetchone()
            return _row_to_persona(row) if row else None
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"유저 페르소나 조회 실패 (id={persona_id}): {e}")
        return None


def add_user_persona(name: str, description: str, avatar_path: Optional[str] = None, activate: bool = False) -> Tuple[bool, str]:
    """새 유저 페르소나를 추가합니다."""
    name = (name or "").strip()
    if not name:
        return False, "❌ 페르소나 이름을 입력해주세요."
    try:
        with get_db_connection() as conn:
            _ensure_table(conn)
            exists = conn.execute("SELECT COUNT(*) FROM user_personas WHERE name = ?", (name,)).fetchone()[0]
            if exists:
                return False, f"⚠️ '{name}' 페르소나가 이미 존재합니다."
            if activate:
                conn.execute("UPDATE user_personas SET is_active = 0")
            conn.execute(
                "INSERT INTO user_personas (name, description, avatar_path, is_active) VALUES (?, ?, ?, ?)",
                (name, (description or "").strip(), avatar_path, 1 if activate else 0),
            )
            conn.commit()
        logger.info(f"유저 페르소나 추가됨: {name}")
        return True, f"✅ '{name}' 페르소나가 추가되었습니다."
    except sqlite3.Error as e:
        logger.error(f"유저 페르소나 추가 실패: {e}")
        return False, f"❌ 추가 실패: {e}"


def update_user_persona(persona_id: int, name: str, description: str, avatar_path: Optional[str] = None) -> Tuple[bool, str]:
    """기존 유저 페르소나를 수정합니다."""
    name = (name or "").strip()
    if not name:
        return False, "❌ 페르소나 이름을 입력해주세요."
    try:
        with get_db_connection() as conn:
            _ensure_table(conn)
            dup = conn.execute("SELECT COUNT(*) FROM user_personas WHERE name = ? AND id != ?", (name, persona_id)).fetchone()[0]
            if dup:
                return False, f"⚠️ '{name}' 이름의 페르소나가 이미 존재합니다."
            if avatar_path is None:
                conn.execute(
                    "UPDATE user_personas SET name = ?, description = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (name, (description or "").strip(), persona_id),
                )
            else:
                conn.execute(
                    "UPDATE user_personas SET name = ?, description = ?, avatar_path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (name, (description or "").strip(), avatar_path, persona_id),
                )
            conn.commit()
        return True, f"✅ '{name}' 페르소나가 수정되었습니다."
    except sqlite3.Error as e:
        logger.error(f"유저 페르소나 수정 실패: {e}")
        return False, f"❌ 수정 실패: {e}"


def delete_user_persona(persona_id: int) -> Tuple[bool, str]:
    """유저 페르소나를 삭제합니다."""
    try:
        with get_db_connection() as conn:
            _ensure_table(conn)
            row = conn.execute("SELECT name FROM user_personas WHERE id = ?", (persona_id,)).fetchone()
            if not row:
                return False, "❌ 삭제할 페르소나를 찾을 수 없습니다."
            conn.execute("DELETE FROM user_personas WHERE id = ?", (persona_id,))
            conn.commit()
        logger.info(f"유저 페르소나 삭제됨: {row[0]}")
        return True, f"✅ '{row[0]}' 페르소나가 삭제되었습니다."
    except sqlite3.Error as e:
        logger.error(f"유저 페르소나 삭제 실패: {e}")
        return False, f"❌ 삭제 실패: {e}"


def set_active_persona(persona_id: Optional[int]) -> Tuple[bool, str]:
    """활성 유저 페르소나를 변경합니다. None 이면 비활성화합니다."""
    try:
        with get_db_connection() as conn:
            _ensure_table(conn)
            conn.execute("UPDATE user_personas SET is_active = 0")
            if persona_id is not None:
                row = conn.execute("SELECT name FROM user_personas WHERE id = ?", (persona_id,)).fetchone()
                if not row:
                    return False, "❌ 해당 페르소나를 찾을 수 없습니다."
                conn.execute("UPDATE user_personas SET is_active = 1 WHERE id = ?", (persona_id,))
                name = row[0]
            else:
                name = None
            conn.commit()
        if name:
            logger.info(f"활성 유저 페르소나 변경: {name}")
            return True, f"✅ '{name}' 페르소나가 활성화되었습니다."
        return True, "✅ 유저 페르소나가 비활성화되었습니다."
    except sqlite3.Error as e:
        logger.error(f"활성 유저 페르소나 변경 실패: {e}")
        return False, f"❌ 변경 실패: {e}"


def set_active_persona_by_name(name: Optional[str]) -> Tuple[bool, str]:
    """이름으로 활성 페르소나를 변경합니다 (드롭다운용)."""
    if not name or name == NO_PERSONA_VALUE:
        return set_active_persona(None)
    persona = get_persona_by_name(name)
    if not persona:
        return False, f"❌ '{name}' 페르소나를 찾을 수 없습니다."
    return set_active_persona(persona.id)


def build_user_persona_context(persona: Optional[UserPersona] = None) -> str:
    """활성 유저 페르소나를 시스템 메시지에 주입할 컨텍스트 문자열로 빌드합니다.

    Returns:
        페르소나가 없거나 비어 있으면 빈 문자열
    """
    if persona is None:
        persona = get_active_persona()
    if not persona:
        return ""

    lines = ["", "### 유저 페르소나 (User Persona) ###"]
    lines.append(f"대화 상대 유저의 이름은 \"{persona.name}\" 입니다. 유저를 이 이름으로 부르십시오.")
    description = (persona.description or "").strip()
    if description:
        lines.append("유저에 대한 정보:")
        lines.append(description)
    return "\n".join(lines) + "\n"


def get_active_user_name() -> str:
    """활성 페르소나의 이름을 반환합니다 ({{user}} 매크로 치환용)."""
    persona = get_active_persona()
    return persona.name if persona else ""