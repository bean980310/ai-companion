from typing import Optional, List, Tuple, Dict, Any
import sqlite3
import logging
from contextlib import contextmanager
from dataclasses import dataclass
import gradio as gr
import json
from datetime import datetime
import csv
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PresetConfig:
    """프리셋 설정을 위한 데이터 클래스"""
    name: str
    character_key: str
    
@dataclass
class PresetResult:
    """프리셋 작업 결과를 나타내는 데이터 클래스"""
    success: bool
    message: str
    
@dataclass
class ChatMessage:
    """채팅 메시지를 표현하는 데이터 클래스"""
    role: str
    content: str
    timestamp: Optional[datetime] = None

@dataclass
class SessionResult:
    """세션 작업 결과를 나타내는 데이터 클래스"""
    success: bool
    message: str
    affected_rows: int = 0

class DatabaseInitError(Exception):
    """데이터베이스 초기화 관련 커스텀 예외"""
    pass

class PresetInsertionError(Exception):
    """프리셋 삽입 관련 커스텀 예외"""
    pass
class PresetManagementError(Exception):
    """프리셋 관리 관련 커스텀 예외"""
    pass

class SessionManagementError(Exception):
    """세션 관리 관련 커스텀 예외"""
    pass


DEFAULT_PRESETS = frozenset([
    'AI_ASSISTANT_PRESET',
    'MINAMI_ASUKA_PRESET',
    'MAKOTONO_AOI_PRESET',
    'AINO_KOITO_PRESET'
])

class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = sqlite3.connect("chat_history.db", timeout=10)
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise DatabaseError(f"Failed to connect to database: {e}")
    finally:
        if conn:
            conn.close()
def backfill_timestamps():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # 'timestamp' 열이 NULL인 경우 현재 시간으로 업데이트
            cursor.execute("""
                UPDATE chat_history
                SET timestamp = CURRENT_TIMESTAMP
                WHERE timestamp IS NULL
            """)
            affected_rows = cursor.rowcount
            conn.commit()
            logger.info(f"Backfilled timestamps for {affected_rows} chat_history records.")
    except Exception as e:
        logger.error(f"Error backfilling timestamps: {e}")
        
def initialize_database() -> None:
    """데이터베이스와 필요한 테이블들을 초기화합니다.
    
    이 함수는 앱 시작 시 항상 실행되어야 합니다.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 시스템 프리셋 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    language TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, language)
                )
            """)
            
            # 채팅 히스토리 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            
            # 세션 관리 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_history_session
                ON chat_history(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_system_presets_lang
                ON system_presets(language)
            """)
            
            cursor.execute("PRAGMA table_info(chat_history)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'timestamp' not in columns:
                cursor.execute("ALTER TABLE chat_history ADD COLUMN timestamp DATETIME DEFAULT (CURRENT_TIMESTAMP)")
                logger.info("'chat_history' 테이블에 'timestamp' 열 추가 완료.")
                backfill_timestamps()
            else:
                logger.info("'chat_history' 테이블에 'timestamp' 열이 이미 존재합니다.")
                
            conn.commit()
            logger.info("Database initialized successfully")
            
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise DatabaseInitError(f"Failed to initialize database: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {e}")
        raise DatabaseInitError(f"Unexpected error: {e}")

def ensure_demo_session() -> None:
    """데모 세션이 존재하는지 확인하고 없으면 생성합니다."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 데모 세션 존재 여부 확인
            cursor.execute("SELECT id FROM sessions WHERE id = 'demo_session'")
            if not cursor.fetchone():
                # 데모 세션 생성
                current_time = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO sessions (id, name, created_at, updated_at, last_activity)
                    VALUES (?, ?, ?, ?, ?)
                """, ('demo_session', 'Demo Session', current_time, current_time, current_time))
                
                # 기본 시스템 메시지 추가
                cursor.execute("""
                    INSERT INTO chat_history (session_id, role, content)
                    VALUES (?, 'system', '당신은 유용한 AI 비서입니다.')
                """, ('demo_session',))
                
                conn.commit()
                logger.info("Demo session created successfully")
            
    except sqlite3.Error as e:
        logger.error(f"Error ensuring demo session: {e}")
        raise DatabaseInitError(f"Failed to ensure demo session: {e}")

def initialize_app(translation_manager) -> None:
    """애플리케이션 시작 시 필요한 모든 초기화를 수행합니다."""
    try:
        initialize_database()
        ensure_demo_session()
        insert_default_presets(translation_manager, overwrite=True)
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        raise
    
def initialize_presets_db() -> None:
    """Initialize system message presets table"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    language TEXT NOT NULL,
                    content TEXT NOT NULL,
                    UNIQUE(name, language)
                )
            """)
            conn.commit()
            logger.info("System message presets table initialized.")
    except DatabaseError as e:
        logger.error(f"Failed to initialize presets DB: {e}")
        raise


# 앱 시작 시 DB 초기화 함수 호출
initialize_presets_db()

# database.py

def insert_default_presets(translation_manager, overwrite=True) -> None:
    """기본 프리셋을 데이터베이스에 삽입 또는 업데이트
    
    Args:
        translation_manager: 번역 관리자 인스턴스
        overwrite (bool): 기존 프리셋을 덮어쓸지 여부
    """
    # 프리셋 설정 정의
    preset_configs = [
        PresetConfig("AI_ASSISTANT_PRESET", "ai_assistant"),
        PresetConfig("MINAMI_ASUKA_PRESET", "minami_asuka"),
        PresetConfig("MAKOTONO_AOI_PRESET", "makotono_aoi"),
        PresetConfig("AINO_KOITO_PRESET", "aino_koito")
    ]
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            languages = translation_manager.get_available_languages()
            
            for preset_config in preset_configs:
                for lang in languages:
                    try:
                        # translation_manager에서 프리셋 내용 가져오기
                        content = translation_manager.get_character_setting(preset_config.character_key)
                        
                        if overwrite:
                            # 프리셋 업데이트
                            cursor.execute("""
                                UPDATE system_presets 
                                SET content = ?
                                WHERE name = ? AND language = ?
                            """, (content, preset_config.name, lang))
                            if cursor.rowcount == 0:
                                # 존재하지 않으면 삽입
                                cursor.execute("""
                                    INSERT INTO system_presets (name, language, content)
                                    VALUES (?, ?, ?)
                                """, (preset_config.name, lang, content))
                                logger.info(f"Inserted default preset: {preset_config.name} (language: {lang})")
                            else:
                                logger.info(f"Updated default preset: {preset_config.name} (language: {lang})")
                        else:
                            # 덮어쓰기 없이 삽입
                            cursor.execute("""
                                INSERT INTO system_presets (name, language, content) 
                                VALUES (?, ?, ?)
                            """, (preset_config.name, lang, content))
                            logger.info(f"Inserted default preset: {preset_config.name} (language: {lang})")
                    
                    except sqlite3.IntegrityError as e:
                        logger.warning(
                            f"Preset already exists: {preset_config.name} "
                            f"(language: {lang}): {e}"
                        )
                        continue
                    
                    except Exception as e:
                        logger.error(
                            f"Error inserting preset {preset_config.name} "
                            f"for language {lang}: {e}"
                        )
                        raise PresetInsertionError(
                            f"Failed to insert preset {preset_config.name}: {e}"
                        )
            
            conn.commit()
            logger.info("All default presets inserted/updated successfully")
            
    except PresetInsertionError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during preset insertion: {e}")
        raise PresetInsertionError(f"Failed to insert/update default presets: {e}")
    
def load_system_presets(language: str) -> Dict[str, str]:
    """시스템 메시지 프리셋을 불러옵니다."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, content 
                FROM system_presets 
                WHERE language = ? 
                ORDER BY name ASC
            """, (language,))
            results = cursor.fetchall()
            presets = {name: content for name, content in results} if results else {}
            logger.debug(f"Loaded presets for language {language}: {list(presets.keys())}")
            return presets
            
    except sqlite3.Error as e:
        logger.error(f"Error loading presets for language {language}: {e}")
        return {}


def add_system_preset(
    name: str,
    language: str,
    content: str,
    overwrite: bool = False
) -> PresetResult:
    """새로운 시스템 메시지 프리셋을 추가하거나 업데이트합니다.

    Args:
        name: 프리셋 이름
        language: 언어 코드
        content: 프리셋 내용
        overwrite: 덮어쓰기 여부

    Returns:
        PresetResult: 작업 결과

    Raises:
        PresetManagementError: 프리셋 추가/수정 중 오류 발생 시
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if overwrite:
                cursor.execute("""
                    UPDATE system_presets 
                    SET content = ?
                    WHERE name = ? AND language = ?
                """, (content, name, language))
                operation = "updated"
            else:
                cursor.execute("""
                    INSERT INTO system_presets (name, language, content) 
                    VALUES (?, ?, ?)
                """, (name, language, content))
                operation = "added"
            
            conn.commit()
            logger.info(f"Preset {name} ({language}) successfully {operation}")
            return PresetResult(True, f"프리셋이 성공적으로 {operation}되었습니다.")
            
    except sqlite3.IntegrityError:
        message = "프리셋이 이미 존재합니다."
        logger.warning(f"Preset '{name}' ({language}) already exists")
        return PresetResult(False, message)
    except sqlite3.Error as e:
        logger.error(f"Database error while handling preset {name}: {e}")
        raise PresetManagementError(f"Failed to handle preset: {e}")
    except Exception as e:
        logger.error(f"Unexpected error handling preset {name}: {e}")
        return PresetResult(False, f"오류 발생: {e}")

def delete_system_preset(name: str, language: str) -> PresetResult:
    """시스템 메시지 프리셋을 삭제합니다.

    Args:
        name: 프리셋 이름
        language: 언어 코드

    Returns:
        PresetResult: 작업 결과

    Raises:
        PresetManagementError: 프리셋 삭제 중 오류 발생 시
    """
    if name in DEFAULT_PRESETS:
        message = "기본 프리셋은 삭제할 수 없습니다."
        logger.warning(f"Attempted to delete default preset '{name}' ({language})")
        return PresetResult(False, message)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM system_presets 
                WHERE name = ? AND language = ?
            """, (name, language))
            
            if cursor.rowcount == 0:
                message = "삭제할 프리셋을 찾을 수 없습니다."
                logger.warning(f"Preset {name} ({language}) not found for deletion")
                return PresetResult(False, message)
            
            conn.commit()
            logger.info(f"Preset {name} ({language}) successfully deleted")
            return PresetResult(True, "프리셋이 성공적으로 삭제되었습니다.")
            
    except sqlite3.Error as e:
        logger.error(f"Database error while deleting preset {name}: {e}")
        raise PresetManagementError(f"Failed to delete preset: {e}")
    except Exception as e:
        logger.error(f"Unexpected error deleting preset {name}: {e}")
        return PresetResult(False, f"오류 발생: {e}")

def preset_exists(name: str, language: str) -> bool:
    """프리셋의 존재 여부를 확인합니다.

    Args:
        name: 프리셋 이름
        language: 언어 코드

    Returns:
        bool: 프리셋 존재 여부
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) 
                FROM system_presets 
                WHERE name = ? AND language = ?
            """, (name, language))
            return cursor.fetchone()[0] > 0
            
    except sqlite3.Error as e:
        logger.error(f"Database error checking preset existence: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking preset existence: {e}")
        return False
    
def get_preset_choices(language: str) -> List[str]:
    """프리셋 선택 목록을 가져옵니다."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name 
                FROM system_presets 
                WHERE language = ? 
                ORDER BY name ASC
            """, (language,))
            results = cursor.fetchall()
            # 이름 목록만 반환
            return [name[0] for name in results] if results else []
            
    except sqlite3.Error as e:
        logger.error(f"Error getting preset choices for language {language}: {e}")
        return []

# 프리셋 추가 핸들러
def handle_add_preset(name, language, content, confirm_overwrite=False):
    if not name.strip() or not content.strip():
        return "❌ 프리셋 이름과 내용을 모두 입력해주세요.", gr.update(choices=get_preset_choices(language)), False
    
    exists = preset_exists(name.strip(), language)
    
    if exists and not confirm_overwrite:
        # 프리셋이 존재하지만 덮어쓰기 확인이 이루어지지 않은 경우
        return "⚠️ 해당 프리셋이 이미 존재합니다. 덮어쓰시겠습니까?", gr.update(choices=get_preset_choices(language)), True  # 추가 출력: 덮어쓰기 필요
    
    success, message = add_system_preset(name.strip(), language, content.strip(), overwrite=exists)
    if success:
        presets = get_preset_choices(language)
        return message, gr.update(choices=presets), False  # 덮어쓰기 완료
    else:
        return message, gr.update(choices=get_preset_choices(language)), False


# 프리셋 삭제 핸들러
def handle_delete_preset(name, language):
    if not name:
        return "❌ 삭제할 프리셋을 선택해주세요.", gr.update(choices=get_preset_choices(language))
    success, message = delete_system_preset(name, language)
    if success:
        presets = get_preset_choices(language)
        return message, gr.update(choices=presets)
    else:
        return message, gr.update(choices=get_preset_choices(language))
    
def get_existing_sessions() -> List[str]:
    """Get list of existing session IDs"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT session_id FROM chat_history ORDER BY session_id ASC")
            return [row[0] for row in cursor.fetchall()]
            
    except DatabaseError as e:
        logger.error(f"Error retrieving sessions: {e}")
        return []
    
def save_chat_history_db(history, session_id="demo_session") -> bool:
    """Save chat history to SQLite database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME
                )
            """)
            
            # 세션 존재 여부 확인
            cursor.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,))
            if cursor.fetchone()[0] == 0:
                # 세션이 존재하지 않으면 생성
                current_time = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO sessions (id, name, created_at, updated_at, last_activity)
                    VALUES (?, ?, ?, ?, ?)
                """, (session_id, f"Session {session_id}", current_time, current_time, current_time))
                logger.info(f"Created new session: {session_id}")
            
            for msg in history:
                cursor.execute("""
                    SELECT COUNT(*) FROM chat_history
                    WHERE session_id = ? AND role = ? AND content = ?
                """, (session_id, msg.get("role"), msg.get("content")))
                count = cursor.fetchone()[0]

                if count == 0:
                    cursor.execute("""
                        INSERT INTO chat_history (session_id, role, content)
                        VALUES (?, ?, ?)
                    """, (session_id, msg.get("role"), msg.get("content")))

            conn.commit()
            logger.info(f"DB에 채팅 히스토리 저장 완료 (session_id={session_id})")
            return True
    except sqlite3.OperationalError as e:
        logger.error(f"DB 작업 중 오류: {e}")
        return False
    except Exception as e:
        logger.error(f"Error saving chat history to DB: {e}")
        return False
    
def save_chat_history(history):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"chat_history_{timestamp}.json"
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f"채팅 히스토리 저장 완료: {file_name}")
        return file_name
    except Exception as e:
        logger.error(f"채팅 히스토리 저장 중 오류: {e}")
        return None

def save_chat_history_csv(history):
    """
    채팅 히스토리를 CSV 형태로 저장
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"chat_history_{timestamp}.csv"
    try:
        # CSV 파일 열기
        with open(file_name, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            # 헤더 작성
            writer.writerow(["role", "content"])
            # 각 메시지 row 작성
            for msg in history:
                writer.writerow([msg.get("role"), msg.get("content")])
        logger.info(f"채팅 히스토리 CSV 저장 완료: {file_name}")
        return file_name
    except Exception as e:
        logger.error(f"채팅 히스토리 CSV 저장 중 오류: {e}")
        return None
    
def save_chat_button_click(history):
    if not history:
        return "채팅 이력이 없습니다."
    saved_path = save_chat_history(history)
    if saved_path is None:
        return "❌ 채팅 기록 저장 실패"
    else:
        return f"✅ 채팅 기록이 저장되었습니다: {saved_path}"
    
def load_chat_from_db(session_id: str) -> List[Dict[str, str]]:
    """특정 세션의 채팅 기록을 데이터베이스에서 불러옵니다.

    Args:
        session_id: 불러올 세션의 ID

    Returns:
        List[Dict[str, str]]: 채팅 메시지 목록

    Raises:
        SessionManagementError: 채팅 기록 로딩 중 오류 발생 시
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content, timestamp 
                FROM chat_history 
                WHERE session_id = ? 
                ORDER BY id ASC
            """, (session_id,))
            
            history = []
            for row in cursor.fetchall():
                role, content, timestamp = row
                message = ChatMessage(
                    role=role,
                    content=content,
                    timestamp=datetime.fromisoformat(timestamp) if timestamp else None
                )
                history.append({"role": message.role, "content": message.content})
            
            logger.info(f"Successfully loaded {len(history)} messages from session '{session_id}'")
            return history
            
    except sqlite3.Error as e:
        logger.error(f"Database error loading session '{session_id}': {e}")
        raise SessionManagementError(f"Failed to load session history: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading session '{session_id}': {e}")
        return []

def delete_session_history(session_id: str) -> SessionResult:
    """특정 세션의 모든 채팅 기록을 삭제합니다.

    Args:
        session_id: 삭제할 세션의 ID

    Returns:
        SessionResult: 작업 결과 객체

    Raises:
        SessionManagementError: 세션 삭제 중 오류 발생 시
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 세션 존재 여부 확인
            cursor.execute("""
                SELECT COUNT(*) 
                FROM chat_history 
                WHERE session_id = ?
            """, (session_id,))
            
            count = cursor.fetchone()[0]
            if count == 0:
                message = f"Session '{session_id}' not found"
                logger.warning(message)
                return SessionResult(False, message, 0)
            
            # 세션 삭제
            cursor.execute("""
                DELETE FROM chat_history 
                WHERE session_id = ?
            """, (session_id,))
            
            affected_rows = cursor.rowcount
            conn.commit()
            
            message = f"Successfully deleted session '{session_id}' ({affected_rows} messages)"
            logger.info(message)
            return SessionResult(True, message, affected_rows)
            
    except sqlite3.Error as e:
        error_msg = f"Database error deleting session '{session_id}': {e}"
        logger.error(error_msg)
        raise SessionManagementError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error deleting session '{session_id}': {e}"
        logger.error(error_msg)
        return SessionResult(False, error_msg, 0)

def delete_all_sessions() -> SessionResult:
    """데이터베이스의 모든 세션과 채팅 기록을 삭제합니다.

    Returns:
        SessionResult: 작업 결과 객체

    Raises:
        SessionManagementError: 세션 삭제 중 오류 발생 시
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 삭제 전 총 메시지 수 확인
            cursor.execute("SELECT COUNT(*) FROM chat_history")
            total_messages = cursor.fetchone()[0]
            
            if total_messages == 0:
                message = "No chat history to delete"
                logger.info(message)
                return SessionResult(True, message, 0)
            
            # 모든 채팅 기록 삭제
            cursor.execute("DELETE FROM chat_history")
            affected_rows = cursor.rowcount
            conn.commit()
            
            message = f"Successfully deleted all sessions ({affected_rows} messages)"
            logger.info(message)
            return SessionResult(True, message, affected_rows)
            
    except sqlite3.Error as e:
        error_msg = f"Database error deleting all sessions: {e}"
        logger.error(error_msg)
        raise SessionManagementError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error deleting all sessions: {e}"
        logger.error(error_msg)
        return SessionResult(False, error_msg, 0)
    
def update_system_message_in_db(session_id: str, new_system_message: str):
    """
    지정된 session_id의 system 메시지를 new_system_message로 교체합니다.
    """
    try:
        with sqlite3.connect("chat_history.db") as conn:
            cursor = conn.cursor()
            # 우선 해당 세션의 기존 system 메시지를 모두 삭제
            cursor.execute("""
                DELETE FROM chat_history 
                WHERE session_id = ? AND role = 'system'
            """, (session_id,))
            
            # 새 system 메시지 삽입
            cursor.execute("""
                INSERT INTO chat_history (session_id, role, content, timestamp)
                VALUES (?, 'system', ?, CURRENT_TIMESTAMP)
            """, (session_id, new_system_message))
            
            conn.commit()
        logger.info(f"[update_system_message_in_db] 세션 {session_id}의 system 메시지가 업데이트되었습니다.")
    except Exception as e:
        logger.error(f"[update_system_message_in_db] DB 업데이트 오류: {e}")