# 로어북(Lorebook) 기능 구현 계획 — 캐릭터 로어북 + 스토리 로어북

## Context (배경)

- 캐릭터 페르소나 챗과 스토리 챗에 각각 로어북(월드 인포) 기능이 필요함
- 현재 코드베이스에 로어북/월드인포 구현은 전혀 없음 (grep 확인)
- **범위 결정 (사용자 확정)**:
  1. 로어북 **데이터/엔진/UI만 먼저** 구현. 스토리 챗 자체 구현은 별도 작업으로 남김 (Gradio storyteller 탭은 현재 "Under Construction")
  2. **SillyTavern 호환** 형식 사용 (카드 `extensions.character_book` 스키마 + ST world info JSON)
  3. 관리 UI: 설정 팝업에 **"로어북 관리" 탭 신설** + 캐릭터 카드 UI(페르소나 탭)에 통합
  4. 트리거: **재귀 활성화(recursive scanning) + 우선순위(priority) + 토큰 예산** 지원

## Approach (제안)

ST 호환 로어북 코어 하나를 만들어 캐릭터/스토리 두 컨텍스트에서 재사용한다.

### 1. 데이터 모델 — SillyTavern character_book 스키마

`src/characters/lorebook.py` (신규):

```python
# 엔트리 (ST character_book.entries 항목과 필드명 호환)
{
  "id": int,
  "keys": [str],              # 트리거 키워드
  "secondary_keys": [str],    # selective AND 조건
  "content": str,
  "comment": str,             # 엔트리 제목/메모
  "constant": bool,           # True면 키워드 무관 상시 삽입
  "selective": bool,          # True면 키워드 트리거 필요
  "enabled": bool,
  "insertion_order": int,     # 낮을수록 먼저 삽입
  "position": "before_char" | "after_char",
  "case_sensitive": bool,
  "priority": int,            # 높을수록 예산 부족 시 마지막에 잘림
  "extensions": {},
}
# 북: {"name": str, "scan_depth": int, "token_budget": int,
#      "recursive_scanning": bool, "entries": [...]}
```

- `normalize_lorebook(raw)` — ST world info JSON / 카드 character_book을 내부 스키마로 정규화 (누락 필드 기본값)
- `lorebook_to_st(book)` — 익스포트용 ST 형식 변환

### 2. 활성화 엔진 — `src/characters/lorebook.py`

```python
def activate_entries(entries, scan_text, *, token_budget=0, recursive=True,
                     max_recursion=3) -> list[dict]
```

- **스캔**: `scan_text`(최근 대화 텍스트, `scan_depth`로 잘린 메시지들)에서 `keys` 매칭. `case_sensitive` 옵션, `selective`는 secondary_keys AND 로직
- **constant** 엔트리는 항상 활성화
- **재귀 활성화**: 활성화된 엔트리의 `content`를 스캔 텍스트에 합쳐 재스캔 → 새 활성화가 없을 때까지 (최대 `max_recursion`회)
- **우선순위/예산**: `token_budget > 0`이면 대략 글자수 기반 예산 추정 후, `priority` 낮은 순서대로 탈락 (동률 시 `insertion_order` 기준)
- 결과는 `insertion_order` 오름차순 정렬 리스트 반환
- 내용에는 `apply_macros`(`character_card.py:165`, {{char}}/{{user}} 치환) 적용

### 3. 저장소 — `src/common/database.py`

- **캐릭터 로어북**: `character_cards.card_json`의 `data.extensions.character_book`에 그대로 저장 (ST 호환 유지, 별도 테이블 불필요). 레거시/마이그레이션 카드는 로어북 없음 → 빈 북 생성 가능
- **스토리 로어북**: 신규 테이블
  ```sql
  CREATE TABLE IF NOT EXISTS lorebooks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL UNIQUE,
      book_json TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
  ```
  (`initialize_database()`에 추가)
- CRUD 함수: `save_lorebook`, `get_lorebook`, `list_lorebooks`, `delete_lorebook`, `get_character_lorebook(name)` (character_cards에서 읽어 정규화)

### 4. 챗 주입 — `src/main/chatbot/chatbot.py`

- `chat_wrapper`(~L515)와 `process_message_bot`의 기존 주입 블록(`persona_context + memory_context`를 시스템 메시지에 덧붙이는 패턴)에 **lorebook_context 추가**:
  - 활성 캐릭터의 카드 번들에서 `extensions.character_book` 조회 (`get_card_bundle` 재사용)
  - 최근 메시지(`scan_depth` 기본 2~4개) 텍스트로 `activate_entries` 실행
  - `position: before_char` 엔트리는 시스템 프롬프트(캐릭터 정보) 앞, `after_char`는 뒤에 배치. 단순화를 위해 1차 구현은 시스템 메시지 끝에 `[World Info]` 블록으로 append하고 position 필드는 저장만 지원 (주석으로 TODO)
  - 주입된 엔트리는 로그로 기록 (검증 용이)
- 스토리 로어북 주입은 스토리 챗 구현 시 사용할 수 있도록 엔진 함수만 준비 (chatbot에는 연동하지 않음)

### 5. 카드 임포트/익스포트 보존 — `src/characters/card_registry.py`, `character_card.py`

- `normalize_card_data`가 `extensions`를 dict로 보존하므로 `character_book`은 이미 임포트 시 함께 들어옴 → 정규화 단계에서 `character_book`이 있으면 `normalize_lorebook`으로 한 번 정규화해서 저장
- `card_to_v2` / PNG·JSON 익스포트는 extensions를 그대로 직렬화하므로 character_book 자동 보존 확인
- 카드에 로어북이 없는 경우 편집 UI에서 새로 만들어 extensions에 저장하는 경로 추가 (`save_card` 재사용)

### 6. 관리 UI

**(a) 설정 팝업 신설 탭** — `src/tabs/setting_tab_lorebook.py` (신규), `src/tabs/__init__.py`의 `create_settings_popup` 내 `with gr.Tabs():` 블록에 `create_lorebook_tab()` 추가

- **캐릭터 로어북 섹션**:
  - 캐릭터(카드) 선택 드롭다운 (`list_cards()` + 런타임 characters)
  - 엔트리 목록 `gr.Dataframe` (comment / keys / constant / priority / insertion_order / enabled)
  - 엔트리 추가/수정/삭제 폼: comment, keys(콤마 구분), secondary_keys, content(textbox), constant, selective, enabled, priority, insertion_order, position
  - 저장 시 카드 번들의 extensions.character_book 갱신 → `save_card`
  - ST world info JSON 파일 임포트 → 해당 캐릭터 북에 병합 or 교체
- **스토리 로어북 섹션**:
  - 북 생성(이름) / 삭제 / 선택 드롭다운
  - 동일한 엔트리 편집 폼 (섹션 간 코드 공유: 엔트리 편집 UI 빌더를 모듈 내 헬퍼 함수로 추출)
  - ST world info JSON 임포트/익스포트 (`lorebooks` 테이블 대상)

**(b) 캐릭터 카드 UI 통합** — `src/tabs/setting_tab_persona.py`

- 카드 미리보기(`handle_preview_card`)에 `character_book.entries` 개수 표시 추가
- 카드 익스포트 시 character_book 보존 확인 (기존 `_build_card_for_export` 흐름 그대로)

### 7. i18n — `translations/{ko,en,ja,zh_CN,zh_TW}.json`

- `lorebook_*` 키 추가 (탭 제목, 섹션 제목, 엔트리 필드 라벨, 버튼, 상태 메시지). 기존 탭의 `_("...")` 패턴(`src/common/translations.py`) 준수

## Files to modify

| 파일 | 작업 |
|---|---|
| `src/characters/lorebook.py` | **신규** — 스키마 정규화, 활성화 엔진(재귀/우선순위/예산), ST 변환 |
| `src/common/database.py` | `lorebooks` 테이블 + CRUD, `get_character_lorebook` |
| `src/characters/card_registry.py` | 임포트 시 character_book 정규화, 카드 북 갱신 헬퍼 |
| `src/main/chatbot/chatbot.py` | `chat_wrapper`/`process_message_bot`에 로어북 주입 |
| `src/tabs/setting_tab_lorebook.py` | **신규** — 로어북 관리 탭 |
| `src/tabs/__init__.py` | 탭 등록 |
| `src/tabs/setting_tab_persona.py` | 카드 미리보기에 로어북 정보 표시 |
| `translations/*.json` (5개) | i18n 키 |

## Reuse

- `src/characters/character_card.py`: `normalize_card_data`, `apply_macros` (L165), `card_to_v2`, `export_card_json/png`
- `src/characters/card_registry.py`: `save_card`, `get_card_bundle`, `list_cards`, `register_card_runtime`
- `src/common/database.py`: `get_db_connection`, `character_cards` 테이블 패턴, `initialize_database()`
- `src/main/chatbot/chatbot.py`: persona/memory 컨텍스트 주입 패턴 (L515~540)
- `src/tabs/setting_tab_persona.py`: 탭 UI + 핸들러 패턴, `ui_component` 공유 패턴

## Steps

- [ ] 1. `src/characters/lorebook.py`: 스키마 정규화(`normalize_lorebook`, `lorebook_to_st`) + 활성화 엔진(`activate_entries`: 키워드 스캔, selective/constant, 재귀, priority/토큰예산, insertion_order 정렬)
- [ ] 2. `database.py`: `lorebooks` 테이블 생성 + `save_lorebook/get_lorebook/list_lorebooks/delete_lorebook/get_character_lorebook`
- [ ] 3. `card_registry.py`: 임포트 시 `extensions.character_book` 정규화 저장 + `update_character_lorebook(name, book)` 헬퍼
- [ ] 4. `chatbot.py`: 주입 로직 (활성 캐릭터 북 → 활성화 → 시스템 메시지 append, 로그 기록)
- [ ] 5. `setting_tab_lorebook.py`: 캐릭터/스토리 로어북 편집 UI + ST JSON 임포트/익스포트, `tabs/__init__.py` 등록
- [ ] 6. `setting_tab_persona.py`: 카드 미리보기에 로어북 엔트리 수 표시
- [ ] 7. i18n 키 5개 언어 추가
- [ ] 8. 검증 (아래)

## Verification

1. **엔진 단위 확인** (python REPL/스크립트):
   - `normalize_lorebook`이 ST world info JSON 샘플을 정규화하는지
   - 키워드 매칭 → 재귀 활성화(엔트리 content가 다른 엔트리 키를 트리거) → priority 낮은 엔트리가 예산 초과 시 탈락하는지
2. **카드 왕복**: character_book이 포함된 ST 카드 PNG/JSON 임포트 → 로어북 탭에서 엔트리 보임 → JSON 익스포트 → character_book 보존 확인
3. **챗 주입**: `python app.py` 실행 → 로어북 탭에서 캐릭터에 엔트리 추가 (예: keys=["검"], content="검은 이름이 '달빛'인 유일한 검") → 챗에서 "검" 언급 시 로그에 활성화 엔트리 기록 + 응답에 반영되는지
4. **스토리 로어북**: 북 생성/엔트리 편집/JSON 임포트·익스포트 동작 확인 (주입은 스토리 챗 구현 시)
5. 프론트엔드 무변경이므로 `npm run lint/build` 불필요. Python 변경이므로 `python app.py` 수동 플로우 테스트를 PR 노트에 기재