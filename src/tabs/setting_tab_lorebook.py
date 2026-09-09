# setting_tab_lorebook.py
"""설정 탭: 로어북(월드 인포) 관리.

- 캐릭터 로어북: 캐릭터 카드의 extensions.character_book 을 편집 (ST 호환)
- 스토리 로어북: lorebooks 테이블에 저장되는 독립 로어북 (스토리 챗 연동은 추후)
- SillyTavern World Info JSON 임포트/익스포트 지원
"""

import gradio as gr

from pathlib import Path

from ai_companion_core import logger

from ..characters.card_registry import (
    get_character_lorebook,
    list_cards,
    update_character_lorebook,
)
from ..characters.lorebook import (
    LorebookError,
    empty_lorebook,
    export_lorebook_json,
    load_lorebook_json,
)
from ..common.database import delete_lorebook, get_lorebook, list_lorebooks, save_lorebook
from ..common.translations import _

EXPORT_DIR = Path("outputs/lorebooks")

DF_HEADERS = [
    _("lorebook_col_id"),
    _("lorebook_col_comment"),
    _("lorebook_col_keys"),
    _("lorebook_col_constant"),
    _("lorebook_col_enabled"),
    _("lorebook_col_priority"),
    _("lorebook_col_order"),
    _("lorebook_col_position"),
]

# 엔트리 편집 폼 컴포넌트 순서 (핸들러 입출력 순서와 일치)
ENTRY_FORM_COUNT = 11


# ---------------------------------------------------------------------------
# 공용 헬퍼
# ---------------------------------------------------------------------------

def _entries_to_df(book: dict) -> list:
    """엔트리 목록을 Dataframe 행으로 변환합니다."""
    rows = []
    for entry in book.get("entries", []):
        rows.append([
            entry.get("id"),
            entry.get("comment", ""),
            ", ".join(entry.get("keys", [])),
            bool(entry.get("constant")),
            bool(entry.get("enabled", True)),
            entry.get("priority", 10),
            entry.get("insertion_order", 100),
            entry.get("position", "before_char"),
        ])
    return rows


def _form_to_entry(comment, keys_str, secondary_str, content, constant, selective, enabled, priority, insertion_order, position, case_sensitive, entry_id=None) -> dict:
    """편집 폼 값을 엔트리 dict로 변환합니다."""
    def _split(value):
        return [k.strip() for k in str(value or "").split(",") if k.strip()]

    entry = {
        "keys": _split(keys_str),
        "secondary_keys": _split(secondary_str),
        "content": str(content or ""),
        "comment": str(comment or "").strip(),
        "constant": bool(constant),
        "selective": bool(selective),
        "enabled": bool(enabled),
        "insertion_order": int(insertion_order) if insertion_order is not None else 100,
        "position": position if position in ("before_char", "after_char") else "before_char",
        "case_sensitive": bool(case_sensitive),
        "priority": int(priority) if priority is not None else 10,
        "extensions": {},
    }
    if entry_id is not None:
        entry["id"] = int(entry_id)
    return entry


def _clear_form():
    """폼 초기화 gr.update 목록을 반환합니다."""
    return [
        gr.update(value=""),   # comment
        gr.update(value=""),   # keys
        gr.update(value=""),   # secondary_keys
        gr.update(value=""),   # content
        gr.update(value=False),  # constant
        gr.update(value=False),  # selective
        gr.update(value=True),   # enabled
        gr.update(value=10),     # priority
        gr.update(value=100),    # insertion_order
        gr.update(value="before_char"),  # position
        gr.update(value=False),  # case_sensitive
        gr.update(value=None),   # row selector
    ]


# ---------------------------------------------------------------------------
# 캐릭터 로어북 핸들러
# ---------------------------------------------------------------------------

def _card_names():
    return [c["name"] for c in list_cards()]


def _load_card_book(name):
    """카드 번들에서 로어북을 읽어옵니다 (없으면 빈 북)."""
    if not name:
        return empty_lorebook()
    book = get_character_lorebook(name)
    if book is None:
        book = empty_lorebook(name)
    book["name"] = name
    return book


def handle_load_character_book(name):
    book = _load_card_book(name)
    count = len(book.get("entries", []))
    status = f"✅ '{name}' 로어북 로드 완료 ({count}개 엔트리)." if name else "❌ 캐릭터를 선택하세요."
    return book, _entries_to_df(book), status


def handle_save_character_book(name, book):
    if not name:
        return book, "❌ 캐릭터를 선택하세요."
    success, message = update_character_lorebook(name, book)
    return book, message


def handle_character_entry_add(book, comment, keys_str, secondary_str, content, constant, selective, enabled, priority, insertion_order, position, case_sensitive):
    if book is None:
        book = empty_lorebook()
    if not str(content or "").strip():
        return book, _entries_to_df(book), "❌ 엔트리 내용을 입력하세요.", *_clear_form()
    entry = _form_to_entry(comment, keys_str, secondary_str, content, constant, selective, enabled, priority, insertion_order, position, case_sensitive)
    entry["id"] = len(book.get("entries", []))
    book.setdefault("entries", []).append(entry)
    status = f"✅ 엔트리가 추가되었습니다. (저장하려면 캐릭터를 선택한 상태에서 변경사항이 자동 반영됩니다)"
    return book, _entries_to_df(book), status, *_clear_form()


def handle_character_entry_update(book, row_index, comment, keys_str, secondary_str, content, constant, selective, enabled, priority, insertion_order, position, case_sensitive):
    if not book or not book.get("entries"):
        return book, _entries_to_df(book or empty_lorebook()), "❌ 편집할 엔트리가 없습니다."
    if row_index is None or int(row_index) < 0 or int(row_index) >= len(book["entries"]):
        return book, _entries_to_df(book), "❌ 목록에서 수정할 엔트리 행을 선택하세요."
    if not str(content or "").strip():
        return book, _entries_to_df(book), "❌ 엔트리 내용을 입력하세요."
    idx = int(row_index)
    entry = _form_to_entry(comment, keys_str, secondary_str, content, constant, selective, enabled, priority, insertion_order, position, case_sensitive, entry_id=book["entries"][idx].get("id", idx))
    book["entries"][idx] = entry
    return book, _entries_to_df(book), "✅ 엔트리가 수정되었습니다."


def handle_character_entry_delete(book, row_index):
    if not book or not book.get("entries"):
        return book, _entries_to_df(book or empty_lorebook()), "❌ 삭제할 엔트리가 없습니다."
    if row_index is None or int(row_index) < 0 or int(row_index) >= len(book["entries"]):
        return book, _entries_to_df(book), "❌ 목록에서 삭제할 엔트리 행을 선택하세요."
    idx = int(row_index)
    removed = book["entries"].pop(idx)
    return book, _entries_to_df(book), f"✅ 엔트리 '{removed.get('comment') or removed.get('id')}'가 삭제되었습니다.", *_clear_form()


def handle_character_row_select(evt: gr.SelectData, book):
    """엔트리 목록 행 선택 시 편집 폼을 채웁니다."""
    row_index = evt.index[0] if evt.index else None
    if not book or not book.get("entries") or row_index is None or row_index >= len(book["entries"]):
        return [gr.update()] * ENTRY_FORM_COUNT + [row_index]
    entry = book["entries"][row_index]
    return [
        gr.update(value=entry.get("comment", "")),
        gr.update(value=", ".join(entry.get("keys", []))),
        gr.update(value=", ".join(entry.get("secondary_keys", []))),
        gr.update(value=entry.get("content", "")),
        gr.update(value=bool(entry.get("constant"))),
        gr.update(value=bool(entry.get("selective"))),
        gr.update(value=bool(entry.get("enabled", True))),
        gr.update(value=entry.get("priority", 10)),
        gr.update(value=entry.get("insertion_order", 100)),
        gr.update(value=entry.get("position", "before_char")),
        gr.update(value=bool(entry.get("case_sensitive"))),
        row_index,
    ]


def handle_character_import(book, file, mode):
    """SillyTavern World Info JSON을 캐릭터 로어북으로 가져옵니다."""
    if not file:
        return book, _entries_to_df(book or empty_lorebook()), "❌ JSON 파일을 업로드하세요."
    try:
        imported = load_lorebook_json(file)
    except LorebookError as e:
        return book, _entries_to_df(book or empty_lorebook()), f"❌ {e}"

    if book is None:
        book = empty_lorebook()
    if mode == "replace":
        book = imported
    else:  # merge
        existing = book.setdefault("entries", [])
        for entry in imported.get("entries", []):
            entry["id"] = len(existing)
            existing.append(entry)
    return book, _entries_to_df(book), f"✅ 로어북을 가져왔습니다 ({len(imported.get('entries', []))}개 엔트리)."


def handle_character_export(book):
    if not book or not book.get("entries"):
        return "❌ 내보낼 엔트리가 없습니다.", None
    safe_name = "".join(c for c in str(book.get("name") or "lorebook") if c.isalnum() or c in " -_") or "lorebook"
    try:
        out_path = export_lorebook_json(book, EXPORT_DIR / f"{safe_name}.json")
    except (LorebookError, OSError) as e:
        logger.error(f"로어북 익스포트 실패: {e}")
        return f"❌ 익스포트 실패: {e}", None
    return f"✅ 로어북을 내보냈습니다: {out_path}", str(out_path)


# ---------------------------------------------------------------------------
# 스토리 로어북 핸들러
# ---------------------------------------------------------------------------

def _refresh_book_dropdown(value=None):
    names = list_lorebooks()
    return gr.update(choices=names, value=value if value in names else (names[0] if names else None))


def handle_select_story_book(name):
    book = get_lorebook(name) if name else None
    if book is None:
        book = empty_lorebook(name or "")
    count = len(book.get("entries", []))
    status = f"✅ '{name}' 로어북 로드 완료 ({count}개 엔트리)." if name else "새 스토리 로어북을 만들거나 선택하세요."
    return book, _entries_to_df(book), status


def handle_create_story_book(name):
    if not name or not str(name).strip():
        return gr.update(), gr.update(), empty_lorebook(), [], "❌ 로어북 이름을 입력하세요."
    name = str(name).strip()
    if get_lorebook(name) is not None:
        return gr.update(), gr.update(), empty_lorebook(), [], f"❌ '{name}' 로어북이 이미 존재합니다."
    book = empty_lorebook(name)
    success, message = save_lorebook(name, book)
    return _refresh_book_dropdown(name), gr.update(value=""), book, [], message


def handle_delete_story_book(name):
    if not name:
        return gr.update(), empty_lorebook(), [], "❌ 삭제할 로어북을 선택하세요."
    success, message = delete_lorebook(name)
    return _refresh_book_dropdown(), empty_lorebook(), [], message


def handle_story_entry_add(book, comment, keys_str, secondary_str, content, constant, selective, enabled, priority, insertion_order, position, case_sensitive):
    if book is None:
        book = empty_lorebook()
    if not str(content or "").strip():
        return book, _entries_to_df(book), "❌ 엔트리 내용을 입력하세요.", *_clear_form()
    entry = _form_to_entry(comment, keys_str, secondary_str, content, constant, selective, enabled, priority, insertion_order, position, case_sensitive)
    entry["id"] = len(book.get("entries", []))
    book.setdefault("entries", []).append(entry)
    return book, _entries_to_df(book), "✅ 엔트리가 추가되었습니다. (저장 버튼으로 DB에 반영하세요)", *_clear_form()


def handle_story_entry_update(book, row_index, comment, keys_str, secondary_str, content, constant, selective, enabled, priority, insertion_order, position, case_sensitive):
    if not book or not book.get("entries"):
        return book, _entries_to_df(book or empty_lorebook()), "❌ 편집할 엔트리가 없습니다."
    if row_index is None or int(row_index) < 0 or int(row_index) >= len(book["entries"]):
        return book, _entries_to_df(book), "❌ 목록에서 수정할 엔트리 행을 선택하세요."
    if not str(content or "").strip():
        return book, _entries_to_df(book), "❌ 엔트리 내용을 입력하세요."
    idx = int(row_index)
    entry = _form_to_entry(comment, keys_str, secondary_str, content, constant, selective, enabled, priority, insertion_order, position, case_sensitive, entry_id=book["entries"][idx].get("id", idx))
    book["entries"][idx] = entry
    return book, _entries_to_df(book), "✅ 엔트리가 수정되었습니다."


def handle_story_entry_delete(book, row_index):
    if not book or not book.get("entries"):
        return book, _entries_to_df(book or empty_lorebook()), "❌ 삭제할 엔트리가 없습니다."
    if row_index is None or int(row_index) < 0 or int(row_index) >= len(book["entries"]):
        return book, _entries_to_df(book), "❌ 목록에서 삭제할 엔트리 행을 선택하세요."
    idx = int(row_index)
    removed = book["entries"].pop(idx)
    return book, _entries_to_df(book), f"✅ 엔트리 '{removed.get('comment') or removed.get('id')}'가 삭제되었습니다.", *_clear_form()


def handle_story_row_select(evt: gr.SelectData, book):
    return handle_character_row_select(evt, book)


def handle_story_save(book):
    name = str((book or {}).get("name") or "").strip()
    if not name:
        return book, gr.update(), "❌ 저장할 로어북 이름이 없습니다. 새 로어북을 만들거나 선택하세요."
    success, message = save_lorebook(name, book)
    return book, _refresh_book_dropdown(name), message


def handle_story_import(book, file, mode):
    if not file:
        return book, _entries_to_df(book or empty_lorebook()), "❌ JSON 파일을 업로드하세요."
    try:
        imported = load_lorebook_json(file)
    except LorebookError as e:
        return book, _entries_to_df(book or empty_lorebook()), f"❌ {e}"

    if mode == "replace":
        book = imported
    else:
        if book is None:
            book = empty_lorebook()
        existing = book.setdefault("entries", [])
        for entry in imported.get("entries", []):
            entry["id"] = len(existing)
            existing.append(entry)
    return book, _entries_to_df(book), f"✅ 로어북을 가져왔습니다 ({len(imported.get('entries', []))}개 엔트리)."


def handle_story_export(book):
    return handle_character_export(book)


# ---------------------------------------------------------------------------
# 엔트리 편집 폼 빌더 (캐릭터/스토리 섹션 공용 UI)
# ---------------------------------------------------------------------------

def _build_entry_editor(book_state):
    """엔트리 목록 + 편집 폼 컴포넌트를 생성하고 반환합니다."""
    entry_df = gr.Dataframe(
        headers=DF_HEADERS,
        datatype=["number", "str", "str", "bool", "bool", "number", "number", "str"],
        interactive=False,
        label=_("lorebook_entry_list"),
        elem_classes="lorebook-entry-table",
    )

    gr.Markdown(f"#### {_('lorebook_entry_form_title')}")
    with gr.Row():
        form_comment = gr.Textbox(label=_("lorebook_entry_comment"), placeholder="예: 달빛 검", interactive=True)
        form_keys = gr.Textbox(label=_("lorebook_entry_keys"), placeholder="검, 검검, 달빛검 (쉼표 구분)", interactive=True)
        form_secondary = gr.Textbox(label=_("lorebook_entry_secondary_keys"), placeholder="선택: 보조 키워드 (쉼표 구분)", interactive=True)
    form_content = gr.Textbox(label=_("lorebook_entry_content"), lines=4, placeholder="이 키워드가 발견되면 시스템 메시지에 삽입될 설정 내용", interactive=True)
    with gr.Row():
        form_constant = gr.Checkbox(label=_("lorebook_entry_constant"), value=False, interactive=True)
        form_selective = gr.Checkbox(label=_("lorebook_entry_selective"), value=False, interactive=True)
        form_enabled = gr.Checkbox(label=_("lorebook_entry_enabled"), value=True, interactive=True)
        form_case_sensitive = gr.Checkbox(label=_("lorebook_entry_case_sensitive"), value=False, interactive=True)
    with gr.Row():
        form_priority = gr.Number(label=_("lorebook_entry_priority"), value=10, precision=0, interactive=True)
        form_insertion_order = gr.Number(label=_("lorebook_entry_insertion_order"), value=100, precision=0, interactive=True)
        form_position = gr.Radio(label=_("lorebook_entry_position"), choices=["before_char", "after_char"], value="before_char", interactive=True)

    form_fields = [form_comment, form_keys, form_secondary, form_content, form_constant, form_selective, form_enabled, form_priority, form_insertion_order, form_position, form_case_sensitive]
    row_index_state = gr.State(None)

    with gr.Row():
        add_btn = gr.Button(_("lorebook_add_entry_button"), variant="primary")
        update_btn = gr.Button(_("lorebook_update_entry_button"), variant="secondary")
        delete_btn = gr.Button(_("lorebook_delete_entry_button"), variant="stop")

    return {
        "entry_df": entry_df,
        "form_fields": form_fields,
        "row_index_state": row_index_state,
        "add_btn": add_btn,
        "update_btn": update_btn,
        "delete_btn": delete_btn,
    }


def _build_import_export_section():
    """ST World Info JSON 임포트/익스포트 컴포넌트를 생성하고 반환합니다."""
    with gr.Row():
        import_file = gr.File(label="JSON", file_types=[".json"], type="filepath")
        import_mode = gr.Radio(label=_("lorebook_import_mode"), choices=[_("lorebook_import_merge"), _("lorebook_import_replace")], value=_("lorebook_import_merge"))
    with gr.Row():
        import_btn = gr.Button(_("lorebook_import_button"), variant="secondary")
        export_btn = gr.Button(_("lorebook_export_button"), variant="secondary")
    export_file = gr.File(label=_("lorebook_exported_file"), interactive=False)
    return import_file, import_mode, import_btn, export_btn, export_file


def create_lorebook_tab():
    with gr.Tab(_("lorebook_tab_title")):
        gr.Markdown(_("lorebook_tab_desc"))

        with gr.Tabs():
            # ----- 캐릭터 로어북 -----
            with gr.Tab(_("lorebook_character_section")):
                gr.Markdown(_("lorebook_character_desc"))
                char_book_state = gr.State(None)

                with gr.Row():
                    char_dropdown = gr.Dropdown(label=_("lorebook_character_select"), choices=_card_names(), value=None, interactive=True)
                    char_load_btn = gr.Button(_("lorebook_load_button"), variant="primary")
                char_status = gr.Textbox(label=_("lorebook_status"), interactive=False)

                editor = _build_entry_editor(char_book_state)

                # 캐릭터 로어북은 엔트리 추가/수정/삭제 시 카드에 즉시 저장
                def _wrap_with_save(handler):
                    def wrapped(book, *args):
                        results = handler(book, *args)
                        new_book = results[0]
                        df = results[1]
                        status = results[2]
                        rest = list(results[3:])
                        name = str((new_book or {}).get("name") or "")
                        if name and new_book is not None:
                            _, save_msg = update_character_lorebook(name, new_book)
                            status = f"{status}\n{save_msg}"
                        return (new_book, df, status, *rest)
                    return wrapped

                char_load_btn.click(fn=handle_load_character_book, inputs=[char_dropdown], outputs=[char_book_state, editor["entry_df"], char_status])

                editor["entry_df"].select(fn=handle_character_row_select, inputs=[char_book_state], outputs=[*editor["form_fields"], editor["row_index_state"]])

                editor["add_btn"].click(
                    fn=_wrap_with_save(handle_character_entry_add),
                    inputs=[char_book_state, *editor["form_fields"]],
                    outputs=[char_book_state, editor["entry_df"], char_status, *editor["form_fields"], editor["row_index_state"]],
                )
                editor["update_btn"].click(
                    fn=_wrap_with_save(handle_character_entry_update),
                    inputs=[char_book_state, editor["row_index_state"], *editor["form_fields"]],
                    outputs=[char_book_state, editor["entry_df"], char_status],
                )
                editor["delete_btn"].click(
                    fn=_wrap_with_save(handle_character_entry_delete),
                    inputs=[char_book_state, editor["row_index_state"]],
                    outputs=[char_book_state, editor["entry_df"], char_status, *editor["form_fields"], editor["row_index_state"]],
                )

                with gr.Accordion(_("lorebook_import_title"), open=False):
                    char_import_file, char_import_mode, char_import_btn, char_export_btn, char_export_file = _build_import_export_section()

                char_import_btn.click(
                    fn=handle_character_import,
                    inputs=[char_book_state, char_import_file, char_import_mode],
                    outputs=[char_book_state, editor["entry_df"], char_status],
                )
                char_export_btn.click(
                    fn=handle_character_export,
                    inputs=[char_book_state],
                    outputs=[char_status, char_export_file],
                )

            # ----- 스토리 로어북 -----
            with gr.Tab(_("lorebook_story_section")):
                gr.Markdown(_("lorebook_story_desc"))
                story_book_state = gr.State(None)

                with gr.Row():
                    story_dropdown = gr.Dropdown(label=_("lorebook_book_select"), choices=list_lorebooks(), value=None, interactive=True)
                    story_name = gr.Textbox(label=_("lorebook_book_name"), placeholder="새 로어북 이름", interactive=True)
                with gr.Row():
                    story_load_btn = gr.Button(_("lorebook_load_button"), variant="secondary")
                    story_create_btn = gr.Button(_("lorebook_new_button"), variant="primary")
                    story_delete_btn = gr.Button(_("lorebook_delete_book_button"), variant="stop")
                    story_save_btn = gr.Button(_("lorebook_save_button"), variant="primary")
                story_status = gr.Textbox(label=_("lorebook_status"), interactive=False)

                story_editor = _build_entry_editor(story_book_state)

                story_dropdown.change(fn=handle_select_story_book, inputs=[story_dropdown], outputs=[story_book_state, story_editor["entry_df"], story_status])
                story_load_btn.click(fn=handle_select_story_book, inputs=[story_dropdown], outputs=[story_book_state, story_editor["entry_df"], story_status])
                story_create_btn.click(fn=handle_create_story_book, inputs=[story_name], outputs=[story_dropdown, story_name, story_book_state, story_editor["entry_df"], story_status])
                story_delete_btn.click(fn=handle_delete_story_book, inputs=[story_dropdown], outputs=[story_dropdown, story_book_state, story_editor["entry_df"], story_status])
                story_save_btn.click(fn=handle_story_save, inputs=[story_book_state], outputs=[story_book_state, story_dropdown, story_status])

                story_editor["entry_df"].select(fn=handle_story_row_select, inputs=[story_book_state], outputs=[*story_editor["form_fields"], story_editor["row_index_state"]])

                story_editor["add_btn"].click(
                    fn=handle_story_entry_add,
                    inputs=[story_book_state, *story_editor["form_fields"][:-1]],
                    outputs=[story_book_state, story_editor["entry_df"], story_status, *story_editor["form_fields"], story_editor["row_index_state"]],
                )
                story_editor["update_btn"].click(
                    fn=handle_story_entry_update,
                    inputs=[story_book_state, story_editor["row_index_state"], *story_editor["form_fields"][:-1]],
                    outputs=[story_book_state, story_editor["entry_df"], story_status],
                )
                story_editor["delete_btn"].click(
                    fn=handle_story_entry_delete,
                    inputs=[story_book_state, story_editor["row_index_state"]],
                    outputs=[story_book_state, story_editor["entry_df"], story_status, *story_editor["form_fields"], story_editor["row_index_state"]],
                )

                with gr.Accordion(_("lorebook_import_title"), open=False):
                    story_import_file, story_import_mode, story_import_btn, story_export_btn, story_export_file = _build_import_export_section()

                story_import_btn.click(
                    fn=handle_story_import,
                    inputs=[story_book_state, story_import_file, story_import_mode],
                    outputs=[story_book_state, story_editor["entry_df"], story_status],
                )
                story_export_btn.click(
                    fn=handle_story_export,
                    inputs=[story_book_state],
                    outputs=[story_status, story_export_file],
                )