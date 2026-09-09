# setting_tab_persona.py
"""설정 탭: SillyTavern 캐릭터 카드 임포트/익스포트 + 유저 페르소나 관리."""

from pathlib import Path

import gradio as gr

from ai_companion_core import logger

from ..start_app import ui_component
from ..common.character_info import characters
from ..common.database import load_system_presets
from ..common.default_language import default_language
from ..characters.character_card import export_card_png, export_card_json, CharacterCardError
from ..characters.card_registry import import_card_file, delete_card, get_card_bundle, list_cards, legacy_prompt_to_card
from ..characters.user_persona import (
    NO_PERSONA_VALUE,
    add_user_persona,
    delete_user_persona,
    get_persona_by_id,
    get_persona_choices,
    set_active_persona,
    update_user_persona,
)

EXPORT_DIR = Path("outputs/st_cards")


def _refresh_character_dropdown():
    return gr.update(choices=list(characters.keys()))


def _refresh_persona_dropdown():
    return gr.update(choices=get_persona_choices())


def handle_preview_card(file):
    """업로드된 카드 파일을 파싱해 미리보기를 반환합니다."""
    if not file:
        return "카드 파일을 업로드하세요.", None
    try:
        from ..characters.character_card import parse_character_card

        card_data = parse_character_card(file)
    except CharacterCardError as e:
        return f"❌ {e}", None

    if not card_data:
        return "❌ 파일에서 캐릭터 카드 데이터를 찾을 수 없습니다. (PNG의 tEXt 'chara'/'ccv3' 청크 또는 ST 형식 JSON 필요)", None

    summary = {
        "name": card_data.get("name"),
        "spec": card_data.get("spec"),
        "creator": card_data.get("creator"),
        "character_version": card_data.get("character_version"),
        "tags": card_data.get("tags"),
        "description": (card_data.get("description") or "")[:300],
        "personality": (card_data.get("personality") or "")[:300],
        "scenario": (card_data.get("scenario") or "")[:300],
        "first_mes": (card_data.get("first_mes") or "")[:300],
        "system_prompt": (card_data.get("system_prompt") or "")[:300],
        "alternate_greetings_count": len(card_data.get("alternate_greetings") or []),
    }
    return f"✅ 카드 파싱 성공: {card_data.get('name')} ({card_data.get('spec')})", summary


def _imported_card_names():
    return [c["name"] for c in list_cards() if c["source"] == "import"]


def refresh_card_dropdowns():
    """페이지 로드 시 임포트 카드/캐릭터 드롭다운을 최신 상태로 갱신합니다.

    브라우저 새로고침 시 컴포넌트가 앱 시작 시점의 choices 스냅샷으로 돌아가므로,
    세션 중 임포트한 카드가 목록에서 사라지는 문제를 방지합니다.
    """
    return gr.update(choices=_imported_card_names()), _refresh_character_dropdown()


def handle_import_card(file):
    """카드 파일을 임포트하여 캐릭터로 등록합니다."""
    if not file:
        return "❌ 카드 파일을 업로드하세요.", None, _refresh_character_dropdown(), gr.update(choices=_imported_card_names())
    success, message, name = import_card_file(file)
    return message, None, _refresh_character_dropdown(), gr.update(choices=_imported_card_names(), value=name if success else None)


def handle_delete_card(name):
    """선택한 임포트 카드를 삭제합니다."""
    if not name:
        return "❌ 삭제할 카드를 선택하세요.", _refresh_character_dropdown(), gr.update(choices=_imported_card_names())
    success, message = delete_card(name)
    return message, _refresh_character_dropdown(), gr.update(choices=_imported_card_names(), value=None)


def _build_card_for_export(name: str):
    """캐릭터 이름으로 익스포트용 card_data를 얻습니다 (카드 없으면 레거시 프리셋에서 변환)."""
    bundle = get_card_bundle(name)
    if bundle:
        return bundle.get("data") or {}
    preset_name = characters.get(name, {}).get("preset_name", name)
    presets = load_system_presets(default_language)
    prompt_text = presets.get(preset_name, "")
    if not prompt_text:
        return None
    return legacy_prompt_to_card(name, prompt_text)


def handle_export_card(name: str, fmt: str):
    """캐릭터를 SillyTavern V2 카드로 익스포트합니다."""
    if not name:
        return "❌ 익스포트할 캐릭터를 선택하세요.", None
    card_data = _build_card_for_export(name)
    if not card_data:
        return f"❌ '{name}' 캐릭터의 프롬프트를 찾을 수 없습니다.", None

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in name if c.isalnum() or c in " -_") or "character"
    avatar_path = characters.get(name, {}).get("profile_image")

    try:
        if fmt == "PNG":
            out_path = export_card_png(card_data, avatar_path, EXPORT_DIR / f"{safe_name}.png")
        else:
            out_path = export_card_json(card_data, EXPORT_DIR / f"{safe_name}.json")
    except (CharacterCardError, OSError) as e:
        logger.error(f"카드 익스포트 실패: {e}")
        return f"❌ 익스포트 실패: {e}", None

    return f"✅ SillyTavern V2 카드로 익스포트했습니다: {out_path}", str(out_path)


# ---------------------------------------------------------------------------
# 유저 페르소나 관리 핸들러
# ---------------------------------------------------------------------------


def handle_select_persona(persona_value: str):
    """페르소나 선택 시 편집 필드를 채웁니다."""
    if not persona_value or persona_value == NO_PERSONA_VALUE:
        return "", "", None
    persona = get_persona_by_id(persona_value)
    if not persona:
        return "", "", None
    return persona.name, persona.description, persona.avatar_path


def _sync_chat_persona_dropdown():
    """채팅 페이지의 유저 페르소나 드롭다운을 활성 페르소나 기준으로 동기화합니다."""
    from ..characters.user_persona import get_active_persona

    active = get_active_persona()
    return gr.update(choices=get_persona_choices(), value=str(active.id) if active else NO_PERSONA_VALUE)


def handle_add_persona(name, description, avatar, activate):
    success, message = add_user_persona(name, description, avatar, activate=bool(activate))
    return message, _refresh_persona_dropdown(), _sync_chat_persona_dropdown()


def handle_update_persona(persona_value, name, description, avatar):
    if not persona_value or persona_value == NO_PERSONA_VALUE:
        return "❌ 수정할 페르소나를 목록에서 선택하세요.", _refresh_persona_dropdown(), gr.update()
    persona = get_persona_by_id(persona_value)
    if not persona:
        return "❌ 페르소나를 찾을 수 없습니다.", _refresh_persona_dropdown(), gr.update()
    success, message = update_user_persona(persona.id, name, description, avatar)
    return message, _refresh_persona_dropdown(), _sync_chat_persona_dropdown()


def handle_delete_persona(persona_value):
    if not persona_value or persona_value == NO_PERSONA_VALUE:
        return "❌ 삭제할 페르소나를 목록에서 선택하세요.", _refresh_persona_dropdown(), "", "", None
    persona = get_persona_by_id(persona_value)
    if not persona:
        return "❌ 페르소나를 찾을 수 없습니다.", _refresh_persona_dropdown(), "", "", None
    success, message = delete_user_persona(persona.id)
    return message, _refresh_persona_dropdown(), "", "", None


def handle_activate_persona(persona_value):
    if not persona_value or persona_value == NO_PERSONA_VALUE:
        success, message = set_active_persona(None)
    else:
        success, message = set_active_persona(int(persona_value))
    return message, _refresh_persona_dropdown(), _sync_chat_persona_dropdown()


def create_persona_management_tab():
    with gr.Tab("캐릭터 & 페르소나"):
        # ----- Section A: SillyTavern 카드 임포트 -----
        with gr.Accordion("SillyTavern 캐릭터 카드 임포트", open=True):
            gr.Markdown("SillyTavern 캐릭터 카드(**PNG V1/V2/V3** 또는 **JSON**)를 업로드하여 새 캐릭터로 추가합니다.")
            with gr.Row():
                card_file = gr.File(label="카드 파일 (.png / .json)", file_types=[".png", ".json"], type="filepath")
                card_preview = gr.JSON(label="카드 미리보기")
            with gr.Row():
                preview_card_btn = gr.Button("미리보기", variant="secondary")
                import_card_btn = gr.Button("임포트", variant="primary")
            card_status = gr.Textbox(label="임포트 결과", interactive=False)

            with gr.Row():
                imported_card_dropdown = gr.Dropdown(label="임포트된 카드 목록", choices=_imported_card_names(), value=None, interactive=True)
                delete_card_btn = gr.Button("임포트 카드 삭제", variant="stop")
            ui_component.imported_card_dropdown = imported_card_dropdown

        # ----- Section B: 카드 익스포트 -----
        with gr.Accordion("SillyTavern 카드 익스포트", open=False):
            gr.Markdown("현재 캐릭터(레거시 프리셋 포함)를 SillyTavern V2 카드로 내보냅니다. SillyTavern에서 그대로 불러올 수 있습니다.")
            with gr.Row():
                export_character_dropdown = gr.Dropdown(label="캐릭터 선택", choices=list(characters.keys()), interactive=True)
                export_format_radio = gr.Radio(label="형식", choices=["PNG", "JSON"], value="PNG")
                export_card_btn = gr.Button("익스포트", variant="primary")
            export_status = gr.Textbox(label="익스포트 결과", interactive=False)
            export_file = gr.File(label="익스포트된 파일", interactive=False)

        # ----- Section C: 유저 페르소나 관리 -----
        with gr.Accordion("유저 페르소나 관리", open=True):
            gr.Markdown("채팅 시 시스템 메시지에 주입될 **유저(당신)의 페르소나**를 만듭니다. 활성화하면 캐릭터가 유저의 이름/정보를 알고 대화합니다.")
            with gr.Row():
                persona_dropdown = gr.Dropdown(label="페르소나 선택", choices=get_persona_choices(), value=NO_PERSONA_VALUE, interactive=True)
                persona_activate_btn = gr.Button("✦ 이 페르소나 활성화", variant="primary")
            with gr.Row():
                persona_name = gr.Textbox(label="이름", placeholder="예: 지훈", interactive=True)
                persona_avatar = gr.Image(label="아바타 (선택)", type="filepath", interactive=True)
            persona_description = gr.Textbox(label="설명 / 성격 / 배경", placeholder="유저에 대한 정보를 자유롭게 적으세요.\n예: 20대 개발자. 커피를 좋아하고 주말엔 등산을 간다.", lines=4, interactive=True)
            persona_activate_on_add = gr.Checkbox(label="추가 후 바로 활성화", value=False)
            with gr.Row():
                persona_add_btn = gr.Button("추가", variant="primary")
                persona_update_btn = gr.Button("수정", variant="secondary")
                persona_delete_btn = gr.Button("삭제", variant="stop")
            persona_status = gr.Textbox(label="페르소나 관리 결과", interactive=False)

        # ----- 이벤트 연결 -----
        preview_card_btn.click(fn=handle_preview_card, inputs=[card_file], outputs=[card_status, card_preview])
        import_card_btn.click(fn=handle_import_card, inputs=[card_file], outputs=[card_status, card_file, ui_component.character_dropdown, imported_card_dropdown])
        delete_card_btn.click(fn=handle_delete_card, inputs=[imported_card_dropdown], outputs=[card_status, ui_component.character_dropdown, imported_card_dropdown])

        export_card_btn.click(fn=handle_export_card, inputs=[export_character_dropdown, export_format_radio], outputs=[export_status, export_file])

        persona_dropdown.change(fn=handle_select_persona, inputs=[persona_dropdown], outputs=[persona_name, persona_description, persona_avatar])
        persona_add_btn.click(fn=handle_add_persona, inputs=[persona_name, persona_description, persona_avatar, persona_activate_on_add], outputs=[persona_status, persona_dropdown, ui_component.user_persona_dropdown])
        persona_update_btn.click(fn=handle_update_persona, inputs=[persona_dropdown, persona_name, persona_description, persona_avatar], outputs=[persona_status, persona_dropdown, ui_component.user_persona_dropdown])
        persona_delete_btn.click(fn=handle_delete_persona, inputs=[persona_dropdown], outputs=[persona_status, persona_dropdown, persona_name, persona_description, persona_avatar])
        persona_activate_btn.click(fn=handle_activate_persona, inputs=[persona_dropdown], outputs=[persona_status, persona_dropdown, ui_component.user_persona_dropdown])