import gradio as gr
from src.common.database import preset_exists, handle_add_preset, load_system_presets, save_chat_history_db, get_preset_choices, handle_delete_preset
from src.tabs.main_tab import MainTab
import logging

from src.characters.preset_images import PRESET_IMAGES

main_tab=MainTab()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def confirm_overwrite(name, content):
    success, message = handle_add_preset(name.strip(), content.strip(), overwrite=True)
    if success:
        return message, gr.update(visible=False), gr.update(visible=False), ""
    else:
        return message, gr.update(visible=False), gr.update(visible=False), ""
                    
def on_add_preset_click(name, content):
    if preset_exists(name.strip()):
        # 프리셋이 이미 존재하면 덮어쓰기 확인을 요청
        return "", gr.update(visible=True), gr.update(visible=True), "⚠️ 해당 프리셋이 이미 존재합니다. 덮어쓰시겠습니까?"
    else:
        success, message = handle_add_preset(name.strip(), content.strip())
        if success:
            return message, gr.update(visible=False), gr.update(visible=False), ""
        else:
            return message, gr.update(visible=False), gr.update(visible=False), ""
        
 # 프리셋 적용 이벤트 수정
def apply_preset(name, session_id, history, language=None):
    if not name:
        return "❌ 적용할 프리셋을 선택해주세요.", history, gr.update()
                
    if language is None:
        language = "ko"
                    
    presets = load_system_presets(language)
    content = presets.get(name, "")
    if not content:
        return "❌ 선택한 프리셋에 내용이 없습니다.", history, gr.update()
        
    # 현재 세션의 히스토리를 초기화하고 시스템 메시지 추가
    new_history = [{"role": "system", "content": content}]
    success = save_chat_history_db(new_history, session_id=session_id)
    if not success:
        return "❌ 프리셋 적용 중 오류가 발생했습니다.", history, gr.update()
    logger.info(f"'{name}' 프리셋을 적용하여 세션을 초기화했습니다.")
                
    image_path = PRESET_IMAGES.get(name)
    return f"✅ '{name}' 프리셋이 적용되었습니다.", new_history, gr.update(value=content), gr.update(value=image_path) if image_path else gr.update()

def create_system_preset_management_tab(default_language, session_id_state, history_state, selected_language_state, system_message_box, profile_image, chatbot):
    with gr.Tab("시스템 메시지 프리셋 관리"):
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                label="프리셋 선택",  # 필요 시 번역 키로 변경
                choices=get_preset_choices(default_language),
                value=get_preset_choices(default_language)[0] if get_preset_choices(default_language) else None
            )
            refresh_preset_button = gr.Button("프리셋 목록 갱신")
            refresh_preset_button.click(
                fn=main_tab.refresh_preset_list,
                inputs=[selected_language_state],
                outputs=[preset_dropdown]
            )
            apply_preset_btn = gr.Button("프리셋 적용")
                
        with gr.Row():
            preset_name = gr.Textbox(
                label="새 프리셋 이름",
                placeholder="예: 친절한 비서",
                interactive=True
            )
            preset_content = gr.Textbox(
                label="프리셋 내용",
                placeholder="프리셋으로 사용할 시스템 메시지를 입력하세요.",
                lines=4,
                interactive=True
            )
                
        with gr.Row():
            add_preset_btn = gr.Button("프리셋 추가", variant="primary")
            delete_preset_btn = gr.Button("프리셋 삭제", variant="secondary")
                
        preset_info = gr.Textbox(
            label="프리셋 관리 결과",
            interactive=False
        )
                
        # 덮어쓰기 확인을 위한 컴포넌트 추가 (처음에는 숨김)
        with gr.Row():
            confirm_overwrite_btn = gr.Button("확인", variant="primary", visible=False)
            cancel_overwrite_btn = gr.Button("취소", variant="secondary", visible=False)
                
        overwrite_message = gr.Textbox(
            label="덮어쓰기 메시지",
            value="",
            interactive=False
        )
                    
        add_preset_btn.click(
            fn=on_add_preset_click,
            inputs=[preset_name, preset_content],
            outputs=[preset_info, confirm_overwrite_btn, cancel_overwrite_btn, overwrite_message]
        )
                    
        confirm_overwrite_btn.click(
            fn=confirm_overwrite,
            inputs=[preset_name, preset_content],
            outputs=[preset_info, confirm_overwrite_btn, cancel_overwrite_btn, overwrite_message]
        )
                
        # 덮어쓰기 취소 버튼 클릭 시
        def cancel_overwrite():
            return "❌ 덮어쓰기가 취소되었습니다.", gr.update(visible=False), gr.update(visible=False), ""
                    
        cancel_overwrite_btn.click(
            fn=cancel_overwrite,
            inputs=[],
            outputs=[preset_info, confirm_overwrite_btn, cancel_overwrite_btn, overwrite_message]
        )
                    
        with gr.Row(visible=False) as delete_preset_confirm_row:
            delete_preset_confirm_msg = gr.Markdown("⚠️ **정말로 선택한 프리셋을 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.**")
            delete_preset_yes_btn = gr.Button("✅ 예", variant="danger")
            delete_preset_no_btn = gr.Button("❌ 아니요", variant="secondary")
                    
        # 프리셋 삭제 확인 버튼 클릭 시 실제 삭제 수행
        def confirm_delete_preset(name, confirm):
            if confirm:
                success, message = handle_delete_preset(name, default_language)
                if success:
                    return message, gr.update(visible=False), gr.update(choices=get_preset_choices(default_language))
                else:
                    return f"❌ {message}", gr.update(visible=False), gr.update(choices=get_preset_choices(default_language))
            else:
                return "❌ 삭제가 취소되었습니다.", gr.update(visible=False), gr.update(choices=get_preset_choices(default_language))
                        
        # 프리셋 삭제 버튼과 확인 버튼의 상호작용 연결
        delete_preset_btn.click(
            fn=lambda : (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)),
            inputs=[preset_dropdown],
            outputs=[delete_preset_confirm_row, delete_preset_yes_btn, delete_preset_no_btn]
        )
                    
        delete_preset_yes_btn.click(
            fn=confirm_delete_preset,
            inputs=[preset_dropdown, gr.State(True)],  # confirm=True
            outputs=[preset_info, delete_preset_confirm_row, preset_dropdown]
        )

        # 프리셋 삭제 취소 버튼 클릭 시
        delete_preset_no_btn.click(
            fn=lambda: ("❌ 삭제가 취소되었습니다.", gr.update(visible=False), preset_dropdown),
            inputs=[],
            outputs=[preset_info, delete_preset_confirm_row, preset_dropdown]
        )
                
        apply_preset_btn.click(
            fn=apply_preset,
            inputs=[preset_dropdown, session_id_state, history_state, selected_language_state],
            outputs=[preset_info, history_state, system_message_box, profile_image]
        ).then(
            fn=main_tab.filter_messages_for_chatbot,
            inputs=[history_state],
            outputs=chatbot
        )