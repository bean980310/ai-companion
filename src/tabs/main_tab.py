import logging
import gradio as gr
import os
import secrets
import sqlite3

from src.models.models import get_all_local_models, generate_answer
from src.common.database import save_chat_history_db, delete_session_history, delete_all_sessions, get_preset_choices, load_system_presets, get_existing_sessions, load_chat_from_db, update_system_message_in_db
from src.common.translations import TranslationManager, translation_manager

from src.characters.preset_images import PRESET_IMAGES
from src.models.api_models import api_models
from src.models.local_models import transformers_local, gguf_local, mlx_local
from src.common.default_language import default_language

import traceback
from src.characters.persona_speech_manager import PersonaSpeechManager
from src.common.args import parse_args

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
generator_choices = api_models + transformers_local + gguf_local + mlx_local + ["사용자 지정 모델 경로 변경"]
generator_choices = list(dict.fromkeys(generator_choices))  # 중복 제거
generator_choices = sorted(generator_choices)  # 정렬

DEFAULT_PROFILE_IMAGE = None

characters={
    "AI 비서": {
        "default_tone": "존댓말",
        "languages": ["ko", "ja", "zh_CN", "zh_TW", "en"],
        "preset_name": "AI_ASSISTANT_PRESET",
        "profile_image": "assets/0_ai_assistant.png"
    },
    "미나미 아스카": {
        "default_tone": "반말", 
        "languages": ["ko", "ja", "zh_CN", "zh_TW", "en"],
        "preset_name": "MINAMI_ASUKA_PRESET",
        "profile_image": "assets/1_minami_asuka.png"
    },
    "마코토노 아오이": {
        "default_tone": "반말", 
        "languages": ["ko", "ja", "zh_CN", "zh_TW", "en"],
        "preset_name": "MAKOTONO_AOI_PRESET",
        "profile_image": "assets/2_makotono_aoi.png"
    },
    "아이노 코이토": {"default_tone": "반말", 
        "languages": ["ko", "ja", "zh_CN", "zh_TW", "en"],
        "preset_name": "AINO_KOITO_PRESET",
        "profile_image": "assets/3_aino_koito.png"
    },
}

speech_manager = PersonaSpeechManager(translation_manager=translation_manager, characters=characters)

session_speech_managers = {}

def get_speech_manager(session_id: str) -> PersonaSpeechManager:
    if session_id not in session_speech_managers:
        session_speech_managers[session_id] = PersonaSpeechManager(translation_manager=translation_manager, characters=characters)
    return session_speech_managers[session_id]

class MainTab:
    def __init__(self):
        self.default_language=default_language
        self.preset_images=PRESET_IMAGES
        self.default_profile_image=DEFAULT_PROFILE_IMAGE
        self.characters=characters
        self.reset_type = None
        
    def handle_change_preset(self, new_preset_name, history, language):
        """
        프리셋을 변경하고, 새로운 시스템 메시지를 히스토리에 추가하며, 프로필 이미지를 변경합니다.

        Args:
            new_preset_name (str): 선택된 새로운 프리셋의 이름.
            history (list): 현재 대화 히스토리.
            language (str): 현재 선택된 언어.

        Returns:
            tuple: 업데이트된 대화 히스토리, 새로운 프로필 이미지 경로.
        """
        # 새로운 프리셋 내용 로드
        presets = load_system_presets(language=language)
        
        if new_preset_name not in presets:
            logger.warning(f"선택한 프리셋 '{new_preset_name}'이 존재하지 않습니다.")
            return history, self.default_profile_image  # 프리셋이 없을 경우 기본 이미지 반환

        new_system_message = {
            "role": "system",
            "content": presets[new_preset_name]
        }
        content = presets.get(new_preset_name, "")

        # 기존 히스토리에 새로운 시스템 메시지 추가
        history.append(new_system_message)
        logger.info(f"프리셋 '{new_preset_name}'로 변경되었습니다.")

        # 프로필 이미지 변경
        image_path = self.preset_images.get(new_preset_name)
        
        if image_path and os.path.isfile(image_path):
            return history, gr.update(value=content), image_path
        else:
            return history, gr.update(value=content), None

    def process_message(self, user_input, session_id, history, system_msg, selected_model, custom_path, image, api_key, device, seed, language, selected_character):
        """
        사용자 메시지를 처리하고 봇 응답을 생성하는 통합 함수.

        Args:
            user_input (str): 사용자가 입력한 메시지.
            session_id (str): 현재 세션 ID.
            history (list): 채팅 히스토리.
            system_msg (str): 시스템 메시지.
            selected_model (str): 선택된 모델 이름.
            custom_path (str): 사용자 지정 모델 경로.
            image (PIL.Image or None): 이미지 입력 (비전 모델용).
            api_key (str or None): API 키 (API 모델용).
            device (str): 사용할 장치 ('cpu', 'cuda', 등).
            seed (int): 시드 값.

        Returns:
            tuple: 업데이트된 입력 필드, 히스토리, Chatbot 컴포넌트, 상태 메시지.
        """
        if not user_input.strip():
            # 빈 입력일 경우 아무 것도 하지 않음
            return "", history, self.filter_messages_for_chatbot(history), ""

        if selected_character and selected_character not in self.characters:
            logger.warning(f"Invalid character selected: {selected_character}")
            selected_character = None
            
        if not history:
            # 히스토리가 없을 경우 시스템 메시지로 초기화
            system_message = {
                "role": "system",
                "content": system_msg
            }
            history = [system_message]

        speech_manager = get_speech_manager(session_id)
        
        try:
            speech_manager.set_character_and_language(selected_character, language)
        except ValueError as e:
            tb = traceback.format_exc()
            logger.error(f"캐릭터 설정 오류: {str(e)}\n{tb}")
            history.append({"role": "assistant", "content": f"❌ 캐릭터 설정 중 오류가 발생했습니다."})
            return "", history, self.filter_messages_for_chatbot(history), "❌ 캐릭터 설정 오류"
    
        
        # 사용자 메시지 추가
        history.append({"role": "user", "content": user_input})
        
        speech_manager.update_tone(user_input)

        try:
            # 봇 응답 생성
            answer = generate_answer(
                history=history,
                selected_model=selected_model,
                model_type=self.determine_model_type(selected_model),
                local_model_path=custom_path if selected_model == "사용자 지정 모델 경로 변경" else None,
                image_input=image,  # image 인자 전달
                api_key=api_key,
                device=device,
                seed=seed,
                character_language=language
            )

            styled_answer = speech_manager.generate_response(answer)
            
            # 응답을 히스토리에 추가
            history.append({"role": "assistant", "content": styled_answer})

            # 데이터베이스에 히스토리 저장
            save_chat_history_db(history, session_id=session_id)

            # 상태 메시지 초기화
            status = ""

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            history.append({"role": "assistant", "content": f"❌ 오류 발생: {str(e)}"})
            status = "❌ 오류가 발생했습니다. 로그를 확인하세요."

        # 업데이트된 히스토리를 Chatbot 형식으로 변환
        chatbot_history = self.filter_messages_for_chatbot(history)

        return "", history, chatbot_history, status
    
    def determine_model_type(self, selected_model):
        if selected_model in api_models:
            return "api"
        elif selected_model in transformers_local:
            return "transformers"
        elif selected_model in gguf_local:
            return "gguf"
        elif selected_model in mlx_local:
            return "mlx"
        else:
            return "transformers"
    

    def filter_messages_for_chatbot(self, history):
        """
        채팅 히스토리를 Gradio Chatbot 컴포넌트에 맞는 형식으로 변환

        Args:
            history (list): 전체 채팅 히스토리

        Returns:
            list: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        messages_for_chatbot = []
        for msg in history:
            if msg["role"] in ("user", "assistant"):
                content = msg["content"] or ""
                character = msg.get("assistant", "")  # 오타 수정
                if character:
                    display_content = f"**{character}:** {content}"
                else:
                    display_content = content
                messages_for_chatbot.append({"role": msg["role"], "content": display_content})
        return messages_for_chatbot

    def reset_session(self, history, chatbot, system_message_default, language=None, session_id="demo_session"):
        """
        특정 세션을 초기화하는 함수.
        
        Returns:
            tuple: (reset_modal, single_content, all_content, msg, new_history, chatbot_history, status)
        """
        if language is None:
            language = self.default_language

        try:
            success = delete_session_history(session_id)
            if not success:
                return (
                    gr.update(visible=False),  # reset_modal
                    gr.update(visible=False),  # single_content
                    gr.update(visible=False),  # all_content
                    gr.update(),               # msg
                    history,                   # history 유지
                    chatbot,                   # chatbot 유지
                    "❌ 세션 초기화에 실패했습니다."  # status
                )

            # 새로운 시스템 메시지로 히스토리 초기화
            default_system = {
                "role": "system",
                "content": system_message_default
            }
            new_history = [default_system]

            # 새 히스토리 저장
            save_chat_history_db(new_history, session_id=session_id)
            
            # chatbot 컴포넌트를 위한 메시지 필터링
            chatbot_history = self.filter_messages_for_chatbot(new_history)

            return (
                gr.update(visible=False),  # reset_modal
                gr.update(visible=False),  # single_content
                gr.update(visible=False),  # all_content
                "",                        # msg
                new_history,              # new_history
                chatbot_history,          # filtered chatbot messages
                "✅ 세션이 초기화되었습니다."  # status
            )

        except Exception as e:
            logger.error(f"Error resetting session: {str(e)}")
            return (
                gr.update(visible=False),  # reset_modal
                gr.update(visible=False),  # single_content
                gr.update(visible=False),  # all_content
                "",                        # msg
                history,                   # history 유지
                chatbot,                   # chatbot 유지
                f"❌ 세션 초기화 중 오류가 발생했습니다: {str(e)}"  # status
            )

    def reset_all_sessions(self, history, chatbot, system_message_default, language=None):
        """
        모든 세션을 초기화하는 함수.
        
        Returns:
            tuple: (reset_modal, single_content, all_content, msg, new_history, chatbot_history, status)
        """
        if language is None:
            language = self.default_language

        try:
            success = delete_all_sessions()
            if not success:
                return (
                    gr.update(visible=False),  # reset_modal
                    gr.update(visible=False),  # single_content
                    gr.update(visible=False),  # all_content
                    gr.update(),               # msg
                    history,                   # history 유지
                    chatbot,                   # chatbot 유지
                    "❌ 모든 세션 초기화에 실패했습니다."  # status
                )

            # 새로운 시스템 메시지로 히스토리 초기화
            default_system = {
                "role": "system",
                "content": system_message_default
            }
            new_history = [default_system]

            # demo_session에 대해 새 히스토리 저장
            save_chat_history_db(new_history, session_id="demo_session")
            
            # chatbot 컴포넌트를 위한 메시지 필터링
            chatbot_history = self.filter_messages_for_chatbot(new_history)

            return (
                gr.update(visible=False),  # reset_modal
                gr.update(visible=False),  # single_content
                gr.update(visible=False),  # all_content
                "",                        # msg
                new_history,              # new_history
                chatbot_history,          # filtered chatbot messages
                "✅ 모든 세션이 초기화되었습니다."  # status
            )

        except Exception as e:
            logger.error(f"Error resetting all sessions: {str(e)}")
            return (
                gr.update(visible=False),  # reset_modal
                gr.update(visible=False),  # single_content
                gr.update(visible=False),  # all_content
                "",                        # msg
                history,                   # history 유지
                chatbot,                   # chatbot 유지
                f"❌ 모든 세션 초기화 중 오류가 발생했습니다: {str(e)}"  # status
            )

    def refresh_preset_list(self, language=None):
        """프리셋 목록을 갱신하는 함수."""
        if language is None:
            language = self.default_language
        presets = get_preset_choices(language)
        return gr.update(choices=presets, value=presets[0] if presets else None)
    
    def refresh_sessions(self):
        """
        세션 목록을 갱신하고, (Dropdown) choices를 반환합니다.
        """
        sessions = get_existing_sessions()
        if not sessions:
            return gr.update(choices=[], value=None), "DB에 세션이 없습니다."
        return gr.update(choices=sessions, value=sessions[0])

    def create_new_session(self, system_message_box_value: str):
        """
        새 세션을 생성하고 DB에 기본 system_message를 저장합니다.
        """
        new_sid = secrets.token_hex(8)
        system_message = {
            "role": "system",
            "content": system_message_box_value
        }
        
        new_history = [system_message]
        # DB에 저장
        save_chat_history_db(new_history, session_id=new_sid)
        return new_sid, f"현재 세션: {new_sid}", new_history

    def apply_session(self, chosen_sid: str):
        """
        선택된 세션의 히스토리를 불러오고, session_id_state를 갱신.
        """
        if not chosen_sid:
            return [], None, "세션 ID를 선택하세요."
        loaded_history = load_chat_from_db(chosen_sid)
        # last_activity 갱신
        with sqlite3.connect("chat_history.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions
                SET last_activity = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (chosen_sid,))
            conn.commit()
            
        return loaded_history, chosen_sid, f"세션 {chosen_sid}이 적용되었습니다."

    def delete_session(self, chosen_sid: str, current_sid: str):
        """
        특정 세션 삭제 로직
        
        Returns:
            tuple: (modal_visible, modal_message, session_dropdown_update)
        """
        if not chosen_sid:
            return (
                gr.update(visible=True),  # modal visible
                "삭제할 세션을 선택하세요.",  # error message
                gr.update()  # no dropdown update
            )
        
        if chosen_sid == current_sid:
            return (
                gr.update(visible=True),  # modal visible
                f"현재 활성 세션 '{chosen_sid}'은(는) 삭제할 수 없습니다.",  # error message
                gr.update()  # no dropdown update
            )
            
        try:
            conn = sqlite3.connect("chat_history.db")
            c = conn.cursor()
            c.execute("DELETE FROM chat_history WHERE session_id = ?", (chosen_sid,))
            conn.commit()
            conn.close()

            sessions = get_existing_sessions()
            return (
                gr.update(visible=False),  # hide modal
                f"세션 '{chosen_sid}'이(가) 삭제되었습니다.",  # success message
                gr.update(choices=sessions, value=sessions[0] if sessions else None)  # update dropdown
            )
        except Exception as e:
            logger.error(f"세션 삭제 오류: {e}")
            return (
                gr.update(visible=True),  # keep modal visible
                f"세션 삭제 실패: {e}",  # error message
                gr.update()  # no dropdown update
            )

    def initial_load_presets(self, language=None):
        """초기 프리셋 로딩 함수"""
        if language is None:
            language = self.default_language
        presets = get_preset_choices(language)
        return gr.update(choices=presets, value=presets[0] if presets else None)


    def process_character_conversation(self, history, selected_characters, model_type, selected_model, custom_path, image, api_key, device, seed):
        try:
            for i, character in enumerate(selected_characters):
                # 각 캐릭터의 시스템 메시지 설정
                system_message = {
                    "role": "system",
                    "content": translation_manager.get_character_setting(character)
                }
                history.append(system_message)
                
                # 캐릭터의 응답 생성
                answer = generate_answer(
                    history=history,
                    selected_model=selected_model,
                    model_type=model_type,
                    local_model_path=custom_path if selected_model == "사용자 지정 모델 경로 변경" else None,
                    image_input=image,
                    api_key=api_key,
                    device=device,
                    seed=seed
                )
                
                history.append({
                    "role": "assistant",
                    "content": answer,
                    "character": character
                })
            
            # 데이터베이스에 히스토리 저장
            save_chat_history_db(history, session_id="character_conversation")
            
            # 프로필 이미지는 None으로 반환
            return history, None  # 여기서 None을 반환하도록 수정

        except Exception as e:
            logger.error(f"Error generating character conversation: {str(e)}", exc_info=True)
            history.append({"role": "assistant", "content": f"❌ 오류 발생: {str(e)}", "character": "System"})
            return history, None  # 오류 발생시에도 None 반환
    
    def toggle_api_key_visibility(self, selected_model):
        """
        OpenAI API Key 입력 필드의 가시성을 제어합니다.
        """
        api_visible = selected_model in api_models
        return gr.update(visible=api_visible)

    def toggle_image_input_visibility(self, selected_model):
        """
        이미지 입력 필드의 가시성을 제어합니다.
        """
        image_visible = (
            "vision" in selected_model.lower() or
            "qwen2-vl" in selected_model.lower() or
            selected_model in [
                "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B",
                "THUDM/glm-4v-9b",
                "openbmb/MiniCPM-Llama3-V-2_5"
            ]
        )
        return gr.update(visible=image_visible)

    def update_model_list(self, selected_type):
        local_models_data = get_all_local_models()
        transformers_local = local_models_data["transformers"]
        gguf_local = local_models_data["gguf"]
        mlx_local = local_models_data["mlx"]
                
        # "전체 목록"이면 => API 모델 + 모든 로컬 모델 + "사용자 지정 모델 경로 변경"
        if selected_type == "all":
            all_models = api_models + transformers_local + gguf_local + mlx_local
            # 중복 제거 후 정렬
            all_models = sorted(list(dict.fromkeys(all_models)))
            return gr.update(choices=all_models, value=all_models[0] if all_models else None)
        
        # API 모델만 선택한 경우
        if selected_type == "api":
            updated_list = api_models
        # 개별 로컬 모델 유형 선택
        elif selected_type == "transformers":
            updated_list = transformers_local
        elif selected_type == "gguf":
            updated_list = gguf_local
        elif selected_type == "mlx":
            updated_list = mlx_local
        else:
        # 혹시 예상치 못한 값이면 transformers로 처리(또는 None)
            updated_list = transformers_local
                
        updated_list = sorted(list(dict.fromkeys(updated_list)))
        return gr.update(choices=updated_list, value=updated_list[0] if updated_list else None)
    
    def show_reset_modal(self, reset_type):
        """초기화 확인 모달 표시"""
        self.reset_type = reset_type
        return (
            gr.update(visible=True),  # modal
            gr.update(visible=reset_type == "single"),  # single_content
            gr.update(visible=reset_type == "all"),  # all_content
        )

    def hide_reset_modal(self):
        """초기화 확인 모달 숨김"""
        return (
            gr.update(visible=False),  # modal
            gr.update(visible=False),  # single_content
            gr.update(visible=False),  # all_content
        )

    def handle_reset_confirm(self, history, chatbot, system_msg, language=None, session_id="demo_session"):
        """초기화 확인 시 처리"""
        try:
            if self.reset_type == "single":
                result = self.reset_session(
                    history=history,
                    chatbot=self.filter_messages_for_chatbot(history),  # 현재 채팅 상태 전달
                    system_message_default=system_msg,
                    language=language,
                    session_id=session_id
                )
            else:  # reset_type == "all"
                result = self.reset_all_sessions(
                    history=history,
                    chatbot=self.filter_messages_for_chatbot(history),  # 현재 채팅 상태 전달
                    system_message_default=system_msg,
                    language=language
                )
            
            # 모달 업데이트와 결과를 함께 반환
            return result
            
        except Exception as e:
            logger.error(f"Reset confirmation error: {str(e)}")
            # 오류 발생 시 현재 상태 유지
            return (
                gr.update(visible=False),  # reset_modal
                gr.update(visible=False),  # single_content
                gr.update(visible=False),  # all_content
                "",                        # msg
                history,                   # 현재 history 유지
                self.filter_messages_for_chatbot(history),  # 현재 chatbot 상태 유지
                f"❌ 초기화 중 오류가 발생했습니다: {str(e)}"  # status
            )
    
def update_system_message_and_profile(
    character_name: str, 
    language_display_name: str, 
    speech_manager: PersonaSpeechManager,
    session_id: str
):
    """
    캐릭터와 언어 선택 시 호출되는 함수.
    - 캐릭터와 언어 설정 적용
    - 시스템 메시지 프리셋 업데이트
    - DB에 system 메시지를 저장/갱신
    """
    try:
        language_code = translation_manager.get_language_code(language_display_name)
        speech_manager.set_character_and_language(character_name, language_code)

        # 실제 프리셋 로딩은 speech_manager 내부에서 처리
        system_message = speech_manager.get_system_message()
        selected_profile_image = speech_manager.characters[character_name]["profile_image"]
        
        # -- DB 업데이트 로직 추가 --
        # session_id가 유효하다면, 새 시스템 메시지를 DB에 반영
        if session_id:
            update_system_message_in_db(session_id, system_message)

        return system_message, selected_profile_image
    except ValueError as ve:
        logger.error(f"Character setting error: {ve}")
        return "시스템 메시지 로딩 중 오류가 발생했습니다.", None
    
def create_reset_confirm_modal():
    """초기화 확인 모달 생성"""
    with gr.Column(visible=False, elem_classes="reset-confirm-modal") as reset_modal:
        gr.Markdown("# ⚠️ 확인", elem_classes="reset-confirm-title")
        with gr.Column() as single_reset_content:
            gr.Markdown("현재 세션의 모든 대화 내용이 삭제됩니다. 계속하시겠습니까?", 
                       elem_classes="reset-confirm-message")
        with gr.Column(visible=False) as all_reset_content:
            gr.Markdown("모든 세션의 대화 내용이 삭제됩니다. 계속하시겠습니까?", 
                       elem_classes="reset-confirm-message")
        with gr.Row(elem_classes="reset-confirm-buttons"):
            cancel_btn = gr.Button("취소", variant="secondary")
            confirm_btn = gr.Button("확인", variant="primary")
            
    return (reset_modal, single_reset_content, all_reset_content, 
            cancel_btn, confirm_btn)
    
def create_delete_session_modal():
    """삭제 확인 모달 생성"""
    with gr.Column(visible=False, elem_classes="delete-session-modal") as delete_modal:
        gr.Markdown("# ⚠️ 세션 삭제 확인", elem_classes="delete-session-title")
        message = gr.Markdown("", elem_classes="delete-session-message")
        with gr.Row(elem_classes="delete-session-buttons"):
            cancel_btn = gr.Button("취소", variant="secondary")
            confirm_btn = gr.Button("삭제", variant="stop")
                
    return delete_modal, message, cancel_btn, confirm_btn
    