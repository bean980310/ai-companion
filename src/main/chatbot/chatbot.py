# import logging
from typing import Any, Generator, Literal
import gradio as gr
# from gradio_i18n import gettext as _, translate_blocks
import os
import secrets
import sqlite3

from src.models.models import get_all_local_models, generate_answer, generate_chat_title
from src.common.database import save_chat_history_db, delete_session_history, delete_all_sessions, get_preset_choices, load_system_presets, get_existing_sessions, get_existing_sessions_with_names, update_session_name, load_chat_from_db, update_system_message_in_db, update_last_character_in_db
from src.common.translations import TranslationManager, translation_manager, _

from src.characters.preset_images import PRESET_IMAGES
from src.models import llm_api_models, openai_llm_api_models, anthropic_llm_api_models, google_genai_llm_api_models, perplexity_llm_api_models, xai_llm_api_models, mistralai_llm_api_models, openrouter_llm_api_models, huggingface_inference_llm_api_models, ollama_llm_models, lmstudio_llm_models, oobabooga_llm_models, vllm_llm_api_models,REASONING_CONTROLABLE, REASONING_KWD, REASONING_BAN, transformers_local, vllm_local, gguf_local, mlx_local
from src.common.default_language import default_language
from src.common.utils import detect_platform

import traceback
from src.characters.persona_speech_manager import PersonaSpeechManager
from src.common.character_info import characters
from src.common.args import parse_args
# from src.common.translations import _
from ...start_app import ui_component
# from translations import i18n as _

import requests
import base64
from io import BytesIO
from PIL import Image, ImageOps, ImageFile
import pandas as pd

from src import logger

# 로깅 설정

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

DEFAULT_PROFILE_IMAGE = None

# os_name, arch = detect_platform()

# speech_manager = PersonaSpeechManager(translation_manager=translation_manager, characters=characters)

class Chatbot:
    def __init__(self):
        self.default_language=default_language
        self.preset_images=PRESET_IMAGES
        self.default_profile_image=DEFAULT_PROFILE_IMAGE
        self.characters=characters
        self.reset_type = None
        self.chat_titles={}

        self.vision_model = False

        self.os_name, self.arch = detect_platform()
        self.session_speech_managers = {}

    def get_speech_manager(self, session_id: str) -> PersonaSpeechManager:
        if session_id not in self.session_speech_managers:
            self.session_speech_managers[session_id] = PersonaSpeechManager(translation_manager=translation_manager, characters=self.characters)
        return self.session_speech_managers[session_id]
    
    def update_system_message_and_profile(
        self,
        character_name: str, 
        language_display_name: str, 
        session_id: str
    ):
        """
        캐릭터와 언어 선택 시 호출되는 함수.
        - 캐릭터와 언어 설정 적용
        - 시스템 메시지 프리셋 업데이트
        - DB에 system 메시지를 저장/갱신
        """
        try:
            speech_manager = self.get_speech_manager(session_id)
            language_code = translation_manager.get_language_code(language_display_name)
            speech_manager.set_character_and_language(character_name, language_code)

            # 실제 프리셋 로딩은 speech_manager 내부에서 처리
            system_message = speech_manager.get_system_message()
            selected_profile_image = speech_manager.characters[character_name]["profile_image"]
            
            # -- DB 업데이트 로직 추가 --
            # session_id가 유효하다면, 새 시스템 메시지를 DB에 반영
            if session_id:
                update_system_message_in_db(session_id, system_message)
                update_last_character_in_db(session_id, character_name)

            return system_message, selected_profile_image, gr.update(value=character_name), gr.update(avatar_images=[None, selected_profile_image])
        except ValueError as ve:
            logger.error(f"Character setting error: {ve}")
            return "시스템 메시지 로딩 중 오류가 발생했습니다.", None, gr.update(), gr.update()
        
    def handle_change_preset(self, new_preset_name: str, history: list[dict[str, str | Any]], language: str):
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
            return history, gr.update(value=content), image_path, gr.update(avatar_images=[None, image_path])
        else:
            return history, gr.update(value=content), None, gr.update()

    def process_message_user(self, user_input: gr.Component | dict[str, str | Image.Image | Any], session_id: str, history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]] | list[gr.MessageDict | gr.ChatMessage], system_msg: str, selected_character: str, language: str):
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
        text = str(user_input.get("text", ""))
        files = str(user_input.get("files", ""))
        if not text.strip() and not files:
            # 빈 입력일 경우 아무 것도 하지 않음
            return "", history, self.filter_messages_for_chatbot(history), ""
        else:
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

        speech_manager = self.get_speech_manager(session_id)
        try:
            speech_manager.set_character_and_language(selected_character, language)
        except ValueError as e:
            tb = traceback.format_exc()
            logger.error(f"캐릭터 설정 오류: {str(e)}\n{tb}")
            history.append({"role": "assistant", "content": "❌ 캐릭터 설정 중 오류가 발생했습니다."})
            return "", history, self.filter_messages_for_chatbot(history), "❌ 캐릭터 설정 오류"

        user_contents = [{"type": "text", "text": text}]
        if files:
            if files.rsplit('.')[-1] in ["jpg", "jpeg", "png", "webp", "gif"]:
                with open(files, "rb") as f:
                    if files.rsplit('.')[-1] == "jpg" or "jpeg":
                        mime_type="image/jpeg"
                    elif files.rsplit('.')[-1] == "png":
                        mime_type="image/png"
                    elif files.rsplit('.')[-1] == "webp":
                        mime_type="image/webp"
                    elif files.rsplit('.')[-1] == "gif":
                        mime_type="image/gif"
                    image = base64.b64encode(f.read()).decode('utf-8')
                user_contents.append({"type": "image", "image_url": f"data:{mime_type};base64,{image}"})
            # elif files.rsplit('.')[-1] == "txt":
                
        new_message = {
            "role": "user",
            "content": user_contents,
        }

        history.append(new_message)
        speech_manager.update_tone(text)
        
        return "", history, self.filter_messages_for_chatbot(history)

    def process_message_bot(self, session_id: str, history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]], selected_model: str | gr.Dropdown, provider: Literal["openai", "anthropic", "google-genai", "perplexity", "xai", "mistralai", "openrouter", "hf-inference", "ollama", "lmstudio", "oobabooga", "self-provided"] | gr.Dropdown, selected_lora: str | gr.Dropdown, custom_path: str, user_input: str | dict[str, str | Image.Image | Any] | Any, api_key: str, device: str, seed: int, max_length: int, temperature: float, top_k: int, top_p: float, repetition_penalty: float, enable_thinking: bool, language: str):
        image = None
        if isinstance(user_input, dict):
            files = user_input.get("files", [])
            if isinstance(files, (list, dict)):
                image = []
                for f in files:
                    image.append(files[f])
            else:
                image = files
            
        chat_title=self.chat_titles.get(session_id)

        if provider == "self-provided":
            model_type=self.determine_model_type(selected_model)
        else:
            model_type=None
        try:
            # 봇 응답 생성
            answer = generate_answer(
                history=history,
                selected_model=selected_model,
                provider=provider,
                model_type=model_type,
                selected_lora=selected_lora if selected_lora != "None" else None,
                local_model_path=custom_path if selected_model == "사용자 지정 모델 경로 변경" else None,
                lora_path=None,
                image_input=image,  # image 인자 전달
                api_key=api_key,
                device=device,
                seed=seed,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                enable_thinking=enable_thinking,
                character_language=language
            )
            
            speech_manager = self.get_speech_manager(session_id)
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
        
        if chat_title is None and len(history)==2:
            chat_title=generate_chat_title(
                first_message=history[1]["content"],
                selected_model=selected_model,
                model_type=model_type,
                provider=provider,
                selected_lora=selected_lora if selected_lora != "None" else None,
                local_model_path=custom_path if selected_model == "사용자 지정 모델 경로 변경" else None,
                lora_path=None,
                device=device,
                image_input=image,
            )

        return history, chatbot_history, status, chat_title

    def chat_wrapper(self, message: str | list[dict[str, str | Image.Image | Any]] | Any, history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]], session_id: str, system_msg: str | gr.Textbox, selected_character: str, language: str,
                     selected_model: str, provider: Literal["openai", "anthropic", "google-genai", "perplexity", "xai", "mistralai", "openrouter", "hf-inference", "ollama", "lmstudio", "oobabooga", "self-provided"], selected_lora: str, custom_path: str, api_key: str, device: str, seed: int,
                     max_length: int, temperature: float, top_k: int, top_p: float, repetition_penalty: float, enable_thinking: bool,
                     is_temp_session=False):
        """
        gr.ChatInterface를 위한 래퍼 함수
        """
        # 1. 사용자 메시지 처리
        # history는 gr.ChatInterface에서 관리하는 list[dict] (type="messages"일 경우)
        # 하지만 우리는 내부 DB와 app_state.history_state를 동기화해야 함.

        # 기존 history에 사용자 메시지 추가
        # message는 str 또는 dict(text=..., files=...)

        # 시스템 메시지가 없으면 추가 (새 세션인 경우)
        if not history:
            history.append({"role": "system", "content": system_msg})

        current_history = history.copy() if history else []

        speech_manager = self.get_speech_manager(session_id)
        try:
            speech_manager.set_character_and_language(selected_character, language)
            
        except ValueError as e:
            tb = traceback.format_exc()
            logger.error(f"캐릭터 설정 오류: {str(e)}\n{tb}")
            history.append({"role": "assistant", "content": "❌ 캐릭터 설정 중 오류가 발생했습니다."})
            return "", history, self.filter_messages_for_chatbot(history), "❌ 캐릭터 설정 오류"

        # 임시 세션 처리: 첫 메시지 전송 시 실제 세션으로 변환
        new_session_id = session_id
        is_temp_after = is_temp_session  # 처리 후 임시 세션 상태

        if is_temp_session and message:
            # 첫 메시지: 임시 세션을 실제 세션으로 변환
            user_content = message.get("text", "")

            # Determine model type
            if provider == "self-provided":
                model_type = self.determine_model_type(selected_model)
            else:
                model_type = None

            new_session_id, title = self.confirm_temp_session(
                temp_history=current_history,
                first_message_content=user_content,
                selected_character=selected_character,
                selected_model=selected_model,
                model_type=model_type,
                provider=provider,
                selected_lora=selected_lora if selected_lora != "None" else None,
                custom_path=custom_path,
                device=device,
                image_input=None
            )
            is_temp_after = False  # 더 이상 임시 세션이 아님
            session_id = new_session_id
            self.chat_titles[session_id] = title
        
        # 사용자 메시지 구성
        user_content = message.get("text", "")
        user_files = message.get("files", [])

        content_list = [{"type": "text", "text": user_content}]
        if user_files:
            # 멀티모달 메시지 처리
            for file_path in user_files:
                with open(file_path, "rb") as f:
                    if file_path.rsplit('.')[-1] == "jpg" or "jpeg":
                        mime_type="image/jpeg"
                    elif file_path.rsplit('.')[-1] == "png":
                        mime_type="image/png"
                    elif file_path.rsplit('.')[-1] == "webp":
                        mime_type="image/webp"
                    elif file_path.rsplit('.')[-1] == "gif":
                        mime_type="image/gif"
                    image = base64.b64encode(f.read()).decode('utf-8')

                    image_url = f"data:{mime_type};base64,{image}"
                # 이미지 파일을 base64로 변환하거나 경로를 사용
                # 여기서는 기존 process_message_user 로직을 참고하여 처리
                # 다만 ChatInterface는 로컬 경로를 넘겨줌
                content_list.append({"type": "image", "image_url": image_url}) 
                
        current_history.append({"role": "user", "content": content_list})


        # 2. 봇 응답 생성
        # process_message_bot 로직 활용
        
        # process_message_bot은 history를 인자로 받아 봇 응답을 추가하고 반환함
        # user_input은 이미 history에 추가했으므로 process_message_bot 호출 시 user_input 인자는 
        # generate_answer 내부에서 이미지 처리를 위해 필요할 수 있음.
        
        # generate_answer를 직접 호출하는 것이 더 깔끔할 수 있음.
        
        # 이미지 입력 준비
        image_input = None
        if user_files:
             image_input = user_files # 리스트 전달
        
        chat_title = self.chat_titles.get(session_id)
        
        if provider == "self-provided":
            model_type = self.determine_model_type(selected_model)
        else:
            model_type = None
        
        try:
            answer = generate_answer(
                history=current_history,
                selected_model=selected_model,
                provider=provider,
                model_type=model_type,
                selected_lora=selected_lora if selected_lora != "None" else None,
                local_model_path=custom_path if selected_model == "사용자 지정 모델 경로 변경" else None,
                lora_path=None,
                image_input=image_input,
                api_key=api_key,
                device=device,
                seed=seed,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                enable_thinking=enable_thinking,
                character_language=language
            )
            
            styled_answer = speech_manager.generate_response(answer)
            
            # 응답을 히스토리에 추가
            # gr.ChatInterface는 리턴된 문자열을 봇의 응답으로 처리하여 히스토리에 추가함
            # 하지만 우리는 DB 저장을 위해 전체 히스토리를 업데이트해야 함
            current_history.append({"role": "assistant", "content": styled_answer})
            
            # 데이터베이스에 히스토리 저장
            save_chat_history_db(current_history, session_id=session_id)
            
            status = ""
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            styled_answer = f"❌ 오류 발생: {str(e)}"
            current_history.append({"role": "assistant", "content": styled_answer})
            status = "❌ 오류가 발생했습니다."

        # 채팅 제목 생성 (필요 시)
        if chat_title is None and len(current_history) >= 2: # 시스템 메시지 포함 최소 2개 이상일 때
             # 첫 번째 사용자 메시지 찾기
            first_user_msg = next((msg for msg in current_history if msg["role"] == "user"), None)
            if first_user_msg:
                # Determine model type for self-provided models
                if provider == "self-provided":
                    model_type = self.determine_model_type(selected_model)
                else:
                    model_type = None

                chat_title = generate_chat_title(
                    first_message=first_user_msg["content"],
                    selected_model=selected_model,
                    model_type=model_type,
                    provider=provider,
                    selected_lora=selected_lora if selected_lora != "None" else None,
                    local_model_path=custom_path if selected_model == "사용자 지정 모델 경로 변경" else None,
                    lora_path=None,
                    device=device,
                    image_input=image_input,
                )
                self.chat_titles[session_id] = chat_title

        # Return values for ChatInterface
        # 1. response (str)
        # 2. Additional outputs: app_state.history_state, status_text, chat_title_box, session_id, is_temp_session

        return (
            styled_answer,
            current_history,
            status,
            gr.update(value=chat_title) if chat_title else gr.update(),
            session_id,  # Return the (possibly new) session_id
            is_temp_after,  # Return the updated is_temp_session state
        )

    @staticmethod
    def determine_model_type(selected_model: str) -> str:
        if selected_model in transformers_local:
            return "transformers"
        elif selected_model in gguf_local:
            return "gguf"
        elif selected_model in mlx_local:
            return "mlx"
        else:
            return "transformers"
    
    @staticmethod
    def filter_messages_for_chatbot(history: list[dict[str, str | Any]]) -> list[dict[str, str | Any]]:
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

    def reset_session(self, history: list[dict[str, str | Any]], chatbot: list[dict[str, str | Any]], system_message_default: str, selected_character: str, language: str | None = None, session_id: str = "demo_session"):
        """
        특정 세션을 초기화하는 함수.

        Returns:
            tuple: (reset_modal, single_content, all_content, msg, new_history, chatbot_history, status, session_dropdown)
        """
        if language is None:
            language = self.default_language

        try:
            success = delete_session_history(session_id)
            if not success:
                sessions = get_existing_sessions()
                return (
                    gr.update(visible=False),  # reset_modal
                    gr.update(visible=False),  # single_content
                    gr.update(visible=False),  # all_content
                    gr.update(),               # msg
                    history,                   # history 유지
                    chatbot,                   # chatbot 유지
                    "❌ 세션 초기화에 실패했습니다.",  # status
                    gr.update(choices=sessions, value=session_id),  # session_dropdown
                )

            # 새로운 시스템 메시지로 히스토리 초기화
            default_system = {
                "role": "system",
                "content": system_message_default
            }
            new_history = [default_system]

            # 새 히스토리 저장 (선택된 캐릭터도 함께 저장)
            save_chat_history_db(new_history, session_id=session_id, selected_character=selected_character)

            # chatbot 컴포넌트를 위한 메시지 필터링
            chatbot_history = self.filter_messages_for_chatbot(new_history)

            sessions = get_existing_sessions()
            return (
                gr.update(visible=False),  # reset_modal
                gr.update(visible=False),  # single_content
                gr.update(visible=False),  # all_content
                "",                        # msg
                new_history,               # new_history
                chatbot_history,           # filtered chatbot messages
                "✅ 세션이 초기화되었습니다.",  # status
                gr.update(choices=sessions, value=session_id),  # session_dropdown
            )

        except Exception as e:
            logger.error(f"Error resetting session: {str(e)}")
            sessions = get_existing_sessions()
            return (
                gr.update(visible=False),  # reset_modal
                gr.update(visible=False),  # single_content
                gr.update(visible=False),  # all_content
                "",                        # msg
                history,                   # history 유지
                chatbot,                   # chatbot 유지
                f"❌ 세션 초기화 중 오류가 발생했습니다: {str(e)}",  # status
                gr.update(choices=sessions, value=session_id),  # session_dropdown
            )

    def reset_all_sessions(self, history: list[dict[str, str | Any]], chatbot: list[dict[str, str | Any]], system_message_default: str, selected_character: str, language: str | None = None):
        """
        모든 세션을 초기화하는 함수.
        demo_session을 제외한 모든 세션을 삭제하고, demo_session을 초기 상태로 되돌립니다.

        Returns:
            tuple: (reset_modal, single_content, all_content, msg, new_history, chatbot_history, status, session_dropdown, session_id)
        """
        if language is None:
            language = self.default_language

        # 기본 캐릭터 (첫 번째 캐릭터)
        default_character = list(self.characters.keys())[0]

        try:
            success = delete_all_sessions()
            if not success:
                sessions = get_existing_sessions()
                return (
                    gr.update(visible=False),  # reset_modal
                    gr.update(visible=False),  # single_content
                    gr.update(visible=False),  # all_content
                    gr.update(),               # msg
                    history,                   # history 유지
                    chatbot,                   # chatbot 유지
                    "❌ 모든 세션 초기화에 실패했습니다.",  # status
                    gr.update(choices=sessions, value="demo_session"),  # session_dropdown
                    "demo_session",            # session_id
                )

            # 새로운 시스템 메시지로 히스토리 초기화
            default_system = {
                "role": "system",
                "content": system_message_default
            }
            new_history = [default_system]

            # demo_session에 대해 새 히스토리 저장 (기본 캐릭터로)
            save_chat_history_db(new_history, session_id="demo_session", selected_character=default_character)

            # chatbot 컴포넌트를 위한 메시지 필터링
            chatbot_history = self.filter_messages_for_chatbot(new_history)

            sessions = get_existing_sessions()
            return (
                gr.update(visible=False),  # reset_modal
                gr.update(visible=False),  # single_content
                gr.update(visible=False),  # all_content
                "",                        # msg
                new_history,               # new_history
                chatbot_history,           # filtered chatbot messages
                "✅ 모든 세션이 초기화되었습니다.",  # status
                gr.update(choices=sessions, value="demo_session"),  # session_dropdown
                "demo_session",            # session_id
            )

        except Exception as e:
            logger.error(f"Error resetting all sessions: {str(e)}")
            sessions = get_existing_sessions()
            return (
                gr.update(visible=False),  # reset_modal
                gr.update(visible=False),  # single_content
                gr.update(visible=False),  # all_content
                "",                        # msg
                history,                   # history 유지
                chatbot,                   # chatbot 유지
                f"❌ 모든 세션 초기화 중 오류가 발생했습니다: {str(e)}",  # status
                gr.update(choices=sessions, value="demo_session"),  # session_dropdown
                "demo_session",            # session_id
            )

    def refresh_preset_list(self, language: str | None = None):
        """프리셋 목록을 갱신하는 함수."""
        if language is None:
            language = self.default_language
        presets = get_preset_choices(language)
        return gr.update(choices=presets, value=presets[0] if presets else None)

    @staticmethod
    def refresh_sessions():
        """
        세션 목록을 갱신하고, (Dropdown) choices를 반환합니다.
        """
        sessions = get_existing_sessions()
        if not sessions:
            return gr.update(choices=[], value=None), "DB에 세션이 없습니다."
        return gr.update(choices=sessions, value=sessions[0])

    @staticmethod
    def refresh_sessions_with_names():
        """
        세션 목록을 (id, name) 튜플로 갱신합니다.
        """
        return get_existing_sessions_with_names()

    @staticmethod
    def create_new_session(system_message_box_value: str, selected_character: str):
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
        save_chat_history_db(new_history, session_id=new_sid, selected_character=selected_character)
        return new_sid, f"현재 세션: {new_sid}", new_history

    @staticmethod
    def create_temp_session(system_message: str, selected_character: str):
        """
        임시 세션을 생성합니다 (DB에 저장하지 않음).
        첫 번째 메시지가 전송될 때 실제 세션으로 변환됩니다.

        Returns:
            tuple: (is_temp_session, temp_history, temp_system_message, temp_character, chatbot_display, status_message)
        """
        temp_history = [{"role": "system", "content": system_message}]
        return (
            True,  # is_temp_session
            temp_history,  # history
            system_message,  # temp_system_message
            selected_character,  # temp_character
            [],  # empty chatbot display
            "New Chat - 메시지를 입력하면 새 세션이 생성됩니다."  # status
        )

    @staticmethod
    def confirm_temp_session(
        temp_history: list,
        first_message_content: str,
        selected_character: str,
        selected_model: str,
        model_type: str,
        provider: Literal["openai", "anthropic", "google-genai", "perplexity", "xai", "mistralai", "openrouter", "hf-inference", "ollama", "lmstudio", "oobabooga", "self-provided"],
        selected_lora: str,
        custom_path: str,
        device: str,
        image_input=None
    ):
        """
        임시 세션을 실제 세션으로 변환합니다.
        첫 번째 사용자 메시지가 전송될 때 호출됩니다.

        Returns:
            tuple: (new_session_id, session_title)
        """
        # Generate new session ID
        new_sid = secrets.token_hex(8)

        # Generate title from first message
        title = None
        try:
            title = generate_chat_title(
                first_message=first_message_content,
                selected_model=selected_model,
                model_type=model_type,
                provider=provider,
                selected_lora=selected_lora,
                local_model_path=custom_path if selected_model == "사용자 지정 모델 경로 변경" else None,
                lora_path=None,
                device=device,
                image_input=image_input
            )
        except Exception as e:
            logger.warning(f"Failed to generate chat title: {e}")

        # Fallback to truncated first message if title generation fails
        if not title:
            if isinstance(first_message_content, str):
                title = first_message_content[:50] + "..." if len(first_message_content) > 50 else first_message_content
            elif isinstance(first_message_content, list):
                # Handle multimodal content
                text_content = next((item.get("text", "") for item in first_message_content if item.get("type") == "text"), "New Chat")
                title = text_content[:50] + "..." if len(text_content) > 50 else text_content
            else:
                title = "New Chat"

        # Save to DB with the generated title
        save_chat_history_db(temp_history, session_id=new_sid, selected_character=selected_character)
        update_session_name(new_sid, title)

        return new_sid, title

    @staticmethod
    def apply_session(chosen_sid: str):
        """
        선택된 세션의 히스토리를 불러오고, session_id_state를 갱신.
        Also loads the last used character for the session.

        Returns:
            tuple: (loaded_history, session_id, last_character, status_message)
        """
        if not chosen_sid:
            return [], None, None, "세션 ID를 선택하세요."
        loaded_history = load_chat_from_db(chosen_sid)

        # 세션의 마지막 사용 캐릭터 조회 및 last_activity 갱신
        last_character = None
        with sqlite3.connect("chat_history.db") as conn:
            cursor = conn.cursor()
            # 먼저 last_character 조회
            cursor.execute("""
                SELECT last_character FROM sessions WHERE id = ?
            """, (chosen_sid,))
            row = cursor.fetchone()
            if row and row[0]:
                last_character = row[0]

            # last_activity 갱신
            cursor.execute("""
                UPDATE sessions
                SET last_activity = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (chosen_sid,))
            conn.commit()

        return loaded_history, chosen_sid, last_character, f"세션 {chosen_sid}이 적용되었습니다."

    @staticmethod
    def delete_session(chosen_sid: str, current_sid: str):
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
            # 먼저 chat_history에서 삭제
            c.execute("DELETE FROM chat_history WHERE session_id = ?", (chosen_sid,))
            # sessions 테이블에서도 삭제
            c.execute("DELETE FROM sessions WHERE id = ?", (chosen_sid,))
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

    def initial_load_presets(self, language: str | None = None):
        """초기 프리셋 로딩 함수"""
        if language is None:
            language = self.default_language
        presets = get_preset_choices(language)
        return gr.update(choices=presets, value=presets[0] if presets else None)


    def process_character_conversation(self, history: list[dict[str, str | Any]], selected_characters: str, model_type: str, selected_model: str, custom_path: str, image, api_key: str, device: str, seed: int, temperature: float, top_k: int, top_p: float, repetition_penalty: float):
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
                    seed=seed,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    character_language=translation_manager.current_language
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
    
    @staticmethod
    def toggle_api_key_visibility(provider: str | gr.Dropdown) -> bool:
        """
        OpenAI API Key 입력 필드의 가시성을 제어합니다.
        """
        api_visible = any(x in provider.lower() for x in ["openai", "anthropic", "google-genai", "perplexity", "xai", "mistralai", "openrouter", "hf-inference"])
        return gr.update(visible=api_visible)

    def toggle_standard_msg_input_visibility(self, selected_model: str | gr.Dropdown) -> bool:
        msg_visible = all(x not in selected_model.lower() for x in [
                "claude",
                "gpt-4o",
                "gpt-4.1",
                "gpt-5",
                "vision",
                "llava",
                "qwen2-vl",
                "qwen2.5-vl",
                "qwen2.5-omni",
                "qwen3-vl",
                "qwen3-onmi",
                "llama-4",
                "pixtral",
                "mistral-small-3",
                "paligemma",
                "gemma-3-4b",
                "gemma-3-12b",
                "gemma-3-27b",
                "gemma-3n",
                "phi-4-multimodal",
                "glm4v",
            ]
        )
        self.vision_model = msg_visible
        return gr.update(visible=msg_visible)

    def toggle_multimodal_msg_input_visibility(self, selected_model: str | gr.Dropdown) -> bool:
        msg_visible = any(x in selected_model.lower() for x in [
                "claude",
                "gpt-4o",
                "gpt-4.1",
                "gpt-5",
                "vision",
                "llava",
                "qwen2-vl",
                "qwen2.5-vl",
                "qwen2.5-omni",
                "qwen3-vl",
                "qwen3-onmi",
                "llama-4",
                "pixtral",
                "mistral-small-3",
                "paligemma",
                "gemma-3-4b",
                "gemma-3-12b",
                "gemma-3-27b",
                "gemma-3n",
                "phi-4-multimodal",
                "glm4v",
            ]
        )
        self.vision_model = msg_visible
        return gr.update(visible=msg_visible)

    # @staticmethod
    # def toggle_image_input_visibility(selected_model: str | gr.Dropdown) -> bool:
    #     """
    #     이미지 입력 필드의 가시성을 제어합니다.
    #     """
    #     image_visible = (
    #         "vision" in selected_model.lower() or
    #         "qwen2-vl" in selected_model.lower() or
    #         "qwen2.5-vl" in selected_model.lower() or
    #         selected_model in [
    #             "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B",
    #             "THUDM/glm-4v-9b",
    #             "openbmb/MiniCPM-Llama3-V-2_5"
    #         ]
    #     )
    #     return gr.update(visible=image_visible)

    @staticmethod
    def toggle_lora_visibility(provider: str | gr.Dropdown) -> bool:
        """
        LORA 파일 경로 입력 필드의 가시성을 제어합니다.
        """
        lora_visible = "self-provided" in provider.lower()
        return gr.update(visible=lora_visible)
    
    @staticmethod
    def toggle_enable_thinking_visibility(selected_model: str | gr.Dropdown) -> bool:
        """
        Thinking 애니메이션의 가시성을 제어합니다.
        """

        enable_thinking = any(x in selected_model.lower() for x in REASONING_KWD) and all(x not in selected_model.lower() for x in REASONING_BAN)
        thinking_visible = any(x in selected_model.lower() for x in REASONING_CONTROLABLE)

        return gr.update(value=enable_thinking, visible=thinking_visible, interactive=thinking_visible)

    def update_model_list(self, provider: str, selected_type: str | None = None):
        local_models_data = get_all_local_models()
        transformers_local = local_models_data["transformers"]
        gguf_local = local_models_data["gguf"]
        mlx_local = local_models_data["mlx"]
        
        if provider != "self-provided":
            if provider == "openai":
                updated_list = openai_llm_api_models
            elif provider == "anthropic":
                updated_list = anthropic_llm_api_models
            elif provider == "google-genai":
                updated_list = google_genai_llm_api_models
            elif provider == "perplexity":
                updated_list = perplexity_llm_api_models
            elif provider == "xai":
                updated_list = xai_llm_api_models
            elif provider == "mistralai":
                updated_list = mistralai_llm_api_models
            elif provider == "openrouter":
                updated_list = openrouter_llm_api_models
            elif provider == "hf-inference":
                updated_list = huggingface_inference_llm_api_models
            elif provider == "ollama":
                updated_list = ollama_llm_models
            elif provider == "lmstudio":
                updated_list = lmstudio_llm_models
            elif provider == "vllm-api":
                updated_list = vllm_llm_api_models
            # elif provider == "oobabooga":
            #     updated_list = oobabooga_llm_models

            updated_list = sorted(list(dict.fromkeys(updated_list)))
            return gr.update(visible='hidden'), gr.update(choices=updated_list, value=updated_list[0] if updated_list else None)
        
        else: # elif provider == "self-provided"
            # "전체 목록"이면 => API 모델 + 모든 로컬 모델 + "사용자 지정 모델 경로 변경"
            if selected_type == "all":
                all_models = self.update_allowed_models()
                all_models = sorted(list(dict.fromkeys(all_models)))
                # 중복 제거 후 정렬
                return gr.update(visible=True), gr.update(choices=all_models, value=all_models[0] if all_models else None)
            
            # 개별 로컬 모델 유형 선택
            elif selected_type == "transformers":
                updated_list = transformers_local
            elif selected_type == "vllm-local":
                updated_list = vllm_local
            elif selected_type == "gguf":
                updated_list = gguf_local
            elif selected_type == "mlx":
                updated_list = mlx_local
            else:
                updated_list = transformers_local
            
            updated_list = sorted(list(dict.fromkeys(updated_list)))
            return gr.update(visible=True), gr.update(choices=updated_list, value=updated_list[0] if updated_list else None)
    
    def update_allowed_models(self):
        if self.os_name == "Darwin":
            return transformers_local + gguf_local + mlx_local
        else:
            return transformers_local + gguf_local

    def show_reset_modal(self, reset_type: bool):
        """초기화 확인 모달 표시"""
        self.reset_type = reset_type
        return (
            gr.update(visible=True),  # modal
            gr.update(visible=reset_type == "single"),  # single_content
            gr.update(visible=reset_type == "all"),  # all_content
        )

    @staticmethod
    def hide_reset_modal():
        """초기화 확인 모달 숨김"""
        return (
            gr.update(visible=False),  # modal
            gr.update(visible=False),  # single_content
            gr.update(visible=False),  # all_content
        )

    def handle_reset_confirm(self, history: list[dict[str, str | Any]], chatbot: list[dict[str, str | Any]], system_msg: str, selected_character: str, language: str | None = None, session_id: str = "demo_session"):
        """초기화 확인 시 처리

        Returns for single reset:
            tuple: (reset_modal, single_content, all_content, msg, new_history, chatbot_history, status, session_dropdown)
        Returns for all reset:
            tuple: (reset_modal, single_content, all_content, msg, new_history, chatbot_history, status, session_dropdown, session_id)
        """
        try:
            if self.reset_type == "single":
                result = self.reset_session(
                    history=history,
                    chatbot=self.filter_messages_for_chatbot(history),
                    system_message_default=system_msg,
                    selected_character=selected_character,
                    language=language,
                    session_id=session_id
                )
            else:  # reset_type == "all"
                result = self.reset_all_sessions(
                    history=history,
                    chatbot=self.filter_messages_for_chatbot(history),
                    system_message_default=system_msg,
                    selected_character=selected_character,
                    language=language
                )

            # 모달 업데이트와 결과를 함께 반환
            return result

        except Exception as e:
            logger.error(f"Reset confirmation error: {str(e)}")
            sessions = get_existing_sessions()
            # 오류 발생 시 현재 상태 유지
            return (
                gr.update(visible=False),  # reset_modal
                gr.update(visible=False),  # single_content
                gr.update(visible=False),  # all_content
                "",                        # msg
                history,                   # 현재 history 유지
                self.filter_messages_for_chatbot(history),  # 현재 chatbot 상태 유지
                f"❌ 초기화 중 오류가 발생했습니다: {str(e)}",  # status
                gr.update(choices=sessions, value=session_id),  # session_dropdown
            )
        
    @staticmethod
    def create_reset_confirm_modal():
        """초기화 확인 모달 생성"""
        with gr.Column(visible=False, elem_classes="reset-confirm-modal") as reset_modal:
            gr.Markdown(f"# {_("reset_confirm_title")}", elem_classes="reset-confirm-title")
            with gr.Column() as single_reset_content:
                gr.Markdown(_("reset_confirm_current_message"), 
                        elem_classes="reset-confirm-message")
            with gr.Column(visible=False) as all_reset_content:
                gr.Markdown(_("reset_confirm_all_message"), 
                        elem_classes="reset-confirm-message")
            with gr.Row(elem_classes="reset-confirm-buttons"):
                cancel_btn = gr.Button(_("cancel"), variant="secondary")
                confirm_btn = gr.Button(_("ok"), variant="primary")
                
        return (reset_modal, single_reset_content, all_reset_content, 
                cancel_btn, confirm_btn)
    
    @staticmethod
    def create_delete_session_modal():
        """삭제 확인 모달 생성"""
        with gr.Column(visible=False, elem_classes="delete-session-modal") as delete_modal:
            gr.Markdown("# ⚠️ 세션 삭제 확인", elem_classes="delete-session-title")
            message = gr.Markdown("", elem_classes="delete-session-message")
            with gr.Row(elem_classes="delete-session-buttons"):
                cancel_btn = gr.Button("취소", variant="secondary")
                confirm_btn = gr.Button("삭제", variant="stop")
                    
        return delete_modal, message, cancel_btn, confirm_btn
    
    def get_allowed_llm_models(self) -> tuple[list[str], list[str]]:
        if self.os_name == "Darwin":
            allowed = transformers_local + gguf_local + mlx_local
            allowed_type = ["all", "transformers", "gguf", "mlx"]

        else:
            allowed = transformers_local + gguf_local
            allowed_type = ["all", "transformers", "gguf"]
            
        allowed = list(dict.fromkeys(allowed))
        
        return sorted(allowed), allowed_type