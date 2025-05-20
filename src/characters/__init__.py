import gradio as gr

from .preset_images import PRESET_IMAGES
from .persona_speech_manager import PersonaSpeechManager

from .. import logger

__all__ = [
    "PersonaSpeechManager",
    "PRESET_IMAGES"
]

def handle_character_change(selected_character, language, speech_manager: PersonaSpeechManager):
    try:
        speech_manager.set_character_and_language(selected_character, language)
        system_message = speech_manager.get_system_message()
        return system_message, gr.update(value=speech_manager.characters[selected_character]["profile_image"])
    except ValueError as e:
        logger.error(str(e))
        return "❌ 선택한 캐릭터가 유효하지 않습니다.", gr.update(value=None)