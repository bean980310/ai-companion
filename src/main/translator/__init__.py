from .translator import translate_interface
from .upload import upload_handler

LANGUAGES = {
    "English": "English",
    "한국어(Korean)": "Korean",
    "日本語(Japanese)": "japanese",
    "简体中文(Simp. Chinese)": "Chinese",
    "Français(French)": "French",
    "Deutsche(German)": "German",
    "Español(Spanish)": "Spanish"
}

def display_language(language):
    lang_display = {
        "English": "English",
        "Korean": "한국어(Korean)",
        "Japanese": "日本語(Japanese)",
        "Chinese": "简体中文(Simp. Chinese)",
        "French": "Français(French)",
        "German": "Deutsche(German)",
        "Spanish": "Español(Spanish)"
    }
    return lang_display.get(language, language)

__all__ = ["translate_interface", "upload_handler", "LANGUAGES"]