from .translator import translate_interface
from .upload import upload_handler

LANGUAGES = [
    "English", 
    "한국어(Korean)", 
    "日本語(Japanese)", 
    "简体中文(Simp. Chinese)", 
    "Français(French)", 
    "Deutsche(German)", 
    "Español(Spanish)"
]


__all__ = ["translate_interface", "upload_handler", "LANGUAGES"]