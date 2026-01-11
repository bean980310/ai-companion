import sqlite3
import importlib
import gradio as gr
from typing import List, Dict, Any, Tuple
from pathlib import Path

from .logging import logger
from .common.args import parse_args
from .common.character_info import characters
from .common.database import (
    add_system_preset, 
    load_system_presets, 
    delete_system_preset,
    initialize_database,
    ensure_demo_session,
    insert_default_presets,
    get_existing_sessions,
    load_chat_from_db,
    get_preset_choices)
from .common.default_language import default_language
from .common.translations import translation_manager, _
from .common.utils import detect_platform
from .characters.persona_speech_manager import PersonaSpeechManager
from ._version import __version__

args = parse_args()
os_name, arch, is_wsl = detect_platform()

__all__ = ['logger']
# __version__ = "0.2.0"

# from . import chat, image_gen, storyteller, tts, translator, download, settings