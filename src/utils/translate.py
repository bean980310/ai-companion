import gradio as gr
from gradio_i18n import translate_blocks

from presets import (
    AI_ASSISTANT_PRESET, 
    SD_IMAGE_GENERATOR_PRESET, 
    MINAMI_ASUKA_PRESET, 
    MAKOTONO_AOI_PRESET, 
    AINO_KOITO_PRESET, 
    ARIA_PRINCESS_FATE_PRESET, 
    ARIA_PRINCE_FATE_PRESET,
    WANG_MEI_LING_PRESET,
    MISTY_LANE_PRESET,
    LILY_EMPRESS_PRESET,
    CHOI_YUNA_PRESET,
    CHOI_YURI_PRESET
    )

from src.common.character_info import characters
from src.common.translations import translation_manager
from src.start_app import app_state

def change_language(selected_lang: str, selected_character: str):
    """언어 변경 처리 함수"""
    lang_map = {
        "한국어": "ko",
        "日本語": "ja",
        "中文(简体)": "zh_CN",
        "中文(繁體)": "zh_TW",
        "English": "en"
    }
        
    lang_code = lang_map.get(selected_lang, "ko")
        
    if translation_manager.set_language(lang_code):
        if selected_lang in characters[selected_character]["languages"]:
            app_state.speech_manager_state.current_language = selected_lang
        else:
            app_state.speech_manager_state.current_language = characters[selected_character]["languages"][0]
                
            
        system_presets: dict[str, dict[str, str]] = {
            "AI 비서 (AI Assistant)": AI_ASSISTANT_PRESET,
            "Image Generator": SD_IMAGE_GENERATOR_PRESET,
            "미나미 아스카 (南飛鳥, みなみあすか, Minami Asuka)": MINAMI_ASUKA_PRESET,
            "마코토노 아오이 (真琴乃葵, まことのあおい, Makotono Aoi)": MAKOTONO_AOI_PRESET,
            "아이노 코이토 (愛野小糸, あいのこいと, Aino Koito)": AINO_KOITO_PRESET,
            "아리아 프린세스 페이트 (アリア·プリンセス·フェイト, Aria Princess Fate)": ARIA_PRINCESS_FATE_PRESET,
            "아리아 프린스 페이트 (アリア·プリンス·フェイト, Aria Prince Fate)": ARIA_PRINCE_FATE_PRESET,
            "왕 메이린 (王美玲, ワン·メイリン, Wang Mei-Ling)": WANG_MEI_LING_PRESET,
            "미스티 레인 (ミスティ·レーン, Misty Lane)": MISTY_LANE_PRESET,
            '릴리 엠프레스 (リリー·エンプレス, Lily Empress)': LILY_EMPRESS_PRESET,
            "최유나 (崔有娜, チェ·ユナ, Choi Yuna)": CHOI_YUNA_PRESET,
            "최유리 (崔有莉, チェ·ユリ, Choi Yuri)": CHOI_YURI_PRESET
        }
                
        preset_name = system_presets.get(selected_character, AI_ASSISTANT_PRESET)
        system_content = preset_name.get(lang_code, "당신은 유용한 AI 비서입니다.")

def translate_page(blocks, translation, selected_lang, persistant=True):
    lang_map = {
        "한국어": "ko",
        "日本語": "ja",
        "中文(简体)": "zh_CN",
        "中文(繁體)": "zh_TW",
        "English": "en"
    }
    lang_code = lang_map.get(selected_lang, "ko")
    
    if lang_code == "zh_CN":
        lang_code = "zh"
    if lang_code == "zh_TW":
        lang_code = "zh-Hant"
    translate_blocks(blocks, translation, lang_code, persistant)
    