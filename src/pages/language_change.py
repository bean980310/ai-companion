import gradio as gr
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
from src.common.translations import translation_manager, _
from src.common_blocks import get_language_code
from src.start_app import app_state

def on_chat_language_change(selected_lang: str, selected_character: str):
    lang_code = get_language_code(selected_lang)
    translation_manager.set_language(lang_code)

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

    return [
        gr.update(value=f"## {_('main_title')}"),
        gr.update(label=_('language_select'), info=_('language_info')),
        gr.update(label=_('system_message')),
        gr.update(label=_('system_message'), value=system_content),
        gr.update(label=_('advanced_setting')),
        gr.update(label=_('seed_label'), info=_('seed_info')),
        gr.update(label=_('temperature_label')),
        gr.update(label=_('top_k_label')),
        gr.update(label=_('top_p_label')),
        gr.update(label=_('repetition_penalty_label')),
        gr.update(value=_('reset_session_button')),
        gr.update(value=_('reset_all_sessions_button')),
        lang_code
    ]

def on_image_gen_language_change(selected_lang: str):
    lang_code = get_language_code(selected_lang)
    translation_manager.set_language(lang_code)
    return [
        gr.update(value=f"## {_('image_gen_title')}"),
        gr.update(label=_('language_select'), info=_('language_info'))
    ]



def on_storyteller_language_change(selected_lang: str):
    lang_code = get_language_code(selected_lang)
    translation_manager.set_language(lang_code)
    return [
        gr.update(value=f"## {_('storyteller_title')}"),
        gr.update(label=_('language_select'), info=_('language_info'))
    ]

def on_tts_language_change(selected_lang: str):
    lang_code = get_language_code(selected_lang)
    translation_manager.set_language(lang_code)
    return [
        gr.update(value=f"## {_('tts_title')}"),
        gr.update(label=_('language_select'), info=_('language_info'))
    ]

def on_translator_language_change(selected_lang: str):
    lang_code = get_language_code(selected_lang)
    translation_manager.set_language(lang_code)
    return [
        gr.update(value=f"## {_('translator_title')}"),
        gr.update(label=_('language_select'), info=_('language_info'))
    ]

def on_settings_language_change(selected_lang: str):
    lang_code = get_language_code(selected_lang)
    translation_manager.set_language(lang_code)
    return [
        gr.update(value=f"## {_('settings_title')}"),
        gr.update(label=_('language_select'), info=_('language_info'))
    ]


def on_mcp_client_language_change(selected_lang: str):
    lang_code = get_language_code(selected_lang)
    translation_manager.set_language(lang_code)
    return [
        gr.update(value=f"## {_('mcp_client_title')}"),
        gr.update(label=_('language_select'), info=_('language_info'))
    ]

def on_download_language_change(selected_lang: str):
    lang_code = get_language_code(selected_lang)
    translation_manager.set_language(lang_code)
    return [
        gr.update(value=f"## {_('download_title')}"),
        gr.update(label=_('language_select'), info=_('language_info'))
    ]