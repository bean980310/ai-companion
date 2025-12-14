import json
from typing import Dict, List
import gradio as gr
import gradio.i18n
gr.i18n = gradio.i18n
# from gradio_i18n import Translate, translate_blocks, gettext as _
from torch._dynamo.utils import key_is_id

from presets import CHARACTER_LIST, character_key

# lang_code_index = ["ko", "ja", "zh_CN", "zh_TW", "en"]

def i18n_en() -> Dict[str, str]:
    with open("translations/en.json", "r", encoding="utf-8") as f:
        data: dict[str, str] = json.load(f)

    for key, value in zip(character_key, CHARACTER_LIST):
        data[key] = value['en']

    return data

def i18n_ko() -> Dict[str, str]:
    with open("translations/ko.json", "r", encoding="utf-8") as f:
        data: dict[str, str] = json.load(f)

    for key, value in zip(character_key, CHARACTER_LIST):
        data[key] = value['ko']

    return data

def i18n_ja() -> Dict[str, str]:
    with open("translations/ja.json", "r", encoding="utf-8") as f:
        data: dict[str, str] = json.load(f)

    for key, value in zip(character_key, CHARACTER_LIST):
        data[key] = value['ja']

    return data

def i18n_zh_CN() -> Dict[str, str]:
    with open("translations/zh_CN.json", "r", encoding="utf-8") as f:
        data: dict[str, str] = json.load(f)

    for key, value in zip(character_key, CHARACTER_LIST):
        data[key] = value['zh_CN']

    return data

def i18n_zh_TW() -> Dict[str, str]:
    with open("translations/zh_TW.json", "r", encoding="utf-8") as f:
        data: dict[str, str] = json.load(f)

    for key, value in zip(character_key, CHARACTER_LIST):
        data[key] = value['zh_TW']

    return data

en = i18n_en()
ko = i18n_ko()
ja = i18n_ja()
zh_CN = i18n_zh_CN()
zh_TW = i18n_zh_TW()

# lang_list_index = [ko, ja, zh_CN, zh_TW, en]
# lang_code_index = ["ko", "ja", "zh_CN", "zh_TW", "en"]


# for key in character_key:
#     tmp = int(key)
#     c_value_index = CHARACTER_LIST[tmp]

#     for code in lang_code_index:
#         c_value_index
    

lang_store = {
    "en": en,
    "ko": ko,
    "ja": ja,
    "zh-CN": zh_CN,
    "zh-TW": zh_TW,
}

# gr.i18n = gradio.i18n

# i18ndata = gr.i18n.I18nData()
i18n = gr.I18n(translations=lang_store)
# i18n_test = gr.I18n(
    
# )

def create_language_component():
    return gr.Dropdown(
        choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
        render=False,
    )