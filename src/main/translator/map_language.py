def map_language(language):
    lang_map={
        "English": "English",
        "한국어(Korean)": "Korean",
        "日本語(Japanese)": "japanese",
        "简体中文(Simp. Chinese)": "Chinese",
        "Français(French)": "French",
        "Deutsche(German)": "German",
        "Español(Spanish)": "Spanish"
    }
    return lang_map.get(language)