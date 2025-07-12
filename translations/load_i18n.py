import json

def i18n_en():
    with open("translations/en.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def i18n_ko():
    with open("translations/ko.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def i18n_ja():
    with open("translations/ja.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def i18n_zh_CN():
    with open("translations/zh_CN.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def i18n_zh_TW():
    with open("translations/zh_TW.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    return data