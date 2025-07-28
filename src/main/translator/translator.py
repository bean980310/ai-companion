from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from src import logger

languages = {
    "English": "en_XX",
    "한국어(Korean)": "ko_KR",
    "日本語(Japanese)": "ja_XX",
    "简体中文(Simp. Chinese)": "zh_CN",
    "Français(French)": "fr_XX",
    "Deutsche(German)": "de_DE",
    "Español(Spanish)": "es_XX"
}

def translate(text, src_lang, tgt_lang):
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated[0]

def translate_interface(text, src_language, tgt_language):
    
    """
    Translates text from source language to target language using the MBart model.
    Args:
        text: The text to translate.
        src_language: The source language as a string.
        tgt_language: The target language as a strings.
    Returns:
        The translated text as a string.
    """
    src_lang = languages[src_language]
    tgt_lang = languages[tgt_language]
    return translate(text, src_lang, tgt_lang)