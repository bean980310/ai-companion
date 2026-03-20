from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, GenerationMixin, PreTrainedTokenizerBase, TokenizersBackend
from src import logger

mbart_languages = {"English": "en_XX", "한국어(Korean)": "ko_KR", "日本語(Japanese)": "ja_XX", "简体中文(Simp. Chinese)": "zh_CN", "Français(French)": "fr_XX", "Deutsche(German)": "de_DE", "Español(Spanish)": "es_XX"}

m2m100_languages = {"English": "en", "한국어(Korean)": "ko", "日本語(Japanese)": "ja", "简体中文(Simp. Chinese)": "zh", "Français(French)": "fr", "Deutsche(German)": "de", "Español(Spanish)": "es"}


def get_lang_code(language: str, selected_model: str):
    if "mbart" in selected_model:
        return mbart_languages[language]
    else:
        return m2m100_languages[language]


def translate(text: str, src_lang: str, tgt_lang: str, model_name: str) -> str:
    tokenizer: AutoTokenizer | TokenizersBackend | PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    model: AutoModelForSeq2SeqLM | PreTrainedModel | GenerationMixin = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated[0]


def translate_interface(text: str, src_language: str, tgt_language: str, selected_model: str) -> str:
    """
    Translates text from source language to target language using the MBart model.
    Args:
        text: The text to translate.
        src_language: The source language as a string.
        tgt_language: The target language as a strings.
    Returns:
        The translated text as a string.
    """

    src_lang = get_lang_code(src_language, selected_model)
    tgt_lang = get_lang_code(tgt_language, selected_model)

    return translate(text, src_lang, tgt_lang, selected_model)
