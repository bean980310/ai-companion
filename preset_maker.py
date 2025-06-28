import json
import re
from textwrap import dedent

def strip_indent(text: str) -> str:
    """문자열에서 줄바꿈 뒤에 오는 공백·탭을 제거한다."""
    return re.sub(r'\n[ \t]+', '\n', text)

def build_preset(template: str, data: dict) -> str:
    """
    템플릿에 데이터를 꽂아 완성된 프리셋 문자열을 돌려준다.
    키 누락이 있으면 KeyError를 던지니 참고!
    """
    return template.format(**data)

PRESET_TEMPLATE = dedent("""
# {preset_name}

[Profile]
- Name: {name}
- Gender: {gender}
- Age: {age}
- Residence: {residence}
- Languages: {languages}

[Personality]{personality}
[Speech Style]{speech_style}
[Appearance]{appearance}
[Hobbies]{hobbies}
[Signature Lines]{signature_lines}
[Conversation Style]{conversation_style}
[Goals]{goals}
[Reactions Based on User Behavior]{response_by_user_actions}
""").strip()

PRESET_TEMPLATE_KOREAN = dedent("""
# {preset_name_korean}

[프로필]
- 이름: {name_korean}
- 성별: {gender_korean}
- 나이: {age_korean}
- 거주지: {residence_korean}
- 언어: {languages_korean}

[성격]{personality_korean}
[말투]{speech_style_korean}
[외형적 특징]{appearance_korean}
[취미]{hobbies_korean}
[대표 대사]{signature_lines_korean}
[대화 스타일]{conversation_style_korean}
[목표]{goals_korean}
[유저의 행동에 따른 반응]{response_by_user_actions_korean}
""").strip()

PRESET_TEMPLATE_JAPANESE = dedent("""
# {preset_name_japanese}

[プロフィール]
- 名前: {name_japanese}
- 性別: {gender_japanese}
- 年齢: {age_japanese}
- 一人称: {first_person_japanese}
- 居住地: {residence_japanese}
- 言語: {languages_japanese}

[性格]{personality_japanese}
[話し方]{speech_style_japanese}
[外見的特徴]{appearance_japanese}
[趣味]{hobbies_japanese}
[代表セリフ]{signature_lines_japanese}
[会話スタイル]{conversation_style_japanese}
[目標]{goals_japanese}
[ユーザーの行動による反応]{response_by_user_actions_japanese}
""").strip()

PRESET_TEMPLATE_CHINESE_SIMPLIFIED = dedent("""
# {preset_name_chinese_simplified}

[简介]
- 姓名: {name_chinese_simplified}
- 性别: {gender_chinese_simplified}
- 年龄: {age_chinese_simplified}
- 居住地: {residence_chinese_simplified}
- 语言: {languages_chinese_simplified}

[性格]{personality_chinese_simplified}
[说话风格]{speech_style_chinese_simplified}
[外貌特征]{appearance_chinese_simplified}
[兴趣]{hobbies_chinese_simplified}
[代表台词]{signature_lines_chinese_simplified}
[对话风格]{conversation_style_chinese_simplified}
[目标]{goals_chinese_simplified}
[针对用户行为的反应]{response_by_user_actions_chinese_simplified}
""").strip()

PRESET_TEMPLATE_CHINESE_TRADITIONAL = dedent("""
# {preset_name_chinese_traditional}

[簡介]
- 姓名: {name_chinese_traditional}
- 性別: {gender_chinese_traditional}
- 年齡: {age_chinese_traditional}
- 居住地: {residence_chinese_traditional}
- 語言: {languages_chinese_traditional}

[性格]{personality_chinese_traditional}
[說話風格]{speech_style_chinese_traditional}
[外貌特徵]{appearance_chinese_traditional}
[興趣]{hobbies_chinese_traditional}
[代表台詞]{signature_lines_chinese_traditional}
[對話風格]{conversation_style_chinese_traditional}
[目標]{goals_chinese_traditional}
[針對使用者行為的反應]{response_by_user_actions_chinese_traditional}
""").strip()

def create_preset_data_en(
    preset_name: str, 
    name: str, 
    gender: str, 
    age: int, 
    residence: str, 
    languages: str, 
    personality: str, 
    speech_style: str,
    appearance: str,
    hobbies: str,
    signature_lines: str,
    conversation_style: str,
    goals: str,
    response_by_user_actions: str
):
    return {
        "preset_name": preset_name,
        "name": name,
        "gender": gender,
        "age": age,
        "residence": residence,
        "languages": languages,
        "personality": personality,
        "speech_style": speech_style,
        "appearance": appearance,
        "hobbies": hobbies,
        "signature_lines": signature_lines,
        "conversation_style": conversation_style,
        "goals": goals,
        "response_by_user_actions": response_by_user_actions
    }

def create_preset_data_ko(
    preset_name: str, 
    name: str, 
    gender: str, 
    age: int, 
    residence: str, 
    languages: str, 
    personality: str, 
    speech_style: str,
    appearance: str,
    hobbies: str,
    signature_lines: str,
    conversation_style: str,
    goals: str,
    response_by_user_actions: str
):
    return {
        "preset_name_korean": preset_name,
        "name_korean": name,
        "gender_korean": gender,
        "age_korean": age,
        "residence_korean": residence,
        "languages_korean": languages,
        "personality_korean": personality,
        "speech_style_korean": speech_style,
        "appearance_korean": appearance,
        "hobbies_korean": hobbies,
        "signature_lines_korean": signature_lines,
        "conversation_style_korean": conversation_style,
        "goals_korean": goals,
        "response_by_user_actions_korean": response_by_user_actions
    }

def create_preset_data_ja(
    preset_name: str, 
    name: str, 
    gender: str, 
    age: int, 
    residence: str, 
    languages: str, 
    personality: str, 
    speech_style: str,
    appearance: str,
    hobbies: str,
    signature_lines: str,
    conversation_style: str,
    goals: str,
    response_by_user_actions: str
):
    return {
        "preset_name_japanese": preset_name,
        "name_japanese": name,
        "gender_japanese": gender,
        "age_japanese": age,
        "residence_japanese": residence,
        "languages_japanese": languages,
        "personality_japanese": personality,
        "speech_style_japanese": speech_style,
        "appearance_japanese": appearance,
        "hobbies_japanese": hobbies,
        "signature_lines_japanese": signature_lines,
        "conversation_style_japanese": conversation_style,
        "goals_japanese": goals,
        "response_by_user_actions_japanese": response_by_user_actions
    }

def create_preset_data_zh_cn(
    preset_name: str, 
    name: str, 
    gender: str, 
    age: int, 
    residence: str, 
    languages: str, 
    personality: str, 
    speech_style: str,
    appearance: str,
    hobbies: str,
    signature_lines: str,
    conversation_style: str,
    goals: str,
    response_by_user_actions: str
):
    return {
        "preset_name_chinese_simplified": preset_name,
        "name_chinese_simplified": name,
        "gender_chinese_simplified": gender,
        "age_chinese_simplified": age,
        "residence_chinese_simplified": residence,
        "languages_chinese_simplified": languages,
        "personality_chinese_simplified": personality,
        "speech_style_chinese_simplified": speech_style,
        "appearance_chinese_simplified": appearance,
        "hobbies_chinese_simplified": hobbies,
        "signature_lines_chinese_simplified": signature_lines,
        "conversation_style_chinese_simplified": conversation_style,
        "goals_chinese_simplified": goals,
        "response_by_user_actions_chinese_simplified": response_by_user_actions
    }

def create_preset_data_zh_tw(
    preset_name: str, 
    name: str, 
    gender: str, 
    age: int, 
    residence: str, 
    languages: str, 
    personality: str, 
    speech_style: str,
    appearance: str,
    hobbies: str,
    signature_lines: str,
    conversation_style: str,
    goals: str,
    response_by_user_actions: str
):
    return {
        "preset_name_chinese_traditional": preset_name,
        "name_chinese_traditional": name,
        "gender_chinese_traditional": gender,
        "age_chinese_traditional": age,
        "residence_chinese_traditional": residence,
        "languages_chinese_traditional": languages,
        "personality_chinese_traditional": personality,
        "speech_style_chinese_traditional": speech_style,
        "appearance_chinese_traditional": appearance,
        "hobbies_chinese_traditional": hobbies,
        "signature_lines_chinese_traditional": signature_lines,
        "conversation_style_chinese_traditional": conversation_style,
        "goals_chinese_traditional": goals,
        "response_by_user_actions_chinese_traditional": response_by_user_actions
    }