from textwrap import dedent

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