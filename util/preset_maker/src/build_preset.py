import json
import re

def strip_indent(text: str) -> str:
    """문자열에서 줄바꿈 뒤에 오는 공백·탭을 제거한다."""
    return re.sub(r'\n[ \t]+', '\n', text)

def build_preset(template: str, data: dict) -> str:
    """
    템플릿에 데이터를 꽂아 완성된 프리셋 문자열을 돌려준다.
    키 누락이 있으면 KeyError를 던지니 참고!
    """
    return template.format(**data)