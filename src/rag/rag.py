import os
import re
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# 언어 이름과 코드 매핑 딕셔너리
LANGUAGE_MAP = {
    "한국어": "ko",
    "日本語": "ja",
    "简体中文": "zh_CN",
    "繁體中文": "zh_TW",
    "English": "en"
}

def extract_language_code(heading):
    """
    헤딩 텍스트 (예: "## 한국어")에서 언어 이름을 추출하여
    매핑 딕셔너리를 통해 언어 코드를 반환하는 함수.
    """
    match = re.match(r"##\s*(.+)", heading)
    if match:
        lang_name = match.group(1).strip()
        return LANGUAGE_MAP.get(lang_name, lang_name)
    return None


def split_document_by_sections(text):
    """
    텍스트에서 [항목] 형태의 섹션을 찾아서 (섹션 제목, 내용) 튜플 리스트로 반환하는 함수.
    """
    # 정규표현식 패턴: 대괄호 안에 항목명, 그 다음 내용은 다음 대괄호가 나오기 전까지
    pattern = re.compile(r'\[(.*?)\]\s*(.*?)(?=\n\s*\[|$)', re.DOTALL)
    sections = []
    for match in pattern.finditer(text):
        section_title = match.group(1).strip()
        section_content = match.group(2).strip()
        sections.append((section_title, section_content))
    return sections

def process_document(doc):
    """
    Document 객체를 받아서 언어 헤딩("## 언어명") 또는 기존 방식으로 언어 코드를 추출하고,
    각 항목별 섹션으로 분리한 후 새로운 Document 리스트로 반환.
    """
    text = doc.page_content
    lines = text.splitlines()
    
    # 파일 첫 줄이 언어 헤딩(예: "## 한국어")이면 추출
    if lines and lines[0].startswith("##"):
        lang_code = extract_language_code(lines[0])
        # 헤딩 부분을 제거한 나머지 내용 사용
        content = "\n".join(lines[1:]).strip()
    else:
        # 기존 방식: "ko": """ 로 시작하는 경우
        lang_match = re.search(r'^"([^"]+)":\s*"""', text)
        if lang_match:
            lang_code = lang_match.group(1)
            content = re.sub(r'^"[^"]+":\s*"""', '', text)
            content = re.sub(r'"""$', '', content).strip()
        else:
            lang_code = doc.metadata.get("language", "unknown")
            content = text

    sections = split_document_by_sections(content)
    processed = []
    for section_title, section_content in sections:
        new_doc = Document(
            page_content=section_content,
            metadata={
                "language": lang_code,
                "section": section_title,
                "source": doc.metadata.get("source", "unknown")
            }
        )
        processed.append(new_doc)
    return processed

loader = DirectoryLoader(path='characters_info', glob="*.txt", loader_cls=TextLoader)

docs=loader.load()
docs=docs.sort(key=lambda x: x.metadata['source'].split('/')[-1])
len(docs)

all_processed_docs = []

for doc in docs:
    processed_docs = process_document(doc)
    all_processed_docs.extend(processed_docs)
    
len(all_processed_docs)
