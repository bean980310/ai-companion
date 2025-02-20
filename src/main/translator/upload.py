import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from .map_language import map_language

languages = {
    "English": "eng",
    "Korean": "kor",
    "Japanese": "jpn",
    "Chinese": "chi_sim",
    "French": "fra",
    "German": "deu",
    "Spanish": "spa"
}


def pdf_to_text(pdf_path: str, poppler_path: str=None, lang: str='eng') -> str:
    """
    PDF 파일을 텍스트로 변환합니다.
    """
    # PDF 파일을 이미지로 변환
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    
    # 이미지에서 텍스트 추출
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image, lang=lang)
    
    return text

def image_to_text(image_path: str, lang: str='eng') -> str:
    """
    이미지 파일을 텍스트로 변환합니다.
    """
    # 이미지 파일 열기
    image = Image.open(image_path)

    # 이미지에서 텍스트 추출
    text = pytesseract.image_to_string(image, lang=lang)

    return text

def upload_handler(file_path, language):
    lang=languages[map_language(language)]
    if file_path.endswith('.pdf'):
        # PDF 파일 처리
        text = pdf_to_text(file_path, lang=lang)
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        # 이미지 파일 처리
        text = image_to_text(file_path, lang=lang)
    else:
        raise ValueError("지원되지 않는 파일 형식입니다.")

    return text