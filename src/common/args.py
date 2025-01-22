import argparse
from src.common.detect_language import detect_system_language

def parse_args():
    parser = argparse.ArgumentParser(description="Easy-LLM Application Setting")
    
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Gradio 서버가 실행될 포트 번호를 지정합니다. (default: %(default)d)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="모델의 예측을 재현 가능하게 하기 위한 시드 값을 지정합니다. (default: %(default)d)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버깅 모드를 활성화합니다. (default: %(default)s)"
    )
    
    parser.add_argument(
        "--inbrowser",
        type=bool,
        default=True,
        help="Gradio 앱을 브라우저에서 실행합니다. (default: %(default)s)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="gradio 앱의 공유 링크를 생성합니다."
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default=detect_system_language(),
        choices=["ko", "ja", "en", "zh_CN", "zh_TW"],
        help="애플리케이션의 기본 언어를 지정합니다. (default: %(default)s)"
    )
    
    return parser.parse_args()