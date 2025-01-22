import logging
import locale

logger = logging.getLogger(__name__)

def detect_system_language() -> str:
    """시스템 기본 언어를 감지합니다."""
    try:
        lang, _ = locale.getdefaultlocale()
        if lang:
            lang_code = lang.split('_')[0]
            if lang_code == 'zh':
                # 중국어의 경우 간체/번체 구분
                if lang.lower() == 'zh_tw':
                    return 'zh_TW'
                return 'zh_CN'
            if lang_code in ['ko', 'ja', 'en']:
                return lang_code
    except Exception as e:
        logger.error(f"Error detecting system language: {e}")
    
    return 'en'