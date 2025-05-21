from .tts import text_to_speech
from .component import create_tts_side

from ...models import tts_api_models, vits_local
from ...start_app import app_state

__all__ = ["text_to_speech"]

def get_tts_models():
    tts_choices = tts_api_models + vits_local
    tts_choices = list(dict.fromkeys(tts_choices))
    tts_choices = sorted(tts_choices)
    
    app_state.tts_choices = tts_choices
    
    return tts_choices