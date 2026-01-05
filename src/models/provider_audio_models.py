import os
from typing import List
import re
from ..common.environ_manager import load_env_variables

def get_openai_asr_models(api_key: str = None):
    import openai
    from openai import OpenAI

    model_list = []

    if not api_key:
        model_list.append("OpenAI API Key가 필요합니다.")
        return model_list

    client = OpenAI(api_key=api_key)

    try:
        model = client.models.list()

        for m in model.data:
            model_id = m.id

            if any(k in model_id.lower() for k in ["whisper", "transcribe"]):
                model_list.append(model_id)

        return model_list

    except openai.AuthenticationError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        return model_list

def get_openai_tts_models(api_key: str = None):
    import openai
    from openai import OpenAI

    model_list = []

    if not api_key:
        model_list.append("OpenAI API Key가 필요합니다.")
        return model_list

    client = OpenAI(api_key=api_key)

    try:
        model = client.models.list()

        for m in model.data:
            model_id = m.id

            if any(k in model_id.lower() for k in ["tts"]):
                model_list.append(model_id)

        return model_list

    except openai.AuthenticationError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        return model_list

def get_openai_audio_models(api_key: str = None):
    import openai
    from openai import OpenAI

    model_list = []

    if not api_key:
        model_list.append("OpenAI API Key가 필요합니다.")
        return model_list

    client = OpenAI(api_key=api_key)

    try:
        model = client.models.list()

        for m in model.data:
            model_id = m.id

            if any(k in model_id.lower() for k in ["audio"]):
                model_list.append(model_id)

        return model_list

    except openai.AuthenticationError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        return model_list

def get_openai_realtime_models(api_key: str = None):
    import openai
    from openai import OpenAI

    model_list = []

    if not api_key:
        model_list.append("OpenAI API Key가 필요합니다.")
        return model_list

    client = OpenAI(api_key=api_key)

    try:
        model = client.models.list()

        for m in model.data:
            model_id = m.id

            if any(k in model_id.lower() for k in ["realtime"]):
                model_list.append(model_id)

        return model_list

    except openai.AuthenticationError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        return model_list

def get_google_genai_tts_models(api_key: str = None):
    from google import genai
    from google.api_core import exceptions

    model_list = []

    if not api_key:
        model_list.append("Google AI API Key가 필요합니다.")
        return model_list

    client = genai.Client(api_key=api_key)

    try:
        model = client.models.list()

        for m in model.page:
            include = any(k in m.name.lower() for k in ["tts"])

            if include:
                model_list.append(m.name)

        return model_list

    except exceptions.Unauthenticated as e:
        model_list.append(f"Google AI API 오류 발생: {e}")
        return model_list