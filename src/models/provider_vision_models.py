import os
import requests
from typing import List
import re
from ..common.environ_manager import load_env_variables

def get_comfyui_image_models(url: str="127.0.0.1:8000", folder: str = "checkpoints"):
    from comfy_sdk import ComfyUI
    client = ComfyUI(host=url.split(":")[0], port=int(url.split(":")[1]))
    model_list = []

    try:
        model = client.models.list(folder=folder)

        if len(model) == 0:
            return ["모델이 존재하지 않습니다."]

        for m in model:
            model_list.append(m)

        return model_list

    except:
        return ["ComfyUI를 설치하고 서버를 실행해주세요."]

def get_openai_image_models(api_key: str = None):
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

            if "image" in model_id.lower():
                model_list.append(model_id)

        return model_list

    except openai.AuthenticationError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        return model_list

def get_openai_video_models(api_key: str = None):
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

            if "sora" in model_id.lower():
                model_list.append(model_id)

        return model_list

    except openai.AuthenticationError as e:
        model_list.append(f"OpenAI API 오류 발생: {e}")
        return model_list

def get_google_genai_image_models(api_key: str = None):
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
            include = any(k in m.name.lower() for k in ["imagen"])

            if include:
                model_list.append(m.name)

        return model_list

    except exceptions.Unauthenticated as e:
        model_list.append(f"Google AI API 오류 발생: {e}")
        return model_list

def get_google_genai_video_models(api_key: str = None):
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
            include = any(k in m.name.lower() for k in ["veo"])

            if include:
                model_list.append(m.name)

        return model_list

    except exceptions.Unauthenticated as e:
        model_list.append(f"Google AI API 오류 발생: {e}")
        return model_list


image_api_models = []
video_api_models = []

comfyui_models = get_comfyui_image_models(folder="checkpoints")
comfyui_loras = get_comfyui_image_models(folder='loras')
comfyui_vae = get_comfyui_image_models(folder='vae')
comfyui_controlnet = get_comfyui_image_models(folder='controlnet')
comfyui_clip = get_comfyui_image_models(folder='clip')
comfyui_clip_vision = get_comfyui_image_models(folder='clip_vision')
comfyui_text_encoders = get_comfyui_image_models(folder='text_encoders')
comfyui_embeddings = get_comfyui_image_models(folder='embeddings')
comfyui_diffusion_models = get_comfyui_image_models(folder='diffusion_models')
comfyui_pretrained_models = get_comfyui_image_models(folder='diffusers')
comfyui_inpaint_models = get_comfyui_image_models(folder='inpaint')
comfyui_ipadapter = get_comfyui_image_models(folder='ipadapter')
comfyui_unet = get_comfyui_image_models(folder='unet')

openai_image_api_models = get_openai_image_models(load_env_variables('OPENAI_API_KEY'))
openai_video_api_models = get_openai_video_models(load_env_variables('OPENAI_API_KEY'))

google_genai_image_api_models = get_google_genai_image_models(load_env_variables('GEMINI_API_KEY'))
google_genai_video_api_models = get_google_genai_video_models(load_env_variables('GEMINI_API_KEY'))

image_api_models.extend(openai_image_api_models)
video_api_models.extend(openai_video_api_models)