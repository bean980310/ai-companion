# models.py

import random
import platform
import numpy as np
import torch
from src.common.cache import models_cache
from src.model_handlers import (
    GGUFModelHandler, MiniCPMLlama3V25Handler, Llama3Handler, GLM4Handler, GLM4VHandler, Llama3VisionModelHandler,
    Aya23Handler, GLM4HfHandler, OtherModelHandler, Qwen2Handler, MlxModelHandler, MlxVisionHandler,
    TransformersCausalModelHandler, TransformersVisionModelHandler, GGUFCausalModelHandler, MlxCausalModelHandler, MlxVisionModelHandler
)
from src.common.utils import ensure_model_available, build_model_cache_key, get_all_local_models, convert_folder_to_modelid
import gradio as gr
from src.models import api_models

from peft import PeftModel, PeftConfig

import traceback
import openai
import anthropic
from google import genai
from google.genai import types

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_ollama import OllamaLLM as Ollama
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace

from src import logger

LOCAL_MODELS_ROOT = "./models"


def get_default_device():
    """
    Automatically selects the best available device:
    - CUDA if NVIDIA GPU is available.
    - MPS if Apple Silicon (M-Series) is available.
    - CPU otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
# Set default device
default_device = get_default_device()
logger.info(f"Default device set to: {default_device}")
def refresh_model_list():
    new_local_models = get_all_local_models()
    global api_models
    local_models = (
        new_local_models["transformers"] + 
        new_local_models["gguf"] + 
        new_local_models["mlx"]
    )
    new_choices = api_models + local_models
    new_choices = sorted(list(dict.fromkeys(new_choices)))
    return gr.update(choices=new_choices), "모델 목록을 새로고침했습니다."


def load_model(selected_model, model_type, selected_lora=None, quantization_bit="Q8_0", local_model_path=None, api_key=None, device="cpu", lora_path=None):
    """
    모델 로드 함수. 특정 모델에 대한 로드 로직을 외부 핸들러로 분리.
    """
    model_id = selected_model
    lora_model_id = selected_lora if selected_lora else None
    if model_type != "transformers" and model_type != "gguf" and model_type != "mlx" and model_type != "api":
        logger.error(f"지원되지 않는 모델 유형: {model_type}")
        return None
    
    # Pass the device to the handler
    handler = None
    if model_type not in ["transformers", "gguf", "mlx", "api"]:
        logger.error(f"지원되지 않는 모델 유형: {model_type}")
        return None
    if model_type == "api":
        # API 모델은 별도의 로드가 필요 없으므로 핸들러 생성 안함
        return None
    if model_type == "gguf":
        # GGUF 모델 로딩 로직
        # handler = GGUFModelHandler(
        #     model_id=model_id,
        #     local_model_path=local_model_path,
        #     model_type=model_type
        # )
        handler = GGUFCausalModelHandler(
            model_id=model_id,
            lora_model_id=lora_model_id,
            model_type=model_type,
            device=device
        )
        cache_key = build_model_cache_key(model_id, model_type)
        models_cache[cache_key] = handler
        return handler
    elif model_type == "mlx":
        if "vision" in model_id.lower() or "qwen2-vl" in model_id.lower() or "qwen2.5-vl" in model_id.lower():
            # handler = MlxVisionHandler(
            #     model_id=model_id,  # model_id가 정의되어 있어야 합니다.
            #     lora_model_id=lora_model_id,
            #     local_model_path=local_model_path,
            #     lora_path=lora_path,
            #     model_type=model_type,
            # )
            handler = MlxVisionModelHandler(
                model_id=model_id,
                lora_model_id=lora_model_id,
                model_type=model_type
            )
            models_cache[build_model_cache_key(model_id, model_type, lora_model_id)] = handler
            return handler
        else:
            # handler = MlxModelHandler(
            #     model_id=model_id,  # model_id가 정의되어 있어야 합니다.
            #     lora_model_id=lora_model_id,
            #     local_model_path=local_model_path,
            #     lora_path=lora_path,
            #     model_type=model_type
            # )
            handler = MlxCausalModelHandler(
                model_id=model_id,
                lora_model_id=lora_model_id,
                model_type=model_type
            )
            models_cache[build_model_cache_key(model_id, model_type, lora_model_id)] = handler
            return handler
    else:
        if "vision" in model_id.lower() or "qwen2-vl" in model_id.lower() or "qwen2.5-vl" in model_id.lower():
            handler = TransformersVisionModelHandler(
                model_id=model_id,
                lora_model_id=lora_model_id,
                model_type=model_type,
                device=device
            )
            models_cache[build_model_cache_key(model_id, model_type, lora_model_id)] = handler
            return handler
        else:
            handler = TransformersCausalModelHandler(
                model_id=model_id, 
                lora_model_id=lora_model_id, 
                model_type=model_type,
                device=device)
            models_cache[build_model_cache_key(model_id, model_type, lora_model_id)] = handler
            return handler

def generate_answer(history, selected_model, model_type, selected_lora=None, local_model_path=None, lora_path=None, image_input=None, api_key=None, device="cpu", seed=42, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0, character_language='ko'):
    """
    사용자 히스토리를 기반으로 답변 생성.
    """
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    else:
        torch.manual_seed(seed)
        
    if not history:
        system_message = {
            "role": "system",
            "content": "당신은 유용한 AI 비서입니다."
        }
        history = [system_message]
    
    cache_key = build_model_cache_key(selected_model, model_type, selected_lora, local_path=local_model_path)
    handler = models_cache.get(cache_key)
    
    last_message = history[-1]
    if last_message["role"] == "assistant":
        last_message = history[-2]
    
    # if character_language == "ko":
    #     # 한국어 모델 사용
    #     model_id = "klue/bert-base"  # 예시
    # elif character_language == "en":
    #     # 영어 모델 사용
    #     model_id = "bert-base-uncased"  # 예시
    # elif character_language == "ja":
    #     # 일본어 모델 사용
    #     model_id = "tohoku-nlp/bert-base-japanese"  # 예시"
    # else:
    #     # 기본 모델 사용
    #     model_id = "gpt-3.5-turbo"
    
    if model_type == "api":
        if "claude" in selected_model:
            if not api_key:
                logger.error("Anthropic API Key가 missing.")
                return "Anthropic API Key가 필요합니다."
            
            client = anthropic.Client(api_key=api_key)
            # Anthropic 메시지 형식으로 변환
            messages = []
            for msg in history:
                if msg["role"] == "system":
                    continue  # Claude API는 시스템 메시지를 별도로 처리하지 않음
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            logger.info(f"[*] Anthropic API 요청: {messages}")
            
            try:
                response = client.messages.create(
                    model=selected_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024
                )
                answer = response.content[0].text
                logger.info(f"[*] Anthropic 응답: {answer}")
                return answer
            except Exception as e:
                logger.error(f"Anthropic API 오류: {str(e)}\n\n{traceback.format_exc()}")
                return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
        elif "gemini" in selected_model:
            if not api_key:
                logger.error("Google API Key가 missing.")
                return "Google API Key가 필요합니다."

            client = genai.Client(api_key=api_key)
            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            config = types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.7,
                top_p=0.9
            )
            logger.info(f"[*] Google API 요청: {messages}")
            try: 
                response = client.models.generate_content(
                    model=selected_model,
                    contents=messages,
                    config=config
                )
                answer = response.text
                logger.info(f"[*] Google 응답: {answer}")
                return answer
            except Exception as e:
                logger.error(f"Google API 오류: {str(e)}\n\n{traceback.format_exc()}")
                return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
        else:
            if not api_key:
                logger.error("OpenAI API Key가 missing.")
                return "OpenAI API Key가 필요합니다."
            openai.api_key = api_key
            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] OpenAI API 요청: {messages}")
            
            try:
                response = openai.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.9,
                )
                answer = response.choices[0].message["content"]
                logger.info(f"[*] OpenAI 응답: {answer}")
                return answer
            except Exception as e:
                logger.error(f"OpenAI API 오류: {str(e)}\n\n{traceback.format_exc()}")
                return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
    
    else:
        if not handler:
            logger.info(f"[*] 모델 로드 중: {selected_model}")
            handler = load_model(selected_model, model_type, selected_lora, local_model_path=local_model_path, device=device, lora_path=lora_path)
        
        if not handler:
            logger.error("모델 핸들러가 로드되지 않았습니다.")
            return "모델 핸들러가 로드되지 않았습니다."
        
        logger.info(f"[*] Generating answer using {handler.__class__.__name__}")
        try:
            if isinstance(handler, TransformersVisionModelHandler) or isinstance(handler, MlxVisionModelHandler):
                answer = handler.generate_answer(history, image_input, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
            else:
                answer = handler.generate_answer(history, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
            return answer
        except Exception as e:
            logger.error(f"모델 추론 오류: {str(e)}\n\n{traceback.format_exc()}")
            return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
        
def generate_chat_title(first_message, selected_model, model_type, selected_lora=None, local_model_path=None, lora_path=None, device="cpu"):
    """
    첫 번째 메시지를 기반으로 채팅 제목을 생성하는 함수.
    모델 핸들러에 generate_chat_title 메서드가 구현되어 있어야 함.
    """
    cache_key = build_model_cache_key(selected_model, model_type, selected_lora, local_path=local_model_path)
    handler = models_cache.get(cache_key)
    
    if not handler:
        logger.info(f"[*] 모델 로드 중: {selected_model}")
        handler = load_model(selected_model, model_type, selected_lora, local_model_path=local_model_path, device=device, lora_path=lora_path)
    
    if not handler:
        logger.error("모델 핸들러가 로드되지 않았습니다.")
        return "모델 핸들러가 로드되지 않았습니다."
    
    logger.info(f"[*] Generating chat title using {handler.__class__.__name__}")
    try:
        if hasattr(handler, "generate_chat_title"):
            title = handler.generate_chat_title(first_message)
            return title
        else:
            logger.error("모델 핸들러에 채팅 제목 생성 기능이 없습니다.")
            return "채팅 제목 생성 기능이 없습니다."
    except Exception as e:
        logger.error(f"채팅 제목 생성 오류: {str(e)}\n\n{traceback.format_exc()}")
        return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
        
# models.py

def generate_stable_diffusion_prompt_cached(user_input, selected_model, model_type, local_model_path=None, api_key=None, device="cpu", seed=42):
    """
    사용자 입력을 기반으로 Stable Diffusion 프롬프트를 생성합니다.
    API 모델과 로컬 모델을 모두 지원합니다.
    """
    try:
        prompt_template = """
        당신은 이미지 생성에 최적화된 프롬프트를 생성하는 AI입니다.
        다음 설명을 바탕으로 상세한 Stable Diffusion 프롬프트를 작성해주세요:
        
        설명: {description}
        
        프롬프트:
        """
        
        template = PromptTemplate(
            input_variables=["description"],
            template=prompt_template
        )
        
        if model_type == "api":
            # API 모델을 사용하는 경우
            if not api_key:
                return "❌ OpenAI API Key가 필요합니다."
            llm = OpenAI(api_key=api_key, model=selected_model, temperature=0.7)
        elif model_type == "transformers":
            # Transformers 로컬 모델을 사용하는 경우
            if not local_model_path:
                return "❌ 로컬 모델 경로가 필요합니다."
            from transformers import pipeline
            # HuggingFace Pipeline 초기화 (예시)
            try:
                hf_pipeline = pipeline("text-generation", model=local_model_path)
                llm = HuggingFacePipeline(pipeline=hf_pipeline)
            except Exception as e:
                logger.error(f"Transformers 로컬 모델 초기화 오류: {e}")
                return f"❌ Transformers 로컬 모델 초기화 오류: {e}"
        elif model_type == "gguf":
            # GGUF 로컬 모델을 사용하는 경우 (예시: LlamaCpp)
            if not local_model_path:
                return "❌ 로컬 모델 경로가 필요합니다."
            try:
                llm = LlamaCpp(model_path=local_model_path, n_ctx=512, n_batch=64)
            except Exception as e:
                logger.error(f"GGUF 로컬 모델 초기화 오류: {e}")
                return f"❌ GGUF 로컬 모델 초기화 오류: {e}"
        elif model_type == "mlx":
            # MLX 로컬 모델을 사용하는 경우 (예시: Ollama)
            if not local_model_path:
                return "❌ 로컬 모델 경로가 필요합니다."
            try:
                llm = Ollama(model=selected_model, model_path=local_model_path)
            except Exception as e:
                logger.error(f"MLX 로컬 모델 초기화 오류: {e}")
                return f"❌ MLX 로컬 모델 초기화 오류: {e}"
        else:
            return "❌ 지원되지 않는 모델 유형입니다."
        
        # LLMChain 생성
        # Runnable 인스턴스 확인 대신, LLMChain이 올바르게 초기화되는지 확인
        if not llm:
            return "❌ LLM 인스턴스가 초기화되지 않았습니다."
        
        chain = LLMChain(llm=llm, prompt=template)
        prompt = chain.run(description=user_input)
        
        return prompt
    except Exception as e:
        logger.error(f"Stable Diffusion 프롬프트 생성 오류: {str(e)}")
        return f"❌ 프롬프트 생성 중 오류가 발생했습니다: {str(e)}"