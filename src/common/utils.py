# utils.py

import os
import shutil
import logging
import asyncio
import traceback
import torch
from pathlib import Path
from typing import Optional, Callable
from huggingface_hub import (
    HfApi, 
    snapshot_download,
    model_info,
    login
)
from llama_cpp import Llama
from model_converter import convert_model_to_float8, convert_model_to_int8, convert_model_to_int4
import platform
import gc
from src.common.cache import models_cache
logger = logging.getLogger(__name__)

LOCAL_MODELS_ROOT = "./models"
comfyui_path=os.path.join(os.path.expanduser('~'), 'ComfyUI', 'models')
class DownloadTracker:
    """다운로드 진행 상황을 추적하는 클래스"""
    def __init__(self, total_size: int, progress_callback: Optional[Callable] = None):
        self.total_size = total_size
        self.current_size = 0
        self.progress_callback = progress_callback

    def update(self, chunk_size: int):
        """다운로드된 청크 크기만큼 진행률 업데이트"""
        self.current_size += chunk_size
        if self.progress_callback:
            progress = (self.current_size / self.total_size) if self.total_size > 0 else 0
            self.progress_callback(min(progress, 1.0))

def make_local_dir_name(model_id: str) -> str:
    """HuggingFace 모델 ID를 로컬 디렉토리 이름으로 변환"""
    return model_id.replace("/", "__")

def convert_folder_to_modelid(folder_name: str) -> str:
    """로컬 디렉토리 이름을 HuggingFace 모델 ID로 변환"""
    return folder_name.replace("__", "/")

def has_required_files(filenames, required_names=None, required_extensions=None):
    """
    파일 목록(filenames) 중에 필수 파일명이 존재하거나,
    필수 확장자(extension)를 가진 파일이 존재하는지 확인.
    기본적으로 config.json이나 (.safetensors, .ckpt, .pt, .pth) 파일을 체크.
    """
    if required_names is None:
        required_names = {"config.json"}
    if required_extensions is None:
        required_extensions = {".safetensors", ".ckpt", ".pt", ".pth"}
    for fname in filenames:
        lower_fname = fname.lower()
        if lower_fname in required_names:
            return True
        for ext in required_extensions:
            if lower_fname.endswith(ext):
                return True
    return False

def scan_files_with_extension(root: str, allowed_extensions: set) -> list:
    """
    지정된 root 디렉토리부터 재귀적으로 파일을 탐색하여,
    allowed_extensions에 해당하는 파일들의 상대경로 목록을 반환.
    """
    results = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in allowed_extensions):
                # root로부터의 상대경로 (하위 디렉토리 포함)
                rel_path = os.path.relpath(os.path.join(dirpath, fname), root)
                results.append(rel_path)
    return results

def scan_diffusion_models(root="./models/diffusion"):
    """
    diffusion 폴더 내의 모델들을 재귀적으로 스캔.
    - diffusers 모델은 기존처럼 폴더(디렉토리) 단위로 모델을 표시
    - 그 외(예: checkpoints, clip, controlnet, vae 등)는 폴더 내의 모델 파일(확장자를 가진 파일)을
      하위 디렉토리 구조와 함께 표시
    """
    if not os.path.isdir(root):
        os.makedirs(root, exist_ok=True)
    
    local_models = []
    # 모델 타입 디렉토리 목록
    model_types = ['checkpoints', 'clip', 'configs', 'controlnet', 'diffusers', 'embeddings', 'loras', 'unet', 'vae', 'vae_approx']
    allowed_exts = {".safetensors", ".ckpt", ".pt", ".pth"}
    
    for mtype in model_types:
        mtype_path = os.path.join(root, mtype)
        if not os.path.isdir(mtype_path):
            continue

        if mtype == "diffusers":
            # diffusers는 기존대로 디렉토리 단위로 스캔
            for dirpath, _, filenames in os.walk(mtype_path):
                if has_required_files(filenames):
                    rel_path = os.path.relpath(dirpath, mtype_path)
                    model_id = mtype if rel_path == "." else os.path.join(mtype, rel_path)
                    local_models.append({"model_id": model_id, "model_type": mtype})
        else:
            # diffusers를 제외한 나머지는 파일 단위로 스캔하여 모델 파일 자체를 목록에 표시
            files = scan_files_with_extension(mtype_path, allowed_exts)
            for f in files:
                # 모델 아이디에 mtype 디렉토리 이름을 접두어로 붙임
                model_id = os.path.join(mtype, f)
                local_models.append({"model_id": model_id, "model_type": mtype})
    
    return local_models

def scan_comfyui_models(root="./comfyui/models"):
    """
    comfyui/models 디렉토리 내의 모델들을 재귀적으로 스캔.
    해당 디렉토리 내에서도, allowed_extensions에 해당하는 파일이 있으면 모델로 인식.
    모델 아이디는 comfyui_models/하위경로 형태로 반환.
    """
    if not os.path.isdir(root):
        return []
    
    local_models = []
    allowed_exts = {".safetensors", ".ckpt", ".pt", ".pth"}
    
    for dirpath, _, filenames in os.walk(root):
        if has_required_files(filenames, required_extensions=allowed_exts):
            rel_path = os.path.relpath(dirpath, root)
            # rel_path가 '.'이면 root 자체를 의미하므로 comfyui_models 로 지정
            model_id = "comfyui_models" if rel_path == "." else os.path.join("comfyui_models", rel_path)
            # model_type을 별도로 'comfyui_models'로 표기
            local_models.append({"model_id": model_id, "model_type": "comfyui_models"})
    
    return local_models
    
def scan_local_models(root="./models/llm", model_type=None):
    """
    로컬 llm 모델들을 스캔.
    - model_type이 지정되지 않으면, 'transformers', 'gguf', 'mlx' 폴더를 각각 스캔.
    - gguf의 경우 gguf 파일 자체를, 그 외에는 디렉토리 단위(필요한 파일이 있는 경우)로 등록.
    """
    if not os.path.isdir(root):
        os.makedirs(root, exist_ok=True)
    
    local_models = []
    # model_type이 지정되지 않았다면, 세부 폴더 목록: transformers, gguf, mlx
    subdirs = ['transformers', 'gguf', 'mlx'] if model_type is None else [model_type]
    
    for subdir in subdirs:
        subdir_path = os.path.join(root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        if subdir == "gguf":
            # gguf는 파일 단위로 스캔 (확장자가 .gguf 인 파일)
            files = scan_files_with_extension(subdir_path, {".gguf"})
            for f in files:
                model_id = os.path.join(subdir, f)
                local_models.append({"model_id": model_id, "model_type": subdir})
        else:
            # transformers와 mlx는 기존대로, 해당 폴더 내에 필요한 파일이 있으면 디렉토리 단위로 등록
            for dirpath, _, filenames in os.walk(subdir_path):
                if has_required_files(filenames, required_extensions={".safetensors", ".ckpt", ".pt", ".pth"}):
                    rel_path = os.path.relpath(dirpath, subdir_path)
                    model_id = subdir if rel_path == "." else os.path.join(subdir, rel_path)
                    local_models.append({"model_id": model_id, "model_type": subdir})
    
    logger.info(f"Scanned local models: {local_models}")
    return local_models

def get_all_loras(lora_root="./models/llm/loras"):
    """
    lora 폴더 내에서, 확장자가 .safetensors, .bin, .pt, .pth 인 파일이 존재하는 폴더를
    재귀적으로 스캔해서 목록에 추가.
    """
    if not os.path.isdir(lora_root):
        os.makedirs(lora_root, exist_ok=True)
    lora_models = ["None"]
    allowed_extensions = {".safetensors", ".bin", ".pt", ".pth"}
    for dirpath, _, filenames in os.walk(lora_root):
        if any(fname.lower().endswith(ext) for fname in filenames for ext in allowed_extensions):
            rel_path = os.path.relpath(dirpath, lora_root)
            # 상대경로가 '.'이면 현재 폴더의 이름를 사용
            model_id = os.path.basename(dirpath) if rel_path == "." else rel_path
            lora_models.append(model_id)
    return lora_models

def get_diffusion_vae(vae_root="./models/diffusion/vae"):
    """
    diffusion/vae 폴더 내에서 필요한 파일들이 있는 폴더를 재귀적으로 스캔하여 VAE 모델 목록 반환.
    """
    models = scan_diffusion_models()
    comfyui_models = scan_diffusion_models(root=comfyui_path)
    all_models = models+comfyui_models
    vae_models = [m['model_id'] for m in all_models if m['model_type'] == "vae"]
    return vae_models


def get_diffusion_loras(lora_root="./models/diffusion/loras"):
    """
    diffusion/loras 폴더 내에서 필요한 파일들이 있는 폴더를 재귀적으로 스캔하여 LoRA 모델 목록 반환.
    """
    models = scan_diffusion_models()
    comfyui_models = scan_diffusion_models(root=comfyui_path)
    all_models = models+comfyui_models
    lora_models = [m['model_id'] for m in all_models if m['model_type'] == "loras"]
    return lora_models

def get_all_local_models():
    """모든 모델 유형별 로컬 모델 목록을 가져옴"""
    models = scan_local_models()  # 모든 유형 스캔
    transformers = [m["model_id"] for m in models if m["model_type"] == "transformers"]
    gguf = [m["model_id"] for m in models if m["model_type"] == "gguf"]
    mlx = [m["model_id"] for m in models if m["model_type"] == "mlx"]
    return {
        "transformers": transformers,
        "gguf": gguf,
        "mlx": mlx
    }

def get_all_diffusion_models():
    models = scan_diffusion_models()
    comfyui_models = scan_diffusion_models(root=comfyui_path)
    all_models=models+comfyui_models
    diffusers=[m['model_id'] for m in all_models if m['model_type'] == "diffusers"]
    checkpoints=[m['model_id'] for m in all_models if m['model_type'] == 'checkpoints']
    return {
        "diffusers": diffusers,
        "checkpoints": checkpoints
    }
    
def remove_hf_cache(model_id):
    """HuggingFace 캐시 폴더 제거"""
    if "/" in model_id:
        user, name = model_id.split("/", maxsplit=1)
        cache_dirname = f"./models/llm/{user}__{name}"
    else:
        cache_dirname = f"./models/llm/{model_id}"

    cache_path = os.path.join(cache_dirname, ".cache")
    if os.path.isdir(cache_path):
        logger.info(f"[*] Removing cache folder: {cache_path}")
        shutil.rmtree(cache_path)
    else:
        logger.info(f"[*] No cache folder found: {cache_path}")

async def get_model_size(repo_id: str, token: Optional[str] = None) -> int:
    """모델의 총 크기를 계산"""
    try:
        api = HfApi()
        info = await asyncio.to_thread(model_info, repo_id, token=token)
        return info.size_in_bytes or 0
    except Exception as e:
        logger.warning(f"모델 크기 계산 실패: {e}")
        return 0
    
def get_model_list_from_hf_hub():
    api=HfApi()
    api.token = login(new_session=False)
    api.list_models(sort="lastModified", direction=-1, limit=100)

def download_model_from_hf(hf_repo_id: str, target_dir: str, model_type: str = "transformers", quantization_bit: str = None) -> str:
    """
    동기식 모델 다운로드
    model_type: "transformers", "gguf", "mlx" 중 선택
    """
    if model_type not in ["transformers", "gguf", "mlx"]:
        model_type = "transformers"  # 기본값 설정

    target_base_dir = os.path.join("./models/llm", model_type)
    os.makedirs(target_base_dir, exist_ok=True)
    target_dir = os.path.join(target_base_dir, make_local_dir_name(hf_repo_id))

    if os.path.isdir(target_dir):
        msg = f"[*] 이미 다운로드됨: {hf_repo_id} → {target_dir}"
        logger.info(msg)
        return msg

    logger.info(f"[*] 모델 '{hf_repo_id}'을(를) '{target_dir}'에 다운로드 중...")
    try:
        if model_type=="gguf":
            Llama.from_pretrained(
                repo_id=hf_repo_id,
                filename=f"*{quantization_bit}.gguf",
                local_dir=target_base_dir,
                hf=True
            )
        else:
            snapshot_download(
                repo_id=hf_repo_id,
                local_dir=target_dir,
                ignore_patterns=["*.md", ".gitattributes", "original/", "LICENSE.txt", "LICENSE"],
                local_dir_use_symlinks=False
            )
        remove_hf_cache(hf_repo_id)
        msg = f"[+] 다운로드 & 저장 완료: {target_dir}"
        logger.info(msg)
        return msg
    except Exception as e:
        error_msg = f"모델 '{hf_repo_id}' 다운로드 실패: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return error_msg

async def download_model_from_hf_async(
    repo_id: str,
    target_dir: str = None,
    token: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> str:
    """
    비동기식 모델 다운로드 (진행률 표시 지원)
    """
    try:
        # 기본 저장 경로 설정
        if not target_dir:
            target_dir = os.path.join("./models/llm", make_local_dir_name(repo_id))
            
        # 이미 다운로드된 경우 확인
        if os.path.exists(target_dir) and any(os.scandir(target_dir)):
            return f"모델이 이미 존재합니다: {target_dir}"
            
        # 저장 경로 생성
        os.makedirs(target_dir, exist_ok=True)
        
        # 모델 크기 계산
        total_size = await get_model_size(repo_id, token)
        if total_size == 0:
            logger.warning("모델 크기를 확인할 수 없습니다. 진행률이 정확하지 않을 수 있습니다.")
        
        # 다운로드 트래커 초기화
        tracker = DownloadTracker(total_size, progress_callback)
        
        # 실제 다운로드 수행
        snapshot_path = await asyncio.to_thread(
            snapshot_download,
            repo_id=repo_id,
            local_dir=target_dir,
            token=token,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.md", ".gitattributes", "original/", "LICENSE.txt", "LICENSE"],
        )
        
        # 다운로드 완료 후 정리
        remove_hf_cache(repo_id)
        if progress_callback:
            progress_callback(1.0)  # 100% 완료 표시
            
        # 결과 메시지 생성
        model_name = repo_id.split("/")[-1]
        size_gb = total_size / (1024**3)
        message = f"""
            다운로드 완료!
            - 모델: {model_name}
            - 저장 위치: {target_dir}
            - 크기: {size_gb:.2f} GB
            """
        logger.info(f"모델 다운로드 완료: {repo_id}")
        return message.strip()
        
    except Exception as e:
        error_msg = f"다운로드 중 오류 발생: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise RuntimeError(error_msg)

def ensure_model_available(model_id, local_model_path=None, model_type=None):
    """
    모델의 가용성을 확인하고, 필요 시 다운로드합니다.
    
    Args:
        model_id (str): 모델의 ID.
        local_model_path (str, optional): 모델이 저장된 로컬 경로.
        model_type (str, optional): 모델 유형 (예: 'transformers', 'gguf', 'mlx').
    
    Returns:
        bool: 모델이 사용 가능하면 True, 아니면 False.
    """
    # model_type을 사용하여 모델 경로 결정 또는 다운로드 로직 수정
    if model_type:
        model_dir = os.path.join("./models/llm", model_type, make_local_dir_name(model_id))
    else:
        model_dir = os.path.join("./models/llm", "transformers", make_local_dir_name(model_id))
    
    if not os.path.exists(model_dir):
        try:
            download_model_from_hf(model_id, model_dir, model_type=model_type)
            return True
        except Exception as e:
            logger.error(f"모델 다운로드 실패: {e}")
            return False
    return True
    
def get_terminators(tokenizer):
    """
    모델별 종료 토큰 ID를 반환하는 함수
    """
    if "glm" in str(tokenizer.__class__).lower():
        # GLM 모델용 특수 처리
        return [tokenizer.eos_token_id]  # GLM의 EOS 토큰 사용
    else:
        # 기존 다른 모델들을 위한 처리
        return [
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
        ]
        
def convert_and_save(model_id, output_dir, push_to_hub, quant_type, model_type="transformers"):
    if not model_id:
        return "모델 ID를 입력해주세요."

    base_output_dir = os.path.join("./models/llm", model_type)
    os.makedirs(base_output_dir, exist_ok=True)

    if quant_type == 'float8':
        if not output_dir:
            output_dir = os.path.join(base_output_dir, f"{model_id.replace('/', '__')}-float8")
        if platform.system() == 'Darwin':
            return "MacOS에서는 float8 변환을 지원하지 않습니다."
        else:
            success = convert_model_to_float8(model_id, output_dir, push_to_hub)
            if success:
                return f"모델이 성공적으로 8비트로 변환되었습니다: {output_dir}"
            else:
                return "모델 변환에 실패했습니다."
    elif quant_type == 'int8':
        if not output_dir:
            output_dir = os.path.join(base_output_dir, f"{model_id.replace('/', '__')}-int8")
        success = convert_model_to_int8(model_id, output_dir, push_to_hub)
        if success:
            return f"모델이 성공적으로 8비트로 변환되었습니다: {output_dir}"
        else:
            return "모델 변환에 실패했습니다."
    elif quant_type == 'int4':
        if not output_dir:
            output_dir = os.path.join(base_output_dir, f"{model_id.replace('/', '__')}-int8")
        success = convert_model_to_int4(model_id, output_dir, push_to_hub)
        if success:
            return f"모델이 성공적으로 4비트로 변환되었습니다: {output_dir}"
        else:
            return "모델 변환에 실패했습니다."
    else:
        return "지원되지 않는 변환 유형입니다."
    
def build_model_cache_key(model_id: str, model_type: str, lora_model_id: str = None, quantization_bit: str = None, local_path: str = None) -> str:
    """
    models_cache에 사용될 key를 구성.
    - 만약 model_id == 'Local (Custom Path)' 이고 local_path가 주어지면 'local::{local_path}'
    - 그 외에는 'auto::{model_type}::{local_dir}::hf::{model_id}::{quantization_bit}' 형태.
    """
    if model_id == "Local (Custom Path)" and local_path:
        return f"local::{local_path}"
    elif model_type == "api":
        return f"api::{model_id}"
    else:
        local_dirname = make_local_dir_name(model_id)
        local_dirpath = os.path.join("./models/llm", model_type, local_dirname)
        if lora_model_id:
            lora_dirname = make_local_dir_name(lora_model_id)
            lora_dirpath = os.path.join("./models/llm/loras", lora_dirname)
            unique_loras=sorted(set(lora_dirpath))
            lora_part = "::".join(unique_loras)
            if quantization_bit:
                return f"auto::{model_type}::{local_dirpath}::hf::{model_id}::{quantization_bit}::lora::{lora_part}"
            else:
                return f"auto::{model_type}::{local_dirpath}::hf::{model_id}::lora::{lora_part}"
        else:
            if quantization_bit:
                return f"auto::{model_type}::{local_dirpath}::hf::{model_id}::{quantization_bit}"
            else:
                return f"auto::{model_type}::{local_dirpath}::hf::{model_id}"

def clear_model_cache(model_id: str, local_path: str = None) -> str:
    """
    특정 모델에 대한 캐시를 제거 (models_cache에서 해당 key를 삭제).
    - 만약 해당 key가 없으면 '이미 없음' 메시지 반환
    - 성공 시 '캐시 삭제 완료' 메시지
    """
    # 모델 유형을 결정해야 합니다.
    if model_id in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]:
        model_type = "api"
    else:
        # 로컬 모델의 기본 유형을 transformers로 설정 (필요 시 수정)
        model_type = "transformers"
    key = build_model_cache_key(model_id, model_type, local_path)
    if key in models_cache:
        del models_cache[key]
        msg = f"[cache] 모델 캐시 제거: {key}"
        logger.info(msg)
        return msg
    else:
        msg = f"[cache] 이미 캐시에 없거나, 로드된 적 없음: {key}"
        logger.info(msg)
        return msg

def clear_all_model_cache():
    """
    현재 메모리에 로드된 모든 모델 캐시(models_cache)를 한 번에 삭제.
    필요하다면, 로컬 폴더의 .cache들도 일괄 삭제할 수 있음.
    """
    for key, handler in list(models_cache.items()):
        # 혹시 model, tokenizer 등 메모리를 점유하는 속성이 있으면 제거
        if hasattr(handler, "model"):
            del handler.model
        if hasattr(handler, "tokenizer"):
            del handler.tokenizer
        # 필요 시 handler 내부의 다른 자원들(예: embeddings 등)도 정리
        
    # 1) 메모리 캐시 전부 삭제
    count = len(models_cache)
    models_cache.clear()
    logger.info(f"[*] 메모리 캐시 삭제: {count}개 모델")

    # 2) (선택) 로컬 폴더 .cache 삭제
    #    예: ./models/*/.cache 폴더 전부 삭제
    #    원치 않으면 주석처리
    
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    gc.collect()
        
    cache_deleted = 0
    for subdir, models in get_all_local_models().items():
        for folder in models:
            folder_path = os.path.join('./models/llm', subdir, folder)
            if os.path.isdir(folder_path):
                cache_path = os.path.join(folder_path, ".cache")
                if os.path.isdir(cache_path):
                    shutil.rmtree(cache_path)
                    cache_deleted += 1
    logger.info(f"[*] 로컬 폴더 .cache 삭제: {cache_deleted}개 폴더 삭제")
    return f"[cache all] {count}개 모델 캐시 삭제 완료. 로컬 폴더 .cache {cache_deleted}개 삭제."