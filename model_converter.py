# model_converter.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from optimum.quanto import Calibration, QuantizedModelForCausalLM, qfloat8, qint4, qint8

def convert_model_to_float8(model_id: str, output_dir: str, push_to_hub: float=False):
    """
    모델을 8비트로 변환하여 저장하는 함수
    """
    try:
        # 모델과 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # 또는 적절한 dtype 사용
            device_map="auto",
        )
        qmodel=QuantizedModelForCausalLM.quantize(model, weights=qfloat8, activations=qfloat8)
        qmodel.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"모델이 성공적으로 8비트로 변환되어 '{output_dir}'에 저장되었습니다.")

        if push_to_hub:
            model_name=f"{model_id.split('/', -1)}-float8"
            qmodel.push_to_hub(f"{model_name}")
            print(f"모델이 성공적으로 8비트로 변환되어 '{model_name}'에 푸시되었습니다.")
        
        return True
    except Exception as e:
        print(f"모델 변환 중 오류 발생: {e}")
        return False
    
def convert_model_to_int8(model_id: str, output_dir: str, push_to_hub: float=False):
    """
    모델을 8비트로 변환하여 저장하는 함수
    """
    try:
        # 모델과 토크나이저 로드
        device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # 또는 적절한 dtype 사용
            trust_remote_code=True
        ).to(device)
        qmodel=QuantizedModelForCausalLM.quantize(model, weights=qint8, activations=qint8)
        qmodel.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"모델이 성공적으로 8비트로 변환되어 '{output_dir}'에 저장되었습니다.")

        if push_to_hub:
            model_name=f"{model_id.split('/', -1)}"
            qmodel.push_to_hub(f"{model_name}")
            print(f"모델이 성공적으로 8비트로 변환되어 '{model_name}'에 푸시되었습니다.")
        
        return True
    except Exception as e:
        print(f"모델 변환 중 오류 발생: {e}")
        return False
    
def convert_model_to_int4(model_id: str, output_dir: str, push_to_hub: float=False):
    """
    모델을 8비트로 변환하여 저장하는 함수
    """
    try:
        # 모델과 토크나이저 로드
        device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # 또는 적절한 dtype 사용
            trust_remote_code=True
        ).to(device)
        qmodel=QuantizedModelForCausalLM.quantize(model, weights=qint4, activations=qint4)
        qmodel.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"모델이 성공적으로 8비트로 변환되어 '{output_dir}'에 저장되었습니다.")

        if push_to_hub:
            model_name=f"{model_id.split('/', -1)}"
            qmodel.push_to_hub(f"{model_name}")
            print(f"모델이 성공적으로 8비트로 변환되어 '{model_name}'에 푸시되었습니다.")
        
        return True
    except Exception as e:
        print(f"모델 변환 중 오류 발생: {e}")
        return False