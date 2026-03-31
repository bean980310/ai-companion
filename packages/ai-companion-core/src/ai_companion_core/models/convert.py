# model_converter.py

import os
import traceback
from ..logging import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, MetalConfig, AwqConfig, GPTQConfig

import gguf

ftype_map: dict[str, gguf.LlamaFileType] = {
    "f32": gguf.LlamaFileType.ALL_F32,
    "f16": gguf.LlamaFileType.MOSTLY_F16,
    "bf16": gguf.LlamaFileType.MOSTLY_BF16,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    "q4_K_M": gguf.LlamaFileType.MOSTLY_Q4_K_M,
    "q4_K_S": gguf.LlamaFileType.MOSTLY_Q4_K_S,
    "q4_1": gguf.LlamaFileType.MOSTLY_Q4_1,
    "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    "q5_K_M": gguf.LlamaFileType.MOSTLY_Q5_K_M,
    "q5_K_S": gguf.LlamaFileType.MOSTLY_Q5_K_S,
    "q6_K": gguf.LlamaFileType.MOSTLY_Q6_K,
    "iq3_M": gguf.LlamaFileType.MOSTLY_IQ3_M,
    "iq3_S": gguf.LlamaFileType.MOSTLY_IQ3_S,
    "iq3_XS": gguf.LlamaFileType.MOSTLY_IQ3_XS,
    "iq3_XXS": gguf.LlamaFileType.MOSTLY_IQ3_XXS,
    "iq2_M": gguf.LlamaFileType.MOSTLY_IQ2_M,
    "iq2_S": gguf.LlamaFileType.MOSTLY_IQ2_S,
    "iq2_XS": gguf.LlamaFileType.MOSTLY_IQ2_XS,
    "iq2_XXS": gguf.LlamaFileType.MOSTLY_IQ2_XXS,
    "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
    "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
    "auto": gguf.LlamaFileType.GUESSED,
}


def convert_model_bnb_4bit(model_id: str, output_dir: str, push_to_hub: float = False, qbit=4):
    try:
        quantize_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # 또는 적절한 dtype 사용
            quantization_config=quantize_config,
            device_map="auto",
        )
        model.save_pretrained(output_dir)

        print(f"모델이 성공적으로 8비트로 변환되어 '{output_dir}'에 저장되었습니다.")

        if push_to_hub:
            model_name = f"{model_id.split('/', -1)}-bnb-4bit"
            model.push_to_hub(f"{model_name}")
            print(f"모델이 성공적으로 8비트로 변환되어 '{model_name}'에 푸시되었습니다.")

        return True
    except Exception as e:
        print(f"모델 변환 중 오류 발생: {e}")
        return False


def convert_model_bnb_8bit(model_id: str, output_dir: str, push_to_hub: float = False, qbit=8):
    try:
        quantize_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # 또는 적절한 dtype 사용
            quantization_config=quantize_config,
            device_map="auto",
        )
        model.save_pretrained(output_dir)

        print(f"모델이 성공적으로 8비트로 변환되어 '{output_dir}'에 저장되었습니다.")

        if push_to_hub:
            model_name = f"{model_id.split('/', -1)}-bnb-8bit"
            model.push_to_hub(f"{model_name}")
            print(f"모델이 성공적으로 8비트로 변환되어 '{model_name}'에 푸시되었습니다.")

        return True
    except Exception as e:
        print(f"모델 변환 중 오류 발생: {e}")
        return False


def convert_model_gptq(model_id: str, output_dir: str, push_to_hub: float = False, qbit=4):
    try:
        quantize_config = GPTQConfig(
            bits=qbit,
            dataset="c4",
            tokenizer=AutoTokenizer.from_pretrained(model_id),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # 또는 적절한 dtype 사용
            quantization_config=quantize_config,
            device_map="auto",
        )
        model.save_pretrained(output_dir)

        print(f"모델이 성공적으로 {qbit}비트로 변환되어 '{output_dir}'에 저장되었습니다.")

        if push_to_hub:
            model_name = f"{model_id.split('/', -1)}-gptq-{qbit}bit"
            model.push_to_hub(f"{model_name}")
            print(f"모델이 성공적으로 {qbit}비트로 변환되어 '{model_name}'에 푸시되었습니다.")

        return True
    except Exception as e:
        print(f"모델 변환 중 오류 발생: {e}")
        return False


def convert_model_awq(model_id: str, output_dir: str, push_to_hub: float = False, qbit=4):
    try:
        quantize_config = AwqConfig(
            bits=qbit,
            do_fuse=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # 또는 적절한 dtype 사용
            quantization_config=quantize_config,
            device_map="auto",
        )
        model.save_pretrained(output_dir)

        print(f"모델이 성공적으로 {qbit}비트로 변환되어 '{output_dir}'에 저장되었습니다.")

        if push_to_hub:
            model_name = f"{model_id.split('/', -1)}-awq-{qbit}bit"
            model.push_to_hub(f"{model_name}")
            print(f"모델이 성공적으로 {qbit}비트로 변환되어 '{model_name}'에 푸시되었습니다.")

        return True
    except Exception as e:
        print(f"모델 변환 중 오류 발생: {e}")
        return False


def convert_model_mlx(model_id: str, output_dir: str, push_to_hub: float = False, qbit: int = 4):
    from mlx_lm import convert

    try:
        convert(model_id, output_dir, quantize=True, q_bits=qbit, dtype="bfloat16", upload_repo=push_to_hub)
        print(f"모델이 성공적으로 MLX로 변환되어 '{output_dir}'에 저장되었습니다.")

        return True
    except Exception as e:
        print(f"모델 변환 중 오류 발생: {e}")
        return False


def convert_model_mlx_vlm(model_id: str, output_dir: str, push_to_hub: float = False, qbit: int = 4):
    from mlx_vlm import convert

    try:
        convert(model_id, output_dir, quantize=True, q_bits=qbit, dtype="bfloat16", upload_repo=push_to_hub)
        print(f"모델이 성공적으로 MLX로 변환되어 '{output_dir}'에 저장되었습니다.")

        return True
    except Exception as e:
        print(f"모델 변환 중 오류 발생: {e}")
        return False


def convert_model_metal(model_id: str, output_dir: str, push_to_hub: float = False, qbit: int = 4):
    try:
        quantize_config = MetalConfig(
            bits=qbit,
            group_size=64,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # 또는 적절한 dtype 사용
            quantization_config=quantize_config,
            device_map="mps",
        )
        model.save_pretrained(output_dir)

        print(f"모델이 성공적으로 {qbit}비트로 변환되어 '{output_dir}'에 저장되었습니다.")

        if push_to_hub:
            model_name = f"{model_id.split('/', -1)}-metal-{qbit}bit"
            model.push_to_hub(f"{model_name}")
            print(f"모델이 성공적으로 {qbit}비트로 변환되어 '{model_name}'에 푸시되었습니다.")

        return True
    except Exception as e:
        print(f"모델 변환 중 오류 발생: {e}")
        return False
