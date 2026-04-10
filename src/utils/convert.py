from __future__ import annotations

import os
from ai_companion_core.models.convert import convert_model_bnb_4bit, convert_model_bnb_8bit, convert_model_gptq, convert_model_awq, convert_model_mlx, convert_model_mlx_vlm, convert_model_metal


def convert_and_save(model_id, output_dir, push_to_hub, quant_type, qbit, model_type="transformers"):
    if not model_id:
        return "모델 ID를 입력해주세요."

    base_output_dir = os.path.join("./models/llm", model_type)
    os.makedirs(base_output_dir, exist_ok=True)

    if quant_type == "bnb_4bit":
        if qbit:
            print(f"bnb_4bit 이므로 {qbit}는 무시됨.")
        if not output_dir:
            output_dir = os.path.join(base_output_dir, f"{model_id.replace('/', '__')}-bnb-4bit")
        else:
            success = convert_model_bnb_4bit(model_id, output_dir, push_to_hub)
            if success:
                return f"모델이 성공적으로 4비트로 변환되었습니다: {output_dir}"
            else:
                return "모델 변환에 실패했습니다."
    elif quant_type == "bnb_8bit":
        if qbit:
            print(f"bnb_8bit 이므로 {qbit}는 무시됨.")
        if not output_dir:
            output_dir = os.path.join(base_output_dir, f"{model_id.replace('/', '__')}-bnb-8bit")
        success = convert_model_bnb_8bit(model_id, output_dir, push_to_hub)
        if success:
            return f"모델이 성공적으로 8비트로 변환되었습니다: {output_dir}"
        else:
            return "모델 변환에 실패했습니다."
    elif quant_type == "gptq":
        if not output_dir:
            output_dir = os.path.join(base_output_dir, f"{model_id.replace('/', '__')}-gptq-{qbit}bit")
        success = convert_model_gptq(model_id, output_dir, push_to_hub, qbit)
        if success:
            return f"모델이 성공적으로 변환되었습니다: {output_dir}"
        else:
            return "모델 변환에 실패했습니다."
    elif quant_type == "awq":
        if not output_dir:
            output_dir = os.path.join(base_output_dir, f"{model_id.replace('/', '__')}-awq-{qbit}bit")
        success = convert_model_awq(model_id, output_dir, push_to_hub, qbit)
        if success:
            return f"모델이 성공적으로 변환되었습니다: {output_dir}"
        else:
            return "모델 변환에 실패했습니다."
    elif quant_type == "mlx":
        if not output_dir:
            output_dir = os.path.join(base_output_dir, f"{model_id.replace('/', '__')}-mlx-{qbit}bit")
        success = convert_model_mlx(model_id, output_dir, push_to_hub, qbit)
        if success:
            return f"모델이 성공적으로 변환되었습니다: {output_dir}"
        else:
            return "모델 변환에 실패했습니다."
    elif quant_type == "mlx_vlm":
        if not output_dir:
            output_dir = os.path.join(base_output_dir, f"{model_id.replace('/', '__')}-mlx_vlm-{qbit}bit")
        success = convert_model_mlx_vlm(model_id, output_dir, push_to_hub, qbit)
        if success:
            return f"모델이 성공적으로 변환되었습니다: {output_dir}"
        else:
            return "모델 변환에 실패했습니다."
    elif quant_type == "metal":
        if not output_dir:
            output_dir = os.path.join(base_output_dir, f"{model_id.replace('/', '__')}-metal-{qbit}bit")
        success = convert_model_metal(model_id, output_dir, push_to_hub, qbit)
        if success:
            return f"모델이 성공적으로 변환되었습니다: {output_dir}"
        else:
            return "모델 변환에 실패했습니다."
    else:
        return "지원되지 않는 변환 유형입니다."
