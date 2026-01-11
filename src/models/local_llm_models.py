from src.common.utils import get_all_local_models

local_models_data = get_all_local_models()
transformers_local = local_models_data["transformers"]
vllm_local = local_models_data["vllm-local"]
gguf_local = local_models_data["gguf"]
mlx_local = local_models_data["mlx"]