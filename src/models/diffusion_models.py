from src.common.utils import get_all_diffusion_models

diffusion_models_data=get_all_diffusion_models()
checkpoints_local=diffusion_models_data["checkpoints"]
diffusers_local=diffusion_models_data["diffusers"]
