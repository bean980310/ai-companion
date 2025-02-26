from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoConfig, AutoModelForCausalLM
from peft import PeftModel

from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler

class TransformersCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, device='cpu'):
        super().__init__(model_id, lora_model_id)
        self.device = device
        self.load_model()
        
    def load_model(self):
        self.config = AutoConfig.from_pretrained(self.local_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, config=self.config, trust_remote_code=True, device_map='auto')
        
    def load_template(self):
        return super().load_template()