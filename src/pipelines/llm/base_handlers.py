from abc import ABC, abstractmethod
import os

class BaseModelHandler(ABC):
    def __init__(self, model_id, lora_model_id=None):
        self.model_id = model_id
        self.config = None
        self.local_model_path = os.path.join("./models/llm", model_id)
        self.local_lora_model_path = os.path.join("./models/llm/loras", lora_model_id) if lora_model_id else None
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate_answer(self, history, **kwargs):
        pass
    
    @abstractmethod
    def get_settings(self):
        pass
    
    @abstractmethod
    def load_template(self, messages):
        pass
        
class BaseCausalModelHandler(BaseModelHandler):
    def __init__(self, model_id, lora_model_id=None):
        super().__init__(model_id, lora_model_id)
        self.tokenizer = None
        self.model = None
        
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate_answer(self, history, **kwargs):
        pass
    
    @abstractmethod
    def get_settings(self, *, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0):
        pass
    
    @abstractmethod
    def load_template(self, messages):
        pass
    
class BaseVisionModelHandler(BaseModelHandler):
    def __init__(self, model_id, lora_model_id=None):
        super().__init__(model_id, lora_model_id)
        self.processor = None
        self.model = None
        
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate_answer(self, history, image_input=None, **kwargs):
        pass
    
    @abstractmethod
    def get_settings(self, *, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0):
        pass
    
    @abstractmethod
    def load_template(self, messages, image_input):
        pass
    