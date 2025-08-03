from abc import ABC, abstractmethod
from typing import Any
import os

class BaseModel(ABC):
    def __init__(self, use_langchain: bool = True, **kwargs):
        self.use_langchain = use_langchain

        self.max_tokens: int = kwargs.get("max_tokens", 2048)
        self.temperature: float = kwargs.get("temperature", 1.0)
        self.top_k: int = kwargs.get("top_k", 50)
        self.top_p: float = kwargs.get("top_p", 1.0)
        self.repetition_penalty: float = kwargs.get("repetition_penalty", 1.0)

        self.langchain_integrator = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_answer(self, history, **kwargs):
        pass

class BaseModelHandler(BaseModel):
    def __init__(self, model_id: str, lora_model_id: str = None, use_langchain: bool = True, image_input = None, **kwargs):
        super().__init__(use_langchain, **kwargs)
        self.model_id = model_id
        self.lora_model_id = lora_model_id
        self.config = None
        self.local_model_path = os.path.join("./models/llm", model_id)
        self.local_lora_model_path = os.path.join("./models/llm/loras", lora_model_id) if lora_model_id else None
        self.image_input = image_input

        self.processor = None
        self.tokenizer = None
        self.model = None

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
    def __init__(self, model_id: str, lora_model_id: str = None, use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)
        
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
class BaseVisionModelHandler(BaseModelHandler):
    def __init__(self, model_id: str, lora_model_id: str = None, use_langchain: bool = True, image_input = None, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, image_input, **kwargs)
        
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
class BaseAPIClientWrapper(BaseModel):
    def __init__(self, selected_model: str, api_key: str = "None", use_langchain: bool = True, **kwargs):
        super().__init__(use_langchain, **kwargs)
        self.model = selected_model
        self.api_key = api_key

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_answer(self, history, **kwargs):
        pass
