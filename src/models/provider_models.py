import lmstudio as lms
import ollama

def get_lmstudio_models():
    llm = []
    downloaded_llm = lms.list_downloaded_models("llm")
    
    for m in downloaded_llm:
        llm.append(m.model_key)

    return llm

def get_lmstudio_embedding_models():
    embedding = []
    downloaded_embedding = lms.list_downloaded_models("embedding")
    for m in downloaded_embedding:
        embedding.append(m.model_key)

    return embedding

def get_ollama_models():
    llm = []
    
    for m in ollama.list().models:
        llm.append(m.model)

    return llm

try:
    lmstudio_models = get_lmstudio_models()
except:
    lmstudio_models = ["LM Studio 서버가 연결되지 않았습니다."]

try:
    ollama_models = get_ollama_models()
except:
    ollama_models = ["Ollama를 설치하고 서버를 실행해주세요."]