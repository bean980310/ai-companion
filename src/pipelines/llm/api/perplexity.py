from .... import logger
import traceback
import requests

class PerplexityClientWrapper:
    def __init__(self, selected_model, api_key="None", **kwargs):
        self.model = selected_model
        self.api_key = api_key

        self.max_tokens=2048
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 1.0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

    def generate_answer(self, history, **kwargs):
        messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
        logger.info(f"[*] Perplexity API 요청: {messages}")
            
        try:
            url = "https://api.perplexity.ai/chat/completions"
            payload = { 
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "frequency_penalty": self.repetition_penalty
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.request("POST", url, json=payload, headers=headers)
            answer = response.text
            logger.info(f"[*] Perplexity 응답: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"Perplexity API 오류: {str(e)}\n\n{traceback.format_exc()}")
            return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"