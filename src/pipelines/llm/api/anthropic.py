from .... import logger
import traceback
import anthropic
class AnthropicClientWrapper:
    def __init__(self, selected_model, api_key="None", **kwargs):
        self.model = selected_model
        self.api_key = api_key

        self.max_tokens=2048
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 1.0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

    def generate_answer(self, history, **kwargs):
        if not self.api_key:
            logger.error("Anthropic API Key가 missing.")
            return "Anthropic API Key가 필요합니다."
                
        client = anthropic.Client(api_key=self.api_key)
        # Anthropic 메시지 형식으로 변환
        messages = []
        for msg in history:
            if msg["role"] == "system":
                system = msg["content"]
                continue  # Claude API는 시스템 메시지를 별도로 처리하지 않음
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
                
        logger.info(f"[*] Anthropic API 요청: {messages}")
                
        try:
            response = client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                # frequency_penalty=repetition_penalty,
                max_tokens=self.max_tokens,
            )
            answer = response.content[0].text
            logger.info(f"[*] Anthropic 응답: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Anthropic API 오류: {str(e)}\n\n{traceback.format_exc()}")
            return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"