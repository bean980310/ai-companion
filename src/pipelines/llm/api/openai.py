from .... import logger
import traceback

import openai

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

class OpenAIClientWrapper:
    def __init__(self, selected_model, api_key="None", **kwargs):
        self.model = selected_model
        self.api_key = api_key

        self.max_tokens=2048
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 1.0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

    def generate_answer(self, history, **kwargs):
        openai.api_key = self.api_key

        messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
        logger.info(f"[*] OpenAI API 요청: {messages}")

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_logprobs=self.top_k,
                frequency_penalty=self.repetition_penalty,
                top_p=self.top_p,
            )
            answer = response.choices[0].message["content"]
            logger.info(f"[*] OpenAI 응답: {answer}")
            return answer
        except Exception as e:
            logger.error(f"OpenAI API 오류: {str(e)}\n\n{traceback.format_exc()}")
            return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"