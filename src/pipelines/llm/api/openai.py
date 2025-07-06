from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

def generate_answer(messages, model, temperature, top_p, top_k, repetition_penalty, api_key, client):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        top_p=top_p,
        top_logprobs=top_k,
        frequency_penalty=repetition_penalty,
        presence_penalty=repetition_penalty,
        api_key=api_key,
        client=client,
        max_tokens=2048,
    )