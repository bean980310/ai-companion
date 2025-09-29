from langchain_huggingface import ChatHuggingFace
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)

class ChatHuggingFaceEnhanced(ChatHuggingFace):
    r"""Hugging Face LLM's as ChatModels.

    Works with `HuggingFaceTextGenInference`, `HuggingFaceEndpoint`,
    `HuggingFaceHub`, and `HuggingFacePipeline` LLMs.

    Upon instantiating this class, the model_id is resolved from the url
    provided to the LLM, and the appropriate tokenizer is loaded from
    the HuggingFace Hub.

    Setup:
        Install ``langchain-huggingface`` and ensure your Hugging Face token
        is saved.

        .. code-block:: bash

            pip install langchain-huggingface

        .. code-block:: python

            from huggingface_hub import login
            login() # You will be prompted for your HF key, which will then be saved locally

    Key init args — completion params:
        llm: `HuggingFaceTextGenInference`, `HuggingFaceEndpoint`, `HuggingFaceHub`, or
            'HuggingFacePipeline' LLM to be used.

    Key init args — client params:
        custom_get_token_ids: Optional[Callable[[str], list[int]]]
            Optional encoder to use for counting tokens.
        metadata: Optional[dict[str, Any]]
            Metadata to add to the run trace.
        tags: Optional[list[str]]
            Tags to add to the run trace.
        tokenizer: Any
        verbose: bool
            Whether to print out response text.

    See full list of supported init args and their descriptions in the params
    section.

    Instantiate:
        .. code-block:: python

            from langchain_huggingface import HuggingFaceEndpoint,
            ChatHuggingFace

            llm = HuggingFaceEndpoint(
                repo_id="microsoft/Phi-3-mini-4k-instruct",
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
            )

            chat = ChatHuggingFace(llm=llm, verbose=True)

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user
                sentence to French."),
                ("human", "I love programming."),
            ]

            chat(...).invoke(messages)

        .. code-block:: python

            AIMessage(content='Je ai une passion pour le programme.\n\nIn
            French, we use "ai" for masculine subjects and "a" for feminine
            subjects. Since "programming" is gender-neutral in English, we
            will go with the masculine "programme".\n\nConfirmation: "J\'aime
            le programme." is more commonly used. The sentence above is
            technically accurate, but less commonly used in spoken French as
            "ai" is used less frequently in everyday speech.',
            response_metadata={'token_usage': ChatCompletionOutputUsage
            (completion_tokens=100, prompt_tokens=55, total_tokens=155),
            'model': '', 'finish_reason': 'length'},
            id='run-874c24b7-0272-4c99-b259-5d6d7facbc56-0')

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk)

        .. code-block:: python

            content='Je ai une passion pour le programme.\n\nIn French, we use
            "ai" for masculine subjects and "a" for feminine subjects.
            Since "programming" is gender-neutral in English,
            we will go with the masculine "programme".\n\nConfirmation:
            "J\'aime le programme." is more commonly used. The sentence
            above is technically accurate, but less commonly used in spoken
            French as "ai" is used less frequently in everyday speech.'
            response_metadata={'token_usage': ChatCompletionOutputUsage
            (completion_tokens=100, prompt_tokens=55, total_tokens=155),
            'model': '', 'finish_reason': 'length'}
            id='run-7d7b1967-9612-4f9a-911a-b2b5ca85046a-0'

    Async:
        .. code-block:: python

            await chat.ainvoke(messages)

        .. code-block:: python

            AIMessage(content='Je déaime le programming.\n\nLittérale : Je
            (j\'aime) déaime (le) programming.\n\nNote: "Programming" in
            French is "programmation". But here, I used "programming" instead
            of "programmation" because the user said "I love programming"
            instead of "I love programming (in French)", which would be
            "J\'aime la programmation". By translating the sentence
            literally, I preserved the original meaning of the user\'s
            sentence.', id='run-fd850318-e299-4735-b4c6-3496dc930b1d-0')

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state,
                e.g. San Francisco, CA")

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state,
                e.g. San Francisco, CA")

            chat_with_tools = chat.bind_tools([GetWeather, GetPopulation])
            ai_msg = chat_with_tools.invoke("Which city is hotter today and
            which is bigger: LA or NY?")
            ai_msg.tool_calls

        .. code-block:: python

            [{'name': 'GetPopulation',
              'args': {'location': 'Los Angeles, CA'},
              'id': '0'}]

    Response metadata
        .. code-block:: python

            ai_msg = chat.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python
            {'token_usage': ChatCompletionOutputUsage(completion_tokens=100,
            prompt_tokens=8, total_tokens=108),
             'model': '',
             'finish_reason': 'length'}

    """  # noqa: E501
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._resolve_model_id()

    def _to_chat_prompt_thinking(
        self,
        messages: list[BaseMessage],
        enable_thinking: bool = False,
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            msg = "At least one HumanMessage must be provided!"
            raise ValueError(msg)

        if not isinstance(messages[-1], HumanMessage):
            msg = "Last message must be a HumanMessage!"
            raise ValueError(msg)

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        )