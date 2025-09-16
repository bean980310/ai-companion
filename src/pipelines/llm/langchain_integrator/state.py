from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

class State(TypedDict):
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    input: str