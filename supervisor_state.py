from typing import TypedDict,Literal
from langchain_core.messages import AnyMessage



class AgentState(TypedDict):
    messages: list[AnyMessage]
    route: Literal["python", "research", "end"]


