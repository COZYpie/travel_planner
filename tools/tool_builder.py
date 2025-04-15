from langchain.agents import Tool
from typing import Callable

def build_tool(name: str, func: Callable, description: str) -> Tool:
    return Tool(name=name, func=func, description=description)