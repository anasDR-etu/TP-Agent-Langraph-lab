from langchain.tools import tool
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.2:3b"
)

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divide two integers."""
    return a / b


tools = [add, multiply, divide]

tools_by_name = {
    tool.name: tool
    for tool in tools
}

model_with_tools = model.bind_tools(tools)
