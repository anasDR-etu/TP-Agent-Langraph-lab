from operator import add
from typing import Literal
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import AnyMessage
from langchain.messages import (
    SystemMessage,
    ToolMessage,
    HumanMessage,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from langgraph.types import (
    interrupt,
    Command,
)

from tools_setup import (
    model_with_tools,
    tools_by_name,
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    llm_calls: int


def llm_call(state: AgentState):

    response = model_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant that solves arithmetic problems using tools."
            )
        ]
        + state["messages"]
    )

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: AgentState):

    last = state["messages"][-1]

    results = []

    for call in last.tool_calls:

        tool = tools_by_name[call["name"]]

        observation = tool.invoke(call["args"])

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=call["id"]
            )
        )

    return {
        "messages": results
    }


def should_continue(
    state: AgentState
) -> Literal["approve", "__end__"]:

    last = state["messages"][-1]

    if getattr(last, "tool_calls", None):
        if last.tool_calls:
            return "approve"

    return END


def approve_node(
    state: AgentState
):

    decision = interrupt(
        {
            "question": "Approve tool execution?",
            "tool_calls": state["messages"][-1].tool_calls
        }
    )

    return Command(
        goto="tool_node" if decision else END
    )


builder = StateGraph(AgentState)

builder.add_node("llm_call", llm_call)
builder.add_node("approve", approve_node)
builder.add_node("tool_node", tool_node)

builder.add_edge(
    START,
    "llm_call"
)

builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["approve", END]
)

builder.add_edge(
    "tool_node",
    "llm_call"
)

checkpointer = InMemorySaver()

agent = builder.compile(
    checkpointer=checkpointer
)

config = {
    "configurable": {
        "thread_id": "thread-1"
    }
}

result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Add 3 and 4."
            )
        ],
        "llm_calls": 0,
    },
    config=config,
)

print("\nINTERRUPT\n")

print(
    result["__interrupt__"][0].value
)

resume = agent.invoke(
    Command(resume=True),
    config=config,
)

print("\nAPPROVED RESULT\n")

print(
    resume["messages"][-1]
)

config_reject = {
    "configurable": {
        "thread_id": "thread-1-reject"
    }
}

result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Multiply 30 and 41."
            )
        ],
        "llm_calls": 0,
    },
    config=config_reject,
)

print("\nREJECT INTERRUPT\n")

print(
    result["__interrupt__"][0].value
)

resume = agent.invoke(
    Command(resume=False),
    config=config_reject,
)

print("\nREJECT RESULT\n")

print(
    resume["messages"][-1]
)

history = list(
    agent.get_state_history(
        config_reject
    )
)

picked = history[1]

new_config = agent.update_state(
    picked.config,
    values={
        "messages": [
            HumanMessage(
                content="Multiply 30 and 41."
            )
        ],
        "llm_calls": 0,
    }
)

forked = agent.invoke(
    None,
    new_config
)

print("\nFORKED RESULT\n")

print(forked)