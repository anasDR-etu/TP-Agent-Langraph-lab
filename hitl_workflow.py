import time
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command


@task
def write_essay(topic: str) -> str:

    time.sleep(1)

    return f"Essay draft about {topic}"


@entrypoint(
    checkpointer=InMemorySaver()
)
def workflow(topic: str):

    draft = write_essay(topic).result()

    approved = interrupt(
        {
            "draft": draft,
            "action": "approve or reject"
        }
    )

    return {
        "draft": draft,
        "approved": approved,
    }


thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id
    }
}

print("\nFIRST EXECUTION\n")

for item in workflow.stream(
    "cats",
    config
):
    print(item)

print("\nSECOND EXECUTION\n")

for item in workflow.stream(
    Command(resume=True),
    config
):
    print(item)