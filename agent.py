import uuid

from deepagents.backends import StateBackend
from langchain.tools import tool
from split_doc.langchain_split import vector_store

backend = StateBackend()


@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Search LangChain documentation and save matching chunks to the agent filesystem.

    Args:
        query: Natural language search query.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/.
    """
    retrieved_docs = vector_store.similarity_search(query, k=4)
    batch_id = uuid.uuid4().hex[:8]
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        content = (
            f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path, content.encode("utf-8")))
        saved_paths.append(path)

    backend.upload_files(uploads)
    return (
        f"Saved {len(saved_paths)} documentation chunks:\n"
        + "\n".join(saved_paths)
    )


from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

max_concurrent_analysts = 3

RAG_WORKFLOW_INSTRUCTIONS = (Path("prompts") / "RAG_WORKFLOW.md").read_text(encoding="utf-8")
SUBAGENT_DELEGATION_INSTRUCTIONS = (Path("prompts") / "SUBAGENT.md").read_text(encoding="utf-8")
CHUNK_ANALYST_INSTRUCTIONS = (Path("prompts") / "CHUNK_ANALYST.md").read_text(encoding="utf-8")

INSTRUCTIONS = (
    RAG_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts=max_concurrent_analysts,
    )
)

chunk_analyst_subagent = {
    "name": "chunk-analyst",
    "description": (
        "Analyze one retrieved documentation chunk file. "
        "Pass the user question and a single file path under /retrieved/."
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
}

model = init_chat_model(model="openrouter:openrouter/free")

agent = create_deep_agent(
    model=model,
    tools=[search_documentation],
    backend=backend,
    system_prompt=INSTRUCTIONS,
    subagents=[chunk_analyst_subagent],
)

from langchain.messages import HumanMessage

EXAMPLE_QUERY = "How do I stream intermediate tool results from a subagent?"

if __name__ == "__main__":
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=EXAMPLE_QUERY)]}
        )
    except Exception as exc:
        print(f"Agent invocation failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    for msg in result.get("messages", []):
        if msg.text:
            print(msg.text)