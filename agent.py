import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from observability.logfire_config import logfire  # noqa: F401

import argparse
import uuid
from functools import lru_cache

from deepagents.backends import StateBackend
from langchain.tools import tool
from retrieval.hybrid import get_hybrid_retriever

backend = StateBackend()


@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Search LangChain documentation using hybrid dense/sparse search with neural reranking.

    Args:
        query: Natural language search query.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/.
    """
    retriever = get_hybrid_retriever()
    retrieved_docs = retriever.search(query, k_dense=10, k_sparse=10, k_final=2)
    batch_id = uuid.uuid4().hex[:8]
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        rerank_info = (
            f" (Rerank Score: {doc.metadata.get('rerank_score', 0):.4f})"
            if "rerank_score" in doc.metadata
            else ""
        )
        content = (
            f"# Source: {doc.metadata.get('source', 'unknown')}{rerank_info}\n\n"
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

max_concurrent_analysts = 2

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

model = init_chat_model(model="gemini-3.6-flash", model_provider="google_genai")

agent = create_deep_agent(
    model=model,
    tools=[search_documentation],
    backend=backend,
    system_prompt=INSTRUCTIONS,
    subagents=[chunk_analyst_subagent],
)

from langchain.messages import HumanMessage

EXAMPLE_QUERY = "How do I stream intermediate tool results from a subagent?"


def run_query(query: str) -> None:
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]}
    )
    messages = result.get("messages", [])
    if messages:
        last_msg = messages[-1]
        content = getattr(last_msg, "text", None) or getattr(last_msg, "content", "")
        if content:
            print(content)


def run_interactive() -> None:
    print("Interactive agent started. Type a question and press Enter. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            return

        run_query(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LangChain RAG agent.")
    parser.add_argument("--query", "-q", help="Run a single query and exit.")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run the agent in interactive mode.",
    )
    args = parser.parse_args()

    if args.query:
        run_query(args.query)
    else:
        run_interactive()