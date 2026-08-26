from observability.logfire_config import logfire  # noqa: F401

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings.langchain_embed import embeddings
from load_doc import load_langchain_docs
from retrieval.bm25 import BM25Index

PERSIST_DIR = "./chroma_langchain_db"
COLLECTION = "example_collection"


def main() -> None:
    print("Step 1/3: Loading documentation pages...")
    docs = load_langchain_docs()
    if not docs:
        raise RuntimeError("No documents loaded — check DOC_PATHS or network.")

    print(f"Step 2/3: Splitting {len(docs)} documents into chunks...")
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    print(f"Generated {len(splits)} text chunks.")

    print("Step 3/3: Embedding and indexing into Chroma + BM25...")
    # 1. Build & persist dense Chroma index
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    store.add_documents(documents=splits)
    print(f"  [OK] Indexed {len(splits)} chunks into Chroma ({PERSIST_DIR}).")

    # 2. Build & persist sparse BM25 index
    bm25_index = BM25Index.build(splits)
    bm25_index.save("./bm25_index/bm25_store.pkl")
    print("  [OK] Indexed into BM25 (./bm25_index/bm25_store.pkl).")
    print("\nIngestion complete! You can now run agent.py.")


if __name__ == "__main__":
    main()
