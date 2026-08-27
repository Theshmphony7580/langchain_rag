import time
from observability.logfire_config import logfire  # noqa: F401

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings.langchain_embed import embeddings
from load_doc import load_langchain_docs
from retrieval.bm25 import BM25Index

PERSIST_DIR = "./chroma_langchain_db"
COLLECTION = "example_collection"
BATCH_SIZE = 50
MAX_RETRIES = 5


def add_documents_with_retry(store: Chroma, batch: list, batch_num: int, total_batches: int) -> None:
    """Add a batch of documents with automatic exponential backoff on 429 rate limits."""
    delay = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            store.add_documents(documents=batch)
            print(f"  [OK] Embedded batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            return
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"  [WAIT] Rate limit hit on batch {batch_num}. Backing off for {delay:.1f}s (Attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(delay)
                delay *= 2.0
            else:
                raise e
    raise RuntimeError(f"Failed to index batch {batch_num} after {MAX_RETRIES} attempts.")


def main() -> None:
    print("Step 1/3: Loading documentation pages...")
    docs = load_langchain_docs()
    if not docs:
        raise RuntimeError("No documents loaded — check DOC_PATHS or network.")

    print(f"Step 2/3: Splitting {len(docs)} documents into chunks...")
    splits = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200
    ).split_documents(docs)
    print(f"Generated {len(splits)} text chunks.")

    print("Step 3/3: Embedding and indexing into Chroma + BM25...")
    # 1. Build & persist dense Chroma index with rate-controlled batching
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    
    total_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE
    for idx, i in enumerate(range(0, len(splits), BATCH_SIZE), start=1):
        batch = splits[i : i + BATCH_SIZE]
        add_documents_with_retry(store, batch, idx, total_batches)
        time.sleep(1.2)  # Safe delay between batches to stay within free-tier quota

    print(f"  [DONE] Indexed {len(splits)} chunks into Chroma ({PERSIST_DIR}).")

    # 2. Build & persist sparse BM25 index
    bm25_index = BM25Index.build(splits)
    bm25_index.save("./bm25_index/bm25_store.pkl")
    print("  [DONE] Indexed into BM25 (./bm25_index/bm25_store.pkl).")
    print("\nIngestion complete! You can now run agent.py.")


if __name__ == "__main__":
    main()
