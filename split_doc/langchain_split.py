from langchain_chroma import Chroma

from embeddings.langchain_embed import embeddings

# Read-only handle over the persisted DB that main.py writes.
# Keep this in sync with main.py's PERSIST_DIR / COLLECTION.
PERSIST_DIR = "./chroma_langchain_db"
COLLECTION = "example_collection"


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )