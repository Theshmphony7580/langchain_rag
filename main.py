from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings.langchain_embed import embeddings
from load_doc import load_langchain_docs

PERSIST_DIR = "./chroma_langchain_db"
COLLECTION = "example_collection"


def main() -> None:
    docs = load_langchain_docs()
    if not docs:
        raise RuntimeError("No documents loaded — check DOC_PATHS or network.")
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    store.add_documents(documents=splits)
    print(f"Indexed {len(splits)} chunks from {len(docs)} pages into {PERSIST_DIR}.")


if __name__ == "__main__":
    main()
