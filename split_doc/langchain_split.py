from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_doc import docs
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print(f"Split documentation into {len(all_splits)} chunks.")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

vector_store.add_documents(documents=all_splits)
print(f"Indexed {len(all_splits)} chunks.")

