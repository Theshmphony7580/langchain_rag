import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Data_paths.langchain_doc import DOCS_BASE,DOC_PATHS

def load_langchain_docs(doc_paths: list[str] | None = None) -> list[Document]:
    """Fetch LangChain documentation pages as Documents."""
    paths = doc_paths or DOC_PATHS
    docs: list[Document] = []
    for path in paths:
        url = f"{DOCS_BASE}/{path}.md"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            continue
        source = f"{DOCS_BASE}/{path}"
        docs.append(
            Document(page_content=response.text, metadata={"source": source})
        )
    return docs


docs = load_langchain_docs()
print(f"Loaded {len(docs)} documentation pages.")

total_chars = sum(len(doc.page_content) for doc in docs)
print(f"Total characters: {total_chars}")
print(docs[0].page_content[:500])