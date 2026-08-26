import requests
from langchain_core.documents import Document
from Data_paths.langchain_doc import DOCS_BASE,DOC_PATHS

def load_langchain_docs(doc_paths: list[str] | None = None) -> list[Document]:
    """Fetch LangChain documentation pages as Documents."""
    paths = doc_paths or DOC_PATHS
    docs: list[Document] = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    print(f"Fetching {len(paths)} documentation pages from {DOCS_BASE}...")
    for path in paths:
        url = f"{DOCS_BASE}/{path}.md"
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            source = f"{DOCS_BASE}/{path}"
            docs.append(
                Document(page_content=response.text, metadata={"source": source})
            )
            print(f"  [OK] {path} ({len(response.text)} chars)")
        except requests.RequestException as e:
            print(f"  [FAIL] {url} -> {e}")
            continue
    print(f"Successfully loaded {len(docs)} / {len(paths)} documents.")
    return docs


if __name__ == "__main__":
    docs = load_langchain_docs()
    print(f"Loaded {len(docs)} documentation pages.")
    if docs:
        total_chars = sum(len(doc.page_content) for doc in docs)
        print(f"Total characters: {total_chars}")
        print(docs[0].page_content[:500])