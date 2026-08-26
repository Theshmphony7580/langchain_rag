from functools import lru_cache
from langchain_chroma import Chroma
from langchain_core.documents import Document

from retrieval.bm25 import BM25Index, DEFAULT_BM25_PATH
from retrieval.reranker import FlashRankReranker
from split_doc.langchain_split import get_vector_store


class HybridRetriever:
    """Two-stage hybrid retriever: Dense (Chroma) + Sparse (BM25) with FlashRank reranking."""

    def __init__(
        self,
        vector_store: Chroma,
        bm25_index: BM25Index,
        reranker: FlashRankReranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.reranker = reranker or FlashRankReranker()

    def search(
        self,
        query: str,
        k_dense: int = 10,
        k_sparse: int = 10,
        k_final: int = 5,
    ) -> list[Document]:
        """Perform two-stage hybrid search: retrieve candidate pool and neural rerank."""
        # 1. Retrieve dense candidates from Chroma
        dense_docs = self.vector_store.similarity_search(query, k=k_dense)

        # 2. Retrieve sparse candidates from BM25
        sparse_docs = self.bm25_index.search(query, k=k_sparse)

        # 3. Deduplicate candidates by page content
        seen_contents = set()
        candidate_pool: list[Document] = []

        for doc in dense_docs + sparse_docs:
            content_key = doc.page_content.strip()
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                candidate_pool.append(doc)

        if not candidate_pool:
            return []

        # 4. Neural rerank with FlashRank
        return self.reranker.rerank(query, candidate_pool, top_n=k_final)


@lru_cache(maxsize=1)
def get_hybrid_retriever(
    bm25_path: str = DEFAULT_BM25_PATH,
) -> HybridRetriever:
    """Load cached singleton HybridRetriever with persistent Chroma and BM25 store."""
    vector_store = get_vector_store()
    bm25_index = BM25Index.load(bm25_path)
    reranker = FlashRankReranker()
    return HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        reranker=reranker,
    )
