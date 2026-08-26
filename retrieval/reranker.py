from functools import lru_cache
from typing import Sequence
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

DEFAULT_RERANKER_MODEL = "ms-marco-TinyBERT-L-2-v2"


@lru_cache(maxsize=1)
def get_ranker(model_name: str = DEFAULT_RERANKER_MODEL) -> Ranker:
    """Cached singleton for FlashRank Ranker ONNX model."""
    return Ranker(model_name=model_name)


class FlashRankReranker:
    """Fast, CPU-optimized ONNX cross-encoder reranker for document candidates."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.ranker = get_ranker(model_name)

    def rerank(
        self, query: str, documents: Sequence[Document], top_n: int = 5
    ) -> list[Document]:
        """Rerank a sequence of Documents against query and return top_n."""
        if not documents:
            return []

        doc_list = list(documents)
        passages = [
            {
                "id": idx,
                "text": doc.page_content,
                "meta": doc.metadata,
            }
            for idx, doc in enumerate(doc_list)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        reranked_docs: list[Document] = []
        for item in results[:top_n]:
            idx = item["id"]
            original_doc = doc_list[idx]
            # Copy and attach rerank score
            meta = dict(original_doc.metadata)
            meta["rerank_score"] = float(item.get("score", 0.0))
            reranked_docs.append(
                Document(page_content=original_doc.page_content, metadata=meta)
            )

        return reranked_docs
