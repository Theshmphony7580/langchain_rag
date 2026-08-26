import os
import pickle
import re
from pathlib import Path
from typing import Sequence
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

DEFAULT_BM25_PATH = "./bm25_index/bm25_store.pkl"


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alpha-numeric tokens including underscores for code."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Sparse keyword search index using BM25Okapi with disk persistence."""

    def __init__(
        self,
        corpus: list[Document] | None = None,
        bm25: BM25Okapi | None = None,
    ) -> None:
        self.corpus: list[Document] = corpus or []
        self.bm25: BM25Okapi | None = bm25

    @classmethod
    def build(cls, documents: Sequence[Document]) -> "BM25Index":
        """Build BM25 index from a list of Document objects."""
        doc_list = list(documents)
        tokenized_corpus = [_tokenize(doc.page_content) for doc in doc_list]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(corpus=doc_list, bm25=bm25)

    def save(self, path: str = DEFAULT_BM25_PATH) -> None:
        """Persist BM25 index and corpus to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump({"corpus": self.corpus, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: str = DEFAULT_BM25_PATH) -> "BM25Index":
        """Load persisted BM25 index and corpus from disk."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run main.py first to build the index."
            )
        with open(target, "rb") as f:
            data = pickle.load(f)
        return cls(corpus=data["corpus"], bm25=data["bm25"])

    def search(self, query: str, k: int = 10) -> list[Document]:
        """Query the BM25 index and return top-k matching Documents."""
        if not self.bm25 or not self.corpus:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [self.corpus[i] for i in top_indices if scores[i] > 0]
