from typing import Any, Dict, List, Tuple

import numpy as np

from tau2.knowledge.registry import register_retriever
from tau2.knowledge.retrievers.base import BaseRetriever


@register_retriever("bm25_dense_rrf")
class BM25DenseRRFRetriever(BaseRetriever):
    """Fuse BM25 and dense rankings with reciprocal rank fusion."""

    def __init__(
        self,
        query_key: str = "query",
        embedding_key: str = "query_embedding",
        bm25_state_key: str = "bm25",
        bm25_doc_ids_state_key: str = "bm25_doc_ids",
        embedding_index_key: str = "doc_embeddings",
        candidate_top_k: int = 50,
        top_k: int = 10,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        **kwargs: Any,
    ):
        if candidate_top_k < 1 or top_k < 1:
            raise ValueError("candidate_top_k and top_k must be positive")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if bm25_weight < 0 or dense_weight < 0:
            raise ValueError("RRF weights must be non-negative")
        if bm25_weight == 0 and dense_weight == 0:
            raise ValueError("At least one RRF weight must be positive")

        super().__init__(
            query_key=query_key,
            embedding_key=embedding_key,
            bm25_state_key=bm25_state_key,
            bm25_doc_ids_state_key=bm25_doc_ids_state_key,
            embedding_index_key=embedding_index_key,
            candidate_top_k=candidate_top_k,
            top_k=top_k,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            **kwargs,
        )
        self.query_key = query_key
        self.embedding_key = embedding_key
        self.bm25_state_key = bm25_state_key
        self.bm25_doc_ids_state_key = bm25_doc_ids_state_key
        self.embedding_index_key = embedding_index_key
        self.candidate_top_k = candidate_top_k
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def retrieve(
        self, input_data: Dict[str, Any], state: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        query = input_data.get(self.query_key)
        query_embedding = input_data.get(self.embedding_key)
        if not query or not query.strip() or query_embedding is None:
            return []

        bm25 = state[self.bm25_state_key]
        bm25_doc_ids = state[self.bm25_doc_ids_state_key]
        doc_embeddings = state[self.embedding_index_key]
        dense_doc_ids = state[f"{self.embedding_index_key}_doc_ids"]

        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_count = min(self.candidate_top_k, len(bm25_doc_ids))
        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda index: (-bm25_scores[index], bm25_doc_ids[index]),
        )[:bm25_count]

        dense_scores = self._cosine_similarity(query_embedding, doc_embeddings)
        dense_count = min(self.candidate_top_k, len(dense_doc_ids))
        dense_indices = sorted(
            range(len(dense_scores)),
            key=lambda index: (-dense_scores[index], dense_doc_ids[index]),
        )[:dense_count]

        fused_scores: Dict[str, float] = {}
        for rank, index in enumerate(bm25_indices, start=1):
            doc_id = bm25_doc_ids[index]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (
                self.bm25_weight / (self.rrf_k + rank)
            )
        for rank, index in enumerate(dense_indices, start=1):
            doc_id = dense_doc_ids[index]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (
                self.dense_weight / (self.rrf_k + rank)
            )

        ranked = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[: min(self.top_k, len(ranked))]

    @staticmethod
    def _cosine_similarity(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
        query = np.asarray(query)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(len(docs))

        doc_norms = np.linalg.norm(docs, axis=1)
        safe_doc_norms = np.where(doc_norms == 0, 1, doc_norms)
        return docs @ query / (safe_doc_norms * query_norm)
