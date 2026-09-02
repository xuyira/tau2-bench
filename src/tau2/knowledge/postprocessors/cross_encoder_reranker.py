from typing import Any, Dict, List, Tuple

from tau2.knowledge.postprocessors.base import BasePostprocessor
from tau2.knowledge.registry import register_postprocessor

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@register_postprocessor("cross_encoder_reranker")
class CrossEncoderReranker(BasePostprocessor):
    """Rerank retrieval candidates with a local CPU cross-encoder."""

    def __init__(
        self,
        model: str = DEFAULT_CROSS_ENCODER_MODEL,
        top_k: int = 10,
        query_key: str = "query",
        batch_size: int = 8,
        max_length: int = 512,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if top_k < 1 or batch_size < 1 or max_length < 1:
            raise ValueError("top_k, batch_size, and max_length must be positive")
        self.model_name = model
        self.top_k = top_k
        self.query_key = query_key
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device="cpu",
                max_length=self.max_length,
            )
        return self._model

    def process(
        self,
        results: List[Tuple[str, float]],
        input_data: Dict[str, Any],
        state: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        if not results:
            return results

        query = input_data.get(self.query_key, "")
        if not query:
            return results[: self.top_k]

        content_map = state.get("doc_content_map", {})
        title_map = state.get("doc_title_map", {})
        chunk_rerank_text_map = state.get("chunk_rerank_text_map", {})
        doc_ids = []
        pairs = []
        for doc_id, _ in results:
            content = content_map.get(doc_id, "")
            if content:
                document_text = chunk_rerank_text_map.get(doc_id)
                if document_text is None:
                    title = title_map.get(doc_id, "")
                    document_text = f"{title}\n{content}" if title else content
                doc_ids.append(doc_id)
                pairs.append((query, document_text))

        if not pairs:
            return results[: self.top_k]

        scores = self._get_model().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        reranked = sorted(
            zip(doc_ids, (float(score) for score in scores)),
            key=lambda item: (-item[1], item[0]),
        )
        return reranked[: self.top_k]
