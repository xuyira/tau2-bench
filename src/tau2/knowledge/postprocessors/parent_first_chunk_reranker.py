from typing import Any, Dict, List, Tuple

from tau2.knowledge.document_preprocessors.markdown_semantic_chunker import (
    MarkdownSemanticChunker,
)
from tau2.knowledge.document_preprocessors.search_text import normalize_document_id
from tau2.knowledge.postprocessors.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from tau2.knowledge.registry import register_postprocessor


@register_postprocessor("parent_first_chunk_reranker")
class ParentFirstChunkReranker(CrossEncoderReranker):
    """Rerank parent candidates by their highest-scoring semantic chunk."""

    def __init__(self, chunk_max_chars: int = 2000, **kwargs: Any):
        super().__init__(**kwargs)
        self.chunker = MarkdownSemanticChunker(max_chars=chunk_max_chars)
        self.chunk_max_chars = chunk_max_chars

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
        pair_parents = []
        pairs = []
        for parent_id, _ in results:
            content = content_map.get(parent_id, "")
            if not content:
                continue
            title = title_map.get(parent_id, parent_id)
            chunks = (
                [("Document", content)]
                if len(content) <= self.chunk_max_chars
                else self.chunker.split_content(content)
            )
            normalized_id = normalize_document_id(parent_id)
            for heading_path, chunk_content in chunks:
                document_text = (
                    f"ID: {normalized_id}\n"
                    f"Title: {title}\n"
                    f"Section: {heading_path}\n"
                    f"Content:\n{chunk_content}"
                )
                pair_parents.append(parent_id)
                pairs.append((query, document_text))

        if not pairs:
            return results[: self.top_k]

        scores = self._get_model().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        parent_scores: Dict[str, float] = {}
        for parent_id, score in zip(pair_parents, scores):
            float_score = float(score)
            if parent_id not in parent_scores or float_score > parent_scores[parent_id]:
                parent_scores[parent_id] = float_score

        reranked = sorted(
            parent_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return reranked[: self.top_k]
