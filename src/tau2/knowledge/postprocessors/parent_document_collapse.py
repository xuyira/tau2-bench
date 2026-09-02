from typing import Any, Dict, List, Tuple

from tau2.knowledge.postprocessors.base import BasePostprocessor
from tau2.knowledge.registry import register_postprocessor


@register_postprocessor("parent_document_collapse")
class ParentDocumentCollapse(BasePostprocessor):
    """Collapse ranked child chunks to unique parent documents."""

    def __init__(self, top_k: int = 10, **kwargs: Any):
        super().__init__(**kwargs)
        self.top_k = top_k

    def process(
        self,
        results: List[Tuple[str, float]],
        input_data: Dict[str, Any],
        state: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        parent_map = state.get("chunk_parent_map", {})
        collapsed = []
        seen = set()
        for chunk_id, score in results:
            parent_id = parent_map.get(chunk_id, chunk_id)
            if parent_id not in seen:
                collapsed.append((parent_id, score))
                seen.add(parent_id)
            if len(collapsed) >= self.top_k:
                break
        return collapsed
