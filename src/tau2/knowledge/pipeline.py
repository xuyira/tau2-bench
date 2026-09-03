import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tau2.knowledge.document_preprocessors.search_text import build_document_metadata
from tau2.knowledge.registry import (
    get_document_preprocessor,
    get_input_preprocessor,
    get_postprocessor,
    get_retriever,
)


@dataclass
class RetrievalTiming:
    input_preprocessing_ms: float = 0.0
    retrieval_ms: float = 0.0
    postprocessing_ms: float = 0.0
    postprocessor_details: Dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return self.input_preprocessing_ms + self.retrieval_ms + self.postprocessing_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_preprocessing_ms": round(self.input_preprocessing_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "postprocessing_ms": round(self.postprocessing_ms, 2),
            "postprocessor_details": {
                k: round(v, 2) for k, v in self.postprocessor_details.items()
            },
            "total_ms": round(self.total_ms, 2),
        }


@dataclass
class RetrievalResult:
    results: List[Tuple[str, float]]
    timing: RetrievalTiming


class RetrievalPipeline:
    def __init__(self, config: Dict[str, Any]):
        from tau2.knowledge import _ensure_registered

        _ensure_registered()

        self.config = config
        self.state: Dict[str, Any] = {}

        self.doc_preprocessors = [
            get_document_preprocessor(dp["type"], dp.get("params", {}))
            for dp in config.get("document_preprocessors", [])
        ]

        self.input_preprocessors = [
            get_input_preprocessor(ip["type"], ip.get("params", {}))
            for ip in config.get("input_preprocessors", [])
        ]

        if "retrievers" in config:
            self.retrievers = [
                get_retriever(ret["type"], ret.get("params", {}))
                for ret in config["retrievers"]
            ]
        else:
            self.retrievers = [
                get_retriever(
                    config["retriever"]["type"], config["retriever"].get("params", {})
                )
            ]

        self.postprocessors = [
            get_postprocessor(pp["type"], pp.get("params", {}))
            for pp in config.get("postprocessors", [])
        ]

        self._retriever_top_k_override: Optional[int] = None
        self._postprocessor_top_k_override: Optional[int] = None

    def set_overrides(
        self,
        retriever_top_k: Optional[int] = None,
        postprocessor_top_k: Optional[int] = None,
    ) -> None:
        self._retriever_top_k_override = retriever_top_k
        self._postprocessor_top_k_override = postprocessor_top_k

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            raise ValueError("Documents list is empty.")

        document_metadata = {
            doc["id"]: build_document_metadata(str(doc["id"])) for doc in documents
        }
        metadata_catalog: Dict[str, Dict[str, List[str]]] = {}
        for doc_id, metadata in document_metadata.items():
            if metadata["resource_type"] != "product":
                continue
            product_name = str(metadata["product_name_candidate"])
            if product_name == "unknown":
                continue
            category = str(metadata["product_category"])
            metadata_catalog.setdefault(category, {}).setdefault(
                product_name.title(), []
            ).append(doc_id)
        self.state["document_metadata"] = document_metadata
        self.state["metadata_catalog"] = metadata_catalog

        for preprocessor in self.doc_preprocessors:
            documents = preprocessor.process(documents, self.state)

        self.state["documents"] = documents

        self.state["doc_content_map"] = {}
        self.state["doc_title_map"] = {}
        for doc in documents:
            doc_id = doc["id"]
            content = doc.get("text") or doc.get("content") or ""
            title = doc.get("title", doc_id)
            self.state["doc_content_map"][doc_id] = content
            self.state["doc_title_map"][doc_id] = title

    def retrieve(
        self, query: str, top_k: int = None, return_timing: bool = False
    ) -> List[Tuple[str, float]] | RetrievalResult:
        if "documents" not in self.state:
            raise ValueError("No documents indexed. Call index_documents() first.")

        timing = RetrievalTiming()

        input_data: Dict[str, Any] = {"query": query}
        if "kb_search_inputs" in self.state:
            input_data.update(self.state["kb_search_inputs"])

        preprocess_start = time.perf_counter()
        for preprocessor in self.input_preprocessors:
            input_data = preprocessor.process(input_data, self.state)
        timing.input_preprocessing_ms = (time.perf_counter() - preprocess_start) * 1000

        original_retriever_top_ks = []
        if self._retriever_top_k_override is not None:
            for retriever in self.retrievers:
                if hasattr(retriever, "top_k"):
                    original_retriever_top_ks.append((retriever, retriever.top_k))
                    retriever.top_k = self._retriever_top_k_override

        retrieval_start = time.perf_counter()
        all_results: Dict[str, float] = {}
        for retriever in self.retrievers:
            retriever_results = retriever.retrieve(input_data, self.state)
            for doc_id, score in retriever_results:
                if doc_id not in all_results or score > all_results[doc_id]:
                    all_results[doc_id] = score
        timing.retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        for retriever, orig_top_k in original_retriever_top_ks:
            retriever.top_k = orig_top_k

        results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            results = results[:top_k]

        original_postprocessor_top_ks = []
        if self._postprocessor_top_k_override is not None:
            for postprocessor in self.postprocessors:
                if hasattr(postprocessor, "top_k"):
                    original_postprocessor_top_ks.append(
                        (postprocessor, postprocessor.top_k)
                    )
                    postprocessor.top_k = self._postprocessor_top_k_override

        postprocess_start = time.perf_counter()
        for postprocessor in self.postprocessors:
            pp_start = time.perf_counter()
            results = postprocessor.process(results, input_data, self.state)
            pp_name = postprocessor.__class__.__name__
            timing.postprocessor_details[pp_name] = (
                time.perf_counter() - pp_start
            ) * 1000
        timing.postprocessing_ms = (time.perf_counter() - postprocess_start) * 1000

        for postprocessor, orig_top_k in original_postprocessor_top_ks:
            postprocessor.top_k = orig_top_k

        if return_timing:
            return RetrievalResult(results=results, timing=timing)
        return results

    def retrieve_batch(
        self, queries: List[str], top_k: int = None
    ) -> List[List[Tuple[str, float]]]:
        if "documents" not in self.state:
            raise ValueError("No documents indexed. Call index_documents() first.")

        extra_inputs = self.state.get("kb_search_inputs", {})
        input_data_list = [{"query": query, **extra_inputs} for query in queries]

        for preprocessor in self.input_preprocessors:
            input_data_list = preprocessor.process_batch(input_data_list, self.state)

        original_retriever_top_ks = []
        if self._retriever_top_k_override is not None:
            for retriever in self.retrievers:
                if hasattr(retriever, "top_k"):
                    original_retriever_top_ks.append((retriever, retriever.top_k))
                    retriever.top_k = self._retriever_top_k_override

        combined_results: List[Dict[str, float]] = [{} for _ in input_data_list]
        for retriever in self.retrievers:
            retriever_results_list = retriever.retrieve_batch(
                input_data_list, self.state
            )
            for i, retriever_results in enumerate(retriever_results_list):
                for doc_id, score in retriever_results:
                    if (
                        doc_id not in combined_results[i]
                        or score > combined_results[i][doc_id]
                    ):
                        combined_results[i][doc_id] = score

        for retriever, orig_top_k in original_retriever_top_ks:
            retriever.top_k = orig_top_k

        results_list = []
        for result_dict in combined_results:
            sorted_results = sorted(
                result_dict.items(), key=lambda x: x[1], reverse=True
            )
            if top_k is not None:
                sorted_results = sorted_results[:top_k]
            results_list.append(sorted_results)

        original_postprocessor_top_ks = []
        if self._postprocessor_top_k_override is not None:
            for postprocessor in self.postprocessors:
                if hasattr(postprocessor, "top_k"):
                    original_postprocessor_top_ks.append(
                        (postprocessor, postprocessor.top_k)
                    )
                    postprocessor.top_k = self._postprocessor_top_k_override

        for postprocessor in self.postprocessors:
            results_list = postprocessor.process_batch(
                results_list, input_data_list, self.state
            )

        for postprocessor, orig_top_k in original_postprocessor_top_ks:
            postprocessor.top_k = orig_top_k

        return results_list

    def get_document_content(self, doc_id: str) -> Optional[str]:
        return self.state.get("doc_content_map", {}).get(doc_id)

    def get_document_title(self, doc_id: str) -> Optional[str]:
        return self.state.get("doc_title_map", {}).get(doc_id)

    def get_metadata_catalog(self) -> Dict[str, Dict[str, List[str]]]:
        """Return the indexed product/category metadata catalog."""
        return self.state.get("metadata_catalog", {})

    def apply_product_catalog_review(
        self, review: Dict[str, Dict[str, List[str]]]
    ) -> None:
        """Validate and filter ID-derived product candidates using a review catalog."""
        candidates = self.get_metadata_catalog()
        candidate_categories = set(candidates)
        reviewed_categories = set(review)
        if candidate_categories != reviewed_categories:
            missing = sorted(candidate_categories - reviewed_categories)
            stale = sorted(reviewed_categories - candidate_categories)
            raise ValueError(
                "Product catalog review categories are out of sync with the "
                f"knowledge base; unreviewed={missing}, stale={stale}"
            )

        confirmed_catalog: Dict[str, Dict[str, List[str]]] = {}
        for category, candidate_products in candidates.items():
            category_review = review[category]
            confirmed = list(category_review.get("products", []))
            excluded = list(category_review.get("excluded_candidates", []))
            reviewed_names = confirmed + excluded
            duplicate_names = sorted(
                {name for name in reviewed_names if reviewed_names.count(name) > 1}
            )
            if duplicate_names:
                raise ValueError(
                    f"Duplicate product catalog review names for {category}: "
                    f"{duplicate_names}"
                )
            candidate_names = set(candidate_products)
            reviewed_name_set = set(reviewed_names)
            if candidate_names != reviewed_name_set:
                unreviewed = sorted(candidate_names - reviewed_name_set)
                stale = sorted(reviewed_name_set - candidate_names)
                raise ValueError(
                    f"Product catalog review for {category} is out of sync with "
                    f"the knowledge base; unreviewed={unreviewed}, stale={stale}"
                )
            confirmed_catalog[category] = {
                name: candidate_products[name] for name in confirmed
            }
        self.state["metadata_catalog_candidates"] = candidates
        self.state["metadata_catalog"] = confirmed_catalog

    def save_state(self, path: str) -> None:
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "wb") as f:
            pickle.dump(self.state, f)

    def load_state(self, path: str) -> None:
        with open(path, "rb") as f:
            self.state = pickle.load(f)

    def get_name(self) -> str:
        parts = []
        for dp in self.config.get("document_preprocessors", []):
            parts.append(dp["type"])
        if "retrievers" in self.config:
            retriever_names = [ret["type"] for ret in self.config["retrievers"]]
            parts.append("+".join(retriever_names))
        else:
            parts.append(self.config["retriever"]["type"])
        for pp in self.config.get("postprocessors", []):
            parts.append(pp["type"])
        return "_".join(parts)
