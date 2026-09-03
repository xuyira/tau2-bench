"""Retrieval capability MixIns for the banking_knowledge domain.

Each MixIn provides a single retrieval tool via @is_tool methods.
The ToolKitType metaclass collects these tools through MRO when
mixed into a concrete toolkit class.

MixIns define no __init__ — they expect the concrete class to set
the required attributes (e.g., _kb_pipeline, _grep_pipeline, _sandbox)
during its own __init__.
"""

import json
from typing import Literal, Optional

from tau2.environment.toolkit import ToolKitType, ToolType, is_tool


def _format_kb_search_result(pipeline, retrieval_result) -> str:
    """Format timed KB search results for all KB search tool variants."""
    results = retrieval_result.results
    timing = retrieval_result.timing

    coverage_output = ""
    if retrieval_result.coverage is not None:
        coverage_output = (
            "[Product Coverage]\n"
            f"{json.dumps(retrieval_result.coverage, ensure_ascii=False)}\n\n"
        )

    if not results:
        output = coverage_output + "No relevant documents found."
        output += f"\n\n[Timing: retrieval={timing.retrieval_ms:.0f}ms"
        if timing.postprocessing_ms > 0:
            output += f", reranking={timing.postprocessing_ms:.0f}ms"
        output += f", total={timing.total_ms:.0f}ms]"
        return output

    formatted = []
    for i, (doc_id, score) in enumerate(results, 1):
        title = pipeline.get_document_title(doc_id) or "Untitled"
        content = pipeline.get_document_content(doc_id) or ""
        formatted.append(
            f"{i}. {title}\n"
            f"   ID: {doc_id}\n"
            f"   Score: {score:.4f}\n"
            f"   Content: {content}\n"
        )

    output = coverage_output + "\n".join(formatted)
    output += f"\n\n[Timing: retrieval={timing.retrieval_ms:.0f}ms"
    if timing.postprocessing_ms > 0:
        output += f", reranking={timing.postprocessing_ms:.0f}ms"
    output += f", total={timing.total_ms:.0f}ms]"
    return output


def _run_kb_search(
    pipeline,
    query: str,
    top_k: int | None = None,
    retrieval_scope: dict | None = None,
) -> str:
    """Run a KB search pipeline with timing and shared formatting."""
    retrieve_kwargs = {"return_timing": True}
    if top_k is not None:
        retrieve_kwargs["top_k"] = top_k
    if retrieval_scope is not None:
        retrieve_kwargs["retrieval_scope"] = retrieval_scope
    retrieval_result = pipeline.retrieve(query, **retrieve_kwargs)
    return _format_kb_search_result(pipeline, retrieval_result)


def _run_all_products_search(pipeline, query: str, product_category: str,
                             product_names: Optional[list[str]] = None,
                             max_attempts: int = 3) -> str:
    """Run product coverage search and repair missing products automatically.

    The first pass uses the caller's category-level query. Subsequent passes are
    restricted to the complete list reported as missing by the pipeline. This is
    intentionally deterministic and keeps the normal relevance path unchanged.
    """
    requested = list(product_names) if product_names else None
    original_requested: Optional[list[str]] = None
    seen_docs: set[str] = set()
    merged: list[tuple[str, float]] = []
    covered: list[str] = []
    last_missing: list[str] = []
    total_timing = None
    search_query = query

    for attempt in range(max_attempts):
        scope = {
            "product_category": product_category,
            "product_names": requested,
            "coverage": "all_products",
        }
        result = pipeline.retrieve(search_query, return_timing=True,
                                   retrieval_scope=scope)
        if total_timing is None:
            total_timing = result.timing
        else:
            total_timing.input_preprocessing_ms += result.timing.input_preprocessing_ms
            total_timing.retrieval_ms += result.timing.retrieval_ms
            total_timing.postprocessing_ms += result.timing.postprocessing_ms
            for name, value in result.timing.postprocessor_details.items():
                total_timing.postprocessor_details[name] = (
                    total_timing.postprocessor_details.get(name, 0.0) + value
                )
        coverage = result.coverage or {}
        if original_requested is None:
            original_requested = list(coverage.get("requested_products", requested or []))
        for product in coverage.get("covered_products", []):
            if product not in covered:
                covered.append(product)
        for doc_id, score in result.results:
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                merged.append((doc_id, score))
        last_missing = [p for p in (original_requested or []) if p not in covered]
        if not last_missing:
            break
        if attempt + 1 >= max_attempts:
            break
        # A scoped retry is useful only if it targets a smaller/new set.
        if requested is not None and set(last_missing) == set(requested):
            search_query = f"{query} Products: {', '.join(last_missing)} Complete product documentation"
        requested = last_missing

    if total_timing is None:
        return "No relevant documents found."
    final_coverage = {
        "product_category": product_category,
        "requested_products": original_requested or [],
        "covered_products": covered,
        "missing_products": last_missing,
        "coverage_complete": not last_missing,
    }
    # Reuse the normal formatter with a lightweight result carrying merged output.
    return _format_kb_search_result(
        pipeline,
        type("_CoverageResult", (), {
            "results": merged,
            "timing": total_timing,
            "coverage": final_coverage,
        })(),
    )


class KBSearchMixin(metaclass=ToolKitType):
    """MixIn that provides the KB_search tool.

    Expects ``self._kb_pipeline`` (a RetrievalPipeline) to be set by the
    concrete class before any tool calls.
    """

    @is_tool(ToolType.READ)
    def KB_search(
        self,
        query: str,
        product_category: Optional[str] = None,
        product_names: Optional[list[str]] = None,
        coverage: Literal["relevance", "all_products"] = "relevance",
    ) -> str:
        """Search the knowledge base for relevant documents.

        Args:
            query: The search query to find relevant documents
            product_category: Product category from the runtime metadata catalog.
            product_names: Products to search within the category. Pass all missing
                products from the previous result when continuing coverage.
            coverage: Use all_products to return at most one full document per
                product and report covered and missing products.

        Returns:
            Relevant document excerpts matching the query
        """
        # TODO: clean up knowledge retrieval pipelines to return structure results
        if coverage == "all_products" and not product_category:
            raise ValueError(
                "product_category is required when coverage is all_products"
            )
        if product_names and not product_category:
            raise ValueError("product_category is required with product_names")
        if coverage == "all_products":
            return _run_all_products_search(
                self._kb_pipeline,
                query,
                product_category,
                product_names,
            )
        retrieval_scope = None
        if coverage == "all_products" or product_category or product_names:
            retrieval_scope = {
                "product_category": product_category,
                "product_names": product_names,
                "coverage": coverage,
            }
        return _run_kb_search(
            self._kb_pipeline,
            query,
            retrieval_scope=retrieval_scope,
        )


class GrepMixin(metaclass=ToolKitType):
    """MixIn that provides the grep tool.

    Expects ``self._grep_pipeline`` (a RetrievalPipeline) to be set.
    """

    @is_tool(ToolType.READ)
    def grep(self, pattern: str) -> str:
        """Search for a regex pattern in all knowledge base documents.

        Returns documents ranked by number of matches, with full content.

        Args:
            pattern: The regex pattern to search for (e.g., 'credit.*card', 'fee|charge')

        Returns:
            Matching documents ranked by relevance (match count)
        """
        results = self._grep_pipeline.retrieve(pattern)

        if not results:
            return f"No matches found for pattern: {pattern}"

        formatted = []
        for i, (doc_id, score) in enumerate(results, 1):
            title = self._grep_pipeline.get_document_title(doc_id) or "Untitled"
            content = self._grep_pipeline.get_document_content(doc_id) or ""
            formatted.append(
                f"{i}. {title}\n"
                f"   ID: {doc_id}\n"
                f"   Score: {score:.4f}\n"
                f"   Content: {content}\n"
            )

        return "\n".join(formatted)


class KBSearchBm25AllToolsMixin(metaclass=ToolKitType):
    """BM25 search for AllTools; expects ``self._kb_bm25_pipeline``."""

    @is_tool(ToolType.READ)
    def KB_search_bm25(self, query: str, k: int = 10) -> str:
        """Search the knowledge base using BM25 sparse retrieval.

        Args:
            query: The search query to find relevant documents.
            k: Maximum number of documents to return (default 10).

        Returns:
            Relevant document excerpts matching the query.
        """
        return _run_kb_search(self._kb_bm25_pipeline, query, top_k=k)


class KBSearchDenseAllToolsMixin(metaclass=ToolKitType):
    """Dense embedding search for AllTools; expects ``self._kb_dense_pipeline``."""

    @is_tool(ToolType.READ)
    def KB_search_dense(self, query: str, k: int = 10) -> str:
        """Search the knowledge base using dense embedding retrieval.

        Args:
            query: The search query to find relevant documents.
            k: Maximum number of documents to return (default 10).

        Returns:
            Relevant document excerpts matching the query.
        """
        return _run_kb_search(self._kb_dense_pipeline, query, top_k=k)


class ShellMixin(metaclass=ToolKitType):
    """MixIn that provides the shell tool.

    Expects ``self._sandbox`` (a SandboxManager) to be set.
    """

    @is_tool(ToolType.READ)
    def shell(self, command: str) -> str:
        """Execute a shell command in the knowledge base directory.

        Use standard Unix utilities to explore and search the knowledge base files.
        Common commands: ls, cat, grep, head, tail, find, wc, awk, sed, etc.

        Args:
            command: The shell command to execute (e.g., "ls -la", "grep -r 'credit card' .", "cat INDEX.txt")

        Returns:
            The command output (stdout) or error message
        """
        if self._sandbox is None:
            return "Error: Sandbox not initialized"

        ret_code, stdout, stderr = self._sandbox.run_command(command)

        if ret_code != 0:
            # grep returns 1 when no matches (not an error)
            if ret_code == 1 and "grep" in command and not stderr:
                return "No matches found."
            if stderr:
                return f"Error (exit code {ret_code}): {stderr}"
            return f"Command failed with exit code {ret_code}"

        return stdout if stdout else "(no output)"


class RewriteContextMixin(metaclass=ToolKitType):
    """MixIn that provides the rewrite_context tool for summarization."""

    @is_tool(ToolType.WRITE, mutates_state=False)
    def rewrite_context(self, new_context: str) -> str:
        """Replace your working context with a new summary or condensed version.

        Use this after searching to summarize findings, extract key points,
        or condense information for easier reference later in the conversation.

        Args:
            new_context: The summarized or rewritten context to store

        Returns:
            The context you provided, for reference
        """
        return f"Context updated:\n\n{new_context}"
