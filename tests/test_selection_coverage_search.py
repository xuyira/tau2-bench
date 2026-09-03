from types import SimpleNamespace

from tau2.domains.banking_knowledge.retrieval_mixins import _run_all_products_search


def test_all_products_search_retries_only_missing_products():
    calls = []

    class FakePipeline:
        def retrieve(self, query, *, return_timing, retrieval_scope):
            calls.append((query, retrieval_scope))
            requested = retrieval_scope["product_names"]
            if requested is None:
                covered, missing = ["Gold"], ["Silver"]
            else:
                covered, missing = ["Silver"], []
            return SimpleNamespace(
                results=[(f"doc_{covered[0]}", 1.0)],
                timing=SimpleNamespace(
                    input_preprocessing_ms=1.0,
                    retrieval_ms=2.0,
                    postprocessing_ms=3.0,
                    postprocessor_details={},
                    total_ms=6.0,
                ),
                coverage={
                    "requested_products": ["Gold", "Silver"],
                    "covered_products": covered,
                    "missing_products": missing,
                    "coverage_complete": not missing,
                },
            )

        def get_document_title(self, doc_id):
            return doc_id

        def get_document_content(self, doc_id):
            return f"Evidence for {doc_id}"

    output = _run_all_products_search(FakePipeline(), "card rewards", "credit_card")

    assert calls == [
        (
            "card rewards",
            {
                "product_category": "credit_card",
                "product_names": None,
                "coverage": "all_products",
            },
        ),
        (
            "card rewards",
            {
                "product_category": "credit_card",
                "product_names": ["Silver"],
                "coverage": "all_products",
            },
        ),
    ]
    assert '"coverage_complete": true' in output
    assert "doc_Gold" in output
    assert "doc_Silver" in output
