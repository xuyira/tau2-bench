from tau2.knowledge.pipeline import RetrievalPipeline


def test_metadata_catalog_groups_products_by_category_before_chunking():
    pipeline = RetrievalPipeline({"retriever": {"type": "bm25", "params": {}}})
    pipeline.index_documents(
        [
            {
                "id": "doc_checking_accounts_blue_account_001",
                "text": "Blue referral terms",
            },
            {
                "id": "doc_checking_accounts_blue_account_010",
                "text": "Blue qualification details",
            },
            {
                "id": "doc_checking_accounts_green_fee-free_account_007",
                "text": "Green referral terms",
            },
            {
                "id": "doc_bank_accounts_bank_accounts_(general)_047",
                "text": "General terms",
            },
        ]
    )

    catalog = pipeline.get_metadata_catalog()

    assert catalog["checking_account"]["Blue Account"] == [
        "doc_checking_accounts_blue_account_001",
        "doc_checking_accounts_blue_account_010",
    ]
    assert catalog["checking_account"]["Green Fee-Free Account"] == [
        "doc_checking_accounts_green_fee-free_account_007"
    ]
    assert "bank_accounts" not in catalog
    assert (
        pipeline.state["document_metadata"]["doc_checking_accounts_blue_account_001"][
            "product_category"
        ]
        == "checking_account"
    )


def test_metadata_catalog_refreshes_when_knowledge_base_is_reindexed():
    pipeline = RetrievalPipeline({"retriever": {"type": "bm25", "params": {}}})
    pipeline.index_documents(
        [
            {
                "id": "doc_checking_accounts_blue_account_001",
                "text": "Blue referral terms",
            }
        ]
    )
    assert "Blue Account" in pipeline.get_metadata_catalog()["checking_account"]

    pipeline.index_documents(
        [
            {
                "id": "doc_checking_accounts_purple_account_001",
                "text": "Purple referral terms",
            }
        ]
    )
    catalog = pipeline.get_metadata_catalog()
    assert "Purple Account" in catalog["checking_account"]
    assert "Blue Account" not in catalog.get("checking_account", {})
    assert (
        "doc_checking_accounts_blue_account_001"
        not in pipeline.state["document_metadata"]
    )


def test_product_catalog_review_filters_candidates_but_keeps_dynamic_doc_ids():
    pipeline = RetrievalPipeline({"retriever": {"type": "bm25", "params": {}}})
    pipeline.index_documents(
        [
            {
                "id": "doc_credit_cards_gold_rewards_card_001",
                "text": "Rewards",
            },
            {
                "id": "doc_credit_cards_credit_card_replacements_001",
                "text": "Replacement process",
            },
        ]
    )

    pipeline.apply_product_catalog_review(
        {
            "credit_card": {
                "products": ["Gold Rewards Card"],
                "excluded_candidates": ["Credit Card Replacements"],
            }
        }
    )

    assert pipeline.get_metadata_catalog() == {
        "credit_card": {"Gold Rewards Card": ["doc_credit_cards_gold_rewards_card_001"]}
    }
    assert (
        "Credit Card Replacements"
        in pipeline.state["metadata_catalog_candidates"]["credit_card"]
    )


def test_product_catalog_review_rejects_unreviewed_candidates():
    pipeline = RetrievalPipeline({"retriever": {"type": "bm25", "params": {}}})
    pipeline.index_documents(
        [
            {
                "id": "doc_credit_cards_new_card_001",
                "text": "New card",
            }
        ]
    )

    try:
        pipeline.apply_product_catalog_review(
            {
                "credit_card": {
                    "products": [],
                    "excluded_candidates": [],
                }
            }
        )
    except ValueError as exc:
        assert "unreviewed=['New Card']" in str(exc)
    else:
        raise AssertionError("Expected an out-of-sync product review error")


def test_scoped_retrieval_reports_all_missing_products_without_truncation():
    documents = [
        {
            "id": f"doc_checking_accounts_product_{index:02d}_account_001",
            "text": "referral bonus terms",
        }
        for index in range(12)
    ]
    pipeline = RetrievalPipeline(
        {
            "document_preprocessors": [
                {"type": "bm25_indexer", "params": {"state_key": "bm25"}}
            ],
            "retriever": {
                "type": "bm25",
                "params": {
                    "bm25_state_key": "bm25",
                    "doc_ids_state_key": "bm25_doc_ids",
                    "top_k": 1,
                },
            },
        }
    )
    pipeline.index_documents(documents)

    result = pipeline.retrieve(
        "referral bonus",
        top_k=2,
        return_timing=True,
        retrieval_scope={
            "product_category": "checking_account",
            "coverage": "all_products",
        },
    )

    assert len(result.results) == 2
    assert result.coverage is not None
    assert len(result.coverage["requested_products"]) == 12
    assert len(result.coverage["covered_products"]) == 2
    assert len(result.coverage["missing_products"]) == 10
    assert result.coverage["coverage_complete"] is False


def test_scoped_retrieval_can_continue_with_all_missing_products():
    pipeline = RetrievalPipeline(
        {
            "document_preprocessors": [
                {"type": "bm25_indexer", "params": {"state_key": "bm25"}}
            ],
            "retriever": {
                "type": "bm25",
                "params": {
                    "bm25_state_key": "bm25",
                    "doc_ids_state_key": "bm25_doc_ids",
                    "top_k": 1,
                },
            },
        }
    )
    pipeline.index_documents(
        [
            {
                "id": "doc_checking_accounts_blue_account_010",
                "text": "Blue referral bonus",
            },
            {
                "id": "doc_checking_accounts_green_account_010",
                "text": "Green referral bonus",
            },
        ]
    )

    result = pipeline.retrieve(
        "referral bonus",
        return_timing=True,
        retrieval_scope={
            "product_category": "checking_account",
            "product_names": ["Blue Account", "Green Account"],
            "coverage": "all_products",
        },
    )

    assert {doc_id for doc_id, _ in result.results} == {
        "doc_checking_accounts_blue_account_010",
        "doc_checking_accounts_green_account_010",
    }
    assert result.coverage is not None
    assert result.coverage["missing_products"] == []
    assert result.coverage["coverage_complete"] is True
