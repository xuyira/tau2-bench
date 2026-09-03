from tau2.knowledge.pipeline import RetrievalPipeline


def test_metadata_catalog_groups_products_by_category_before_chunking():
    pipeline = RetrievalPipeline({"retriever": {"type": "bm25", "params": {}}})
    pipeline.index_documents(
        [
            {"id": "doc_checking_accounts_blue_account_001", "text": "Blue"},
            {"id": "doc_checking_accounts_blue_account_010", "text": "Blue"},
            {
                "id": "doc_checking_accounts_green_fee-free_account_007",
                "text": "Green",
            },
            {
                "id": "doc_bank_accounts_bank_accounts_(general)_047",
                "text": "General",
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


def test_metadata_catalog_refreshes_when_knowledge_base_is_reindexed():
    pipeline = RetrievalPipeline({"retriever": {"type": "bm25", "params": {}}})
    pipeline.index_documents(
        [{"id": "doc_checking_accounts_blue_account_001", "text": "Blue"}]
    )
    pipeline.index_documents(
        [{"id": "doc_checking_accounts_purple_account_001", "text": "Purple"}]
    )
    catalog = pipeline.get_metadata_catalog()
    assert "Purple Account" in catalog["checking_account"]
    assert "Blue Account" not in catalog["checking_account"]


def test_product_catalog_review_filters_candidates_but_keeps_dynamic_doc_ids():
    pipeline = RetrievalPipeline({"retriever": {"type": "bm25", "params": {}}})
    pipeline.index_documents(
        [
            {"id": "doc_credit_cards_gold_rewards_card_001", "text": "Rewards"},
            {
                "id": "doc_credit_cards_credit_card_replacements_001",
                "text": "Replacement",
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


def test_product_catalog_review_rejects_unreviewed_candidates():
    pipeline = RetrievalPipeline({"retriever": {"type": "bm25", "params": {}}})
    pipeline.index_documents(
        [{"id": "doc_credit_cards_new_card_001", "text": "New card"}]
    )
    try:
        pipeline.apply_product_catalog_review(
            {"credit_card": {"products": [], "excluded_candidates": []}}
        )
    except ValueError as exc:
        assert "unreviewed=['New Card']" in str(exc)
    else:
        raise AssertionError("Expected an out-of-sync product review error")
