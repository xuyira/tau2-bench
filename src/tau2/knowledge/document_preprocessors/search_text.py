"""Shared formatting for text indexed by sparse and dense retrievers."""

import re
from typing import Any, Dict

_PRODUCT_NAMESPACES = {
    "checking_accounts": "checking_account",
    "business_checking_accounts": "business_checking_account",
    "savings_accounts": "savings_account",
    "business_savings_accounts": "business_savings_account",
    "credit_cards": "credit_card",
    "business_credit_cards": "business_credit_card",
}
_SERVICE_PREFIXES = (
    "buy_now_pay_later",
    "everyone_pay",
    "personal_subscriptions",
)
_TOPIC_PREFIXES = ("bank_accounts", "customer_support")


def normalize_document_id(document_id: str) -> str:
    """Split an ID into searchable words without its ``doc`` prefix or index."""
    without_prefix = re.sub(r"^doc_", "", document_id)
    without_index = re.sub(r"_\d+$", "", without_prefix)
    return without_index.replace("_", " ")


def build_document_metadata(document_id: str) -> Dict[str, Any]:
    """Parse conservative, ID-derived metadata for a knowledge document.

    Values are intentionally limited to facts represented by the ID. In
    particular, the absence of ``business`` does not imply a personal segment.
    """
    raw_id = str(document_id)
    without_prefix = re.sub(r"^doc_", "", raw_id)
    index_match = re.search(r"_(\d+)$", without_prefix)
    document_index = int(index_match.group(1)) if index_match else "unknown"
    stem = without_prefix[: index_match.start()] if index_match else without_prefix
    normalized_id = stem.replace("_", " ")
    id_tokens = normalized_id.split()
    is_general = stem.endswith("_(general)")

    resource_type = "unknown"
    product_category = "unknown"
    product_name_candidate = "unknown"
    service_category = "unknown"
    topic_category = "unknown"
    for candidate, category in _PRODUCT_NAMESPACES.items():
        if stem == candidate or stem.startswith(f"{candidate}_"):
            resource_type = "product"
            product_category = category
            suffix = stem[len(candidate) :].lstrip("_")
            if suffix and not is_general:
                product_name_candidate = suffix.replace("_", " ")
            break
    else:
        service_prefix = next(
            (
                prefix
                for prefix in _SERVICE_PREFIXES
                if stem == prefix or stem.startswith(f"{prefix}_")
            ),
            None,
        )
        topic_prefix = None
        if service_prefix:
            resource_type = "service"
            service_category = service_prefix
        else:
            topic_prefix = next(
                (
                    prefix
                    for prefix in _TOPIC_PREFIXES
                    if stem == prefix or stem.startswith(f"{prefix}_")
                ),
                None,
            )
        if resource_type == "unknown" and topic_prefix:
            resource_type = "topic"
            topic_category = topic_prefix

    return {
        "document_id": raw_id,
        "normalized_id": normalized_id,
        "document_index": document_index,
        "resource_type": resource_type,
        "customer_segment": "business" if "business" in id_tokens else "unknown",
        "product_category": product_category,
        "product_name_candidate": product_name_candidate,
        "service_category": service_category,
        "topic_category": topic_category,
        "is_general": is_general,
    }


def build_document_search_text(
    document: Dict[str, Any], content_field: str = "text"
) -> str:
    """Combine a normalized document ID, title, and content for retrieval."""
    content = (
        document.get(content_field) or document.get("content") or document.get("text")
    )
    if content is None:
        raise ValueError(
            f"Document {document.get('id', 'unknown')} missing content field"
        )

    metadata = build_document_metadata(str(document.get("id", "")))
    normalized_id = str(document.get("normalized_id", metadata["normalized_id"]))
    title = str(document.get("title", ""))
    section = document.get("section")
    section_text = f"Section: {section}\n" if section else ""
    return f"ID: {normalized_id}\nTitle: {title}\n{section_text}Content:\n{content}"
