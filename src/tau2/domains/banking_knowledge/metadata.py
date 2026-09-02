"""Reviewed metadata used to disambiguate ID-derived product candidates."""

import json
from typing import Any

from tau2.domains.banking_knowledge.utils import KNOWLEDGE_METADATA_DIR

PRODUCT_CATALOG_REVIEW_PATH = KNOWLEDGE_METADATA_DIR / "product_catalog.json"


def load_product_catalog_review() -> dict[str, dict[str, list[str]]]:
    """Load the reviewed product names for the current banking knowledge base."""
    with open(PRODUCT_CATALOG_REVIEW_PATH) as file:
        payload: dict[str, Any] = json.load(file)
    if payload.get("schema_version") != 1:
        raise ValueError(
            "Unsupported product catalog review schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    categories = payload.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("Product catalog review must contain a categories object")
    for category, review in categories.items():
        if not isinstance(review, dict):
            raise ValueError(f"Product catalog review for {category} must be an object")
        for field in ("products", "excluded_candidates"):
            names = review.get(field)
            if not isinstance(names, list) or not all(
                isinstance(name, str) for name in names
            ):
                raise ValueError(
                    f"Product catalog review {category}.{field} must be a string list"
                )
    return categories
