# Banking knowledge metadata review

`product_catalog.json` reviews the product-name candidates parsed from document
IDs. It does not store document IDs and does not classify documents as workflow,
policy, or feature.

For every ID-derived category, each candidate name must appear exactly once in
either:

- `products`: confirmed products exposed to planning and `all_products`
  retrieval.
- `excluded_candidates`: names under a product namespace that are services,
  processes, topics, or otherwise not standalone products.

Indexing fails with an `unreviewed`/`stale` report whenever the knowledge base
and this review are out of sync. After adding, deleting, or renaming document IDs
under a product namespace, update this file. Changes to document text, titles,
or additional documents for an existing product name require no metadata edit.

Keep `schema_version` unchanged unless the loader and all consumers are migrated
together.
