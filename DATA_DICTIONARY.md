# Data dictionary

The public repository does not include the study corpus. The pipeline expects a CSV containing the textual and bibliographic fields available for the authorized corpus.

## Core fields

- `title` — publication title.
- `abstract` — publication abstract when available.
- `full_text` — full text when legally/technically available; may be empty.
- `publication_year` — publication year.
- `cited_by_count` — citation count when available.
- `countries` — country information when available for scientometric analysis.

## Generated fields

- `text_minimal` — minimally normalized text selected for analysis.
- `text_semantic` — processed text used by the semantic/topic-modeling workflow.
- `has_full_text` — whether a non-empty full-text field was available.
- `topic` — BERTopic topic identifier.
- `UMAP-1`, `UMAP-2` — coordinates from the separate 2-D semantic visualization.

A reviewed `thematic_domain` field may be merged downstream for analyses using the four higher-order thematic domains (T1–T4). It is not automatically inferred by the cleaned public pipeline.
