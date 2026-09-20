# Data dictionary

Minimum input fields:
- `id`: OpenAlex work identifier
- `doi`
- `title`
- `abstract`
- `full_text`: text used when legally/technically available
- `publication_year`
- `cited_by_count`
- `technology_domain`: MFC / MEC / MES where independently assigned
- `countries`: semicolon-separated or serialized list of author-affiliation countries

Derived fields:
- `text_minimal`, `text_semantic`, `has_full_text`
- `topic`: BERTopic topic id
- `thematic_domain`: T1–T4 higher-order thematic domain
- `UMAP-1`, `UMAP-2`
- `lexical_density`, `mattr`, `hedging_frequency`, `boosting_frequency`
- `current_density`, `power_density`, `hydrogen_efficiency`, `applied_potential`, `pH`, `substrate_type`
- `citation_rate_per_year`

Do not include copyrighted full text in a public repository unless redistribution is permitted.
