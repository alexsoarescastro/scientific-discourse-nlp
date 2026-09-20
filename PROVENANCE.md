# Provenance and reconstruction audit

## Recovered/project-supported elements
- Corpus size reported in manuscript: 5,127 publications, 2000–2025.
- Data source in revised manuscript: OpenAlex.
- SciBERT contextual embeddings, 768-dimensional representation.
- BERTopic thematic modeling.
- 37 interpretable topics.
- UMAP + HDBSCAN structure.
- Reconstructed BERTopic UMAP: n_neighbors=15, n_components=5, metric=cosine, random_state=42.
- Reconstructed HDBSCAN: min_cluster_size=15, cluster_selection_method=eom.
- C_v topic coherence is the reconstructed coherence metric and must be empirically rechecked.
- MATTR window reconstructed as 50.
- Linguistic variables in revised manuscript: lexical density, MATTR, hedging frequency, boosting frequency.
- Random Forest + SHAP in IP&M manuscript.
- Revised validation requirement: stratified 5-fold CV.
- SHAP figure values sum to 1 and are therefore treated as normalized mean absolute SHAP importance.

## Deliberately not fabricated
- Exact original source-code bytes.
- Exact package versions.
- Exact historical hedge/booster inventories.
- Exact T1–T4 assignment of all 37 topics.
- Exact experimental extraction regex/rules from the original run.
- Random-Forest performance metrics for the revised T1–T4 task.
- Confusion-matrix cells for the revised task.
- Exact raw mean absolute SHAP values.
- Exact OpenAlex query string/PRISMA counts.
- Exact full-text coverage count.

These must be recovered from the original data/code or recomputed. The code package provides reproducible implementations for recomputation.
