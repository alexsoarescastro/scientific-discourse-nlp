# Manuscript–code map

| Manuscript component | Repository implementation |
|---|---|
| Text preprocessing | `src/preprocess.py` |
| SciBERT contextual representation | `src/embeddings.py` |
| BERTopic topic modeling | `src/topics.py` |
| 5-D UMAP + HDBSCAN | `src/topics.py`, `config/config.yaml` |
| c-TF-IDF topic representation | BERTopic configuration in `src/topics.py` |
| 2-D semantic visualization | `src/semantic_map.py` |
| Scientometric summaries | `src/scientometrics.py` |
| Temporal topic/domain summaries | `src/temporal.py` |
| Main semantic workflow | `run_pipeline.py` |

## Higher-order thematic domains

The 37 research topics constitute the primary semantic structure. The four higher-order thematic domains are an interpretive aggregation used for higher-level longitudinal analysis. The cleaned public pipeline therefore does not use an automatic classifier to generate T1–T4.

## Excluded legacy analyses

Random Forest, SHAP, lexical-density/MATTR, hedging/boosting, and experimental-variable prediction modules are not part of the submitted analytical framework and are not included in this submission-aligned package.
