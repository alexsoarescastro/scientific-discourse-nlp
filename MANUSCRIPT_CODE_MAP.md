# Manuscript ↔ code map

| Manuscript component | Code |
|---|---|
| 3.1 Corpus / OpenAlex | `src/openalex.py`, `src/validate.py` |
| 3.2 Preprocessing | `src/preprocess.py` |
| 3.3 SciBERT embeddings | `src/embeddings.py` |
| 3.4 BERTopic / 37 topics / C_v | `src/topics.py` |
| 3.5 2-D UMAP semantic map | `src/semantic_map.py` |
| 3.6 LD / MATTR / HF / BF | `src/linguistics.py`, `resources/*.txt` |
| 3.7 Experimental/scientometric variables | `src/experimental.py`, `src/scientometrics.py` |
| 3.8 Random Forest / 5-fold CV / SHAP | `src/ml.py`, `src/shap_analysis.py` |
| 3.9 Statistical analysis | `src/statistics.py`, `src/posthoc.py` |
| 3.10 Reproducibility | config, requirements, notebook, provenance |
| 4.1 Temporal/scientometric results | `src/scientometrics.py`, `src/temporal.py` |
| 4.2 Semantic/topic results | `src/topics.py`, `src/semantic_map.py` |
| 4.3 Linguistic + RF/SHAP | `src/linguistics.py`, `src/ml.py`, `src/shap_analysis.py` |
| 4.4 Temporal T1–T4 | `src/temporal.py` |
| 4.5 Experimental associations | `src/experimental.py`, `src/statistics.py` |
| Figures | `src/figures.py` |

## Important reproducibility distinction
This repository contains executable code capable of reproducing the analyses described in the revised manuscript from the study data. Historical code that cannot be verified is not represented as original code. Reconstructed settings are documented in `PROVENANCE.md`.
