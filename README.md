# Code accompanying the BES scientific-discourse study

This repository contains the executable analysis pipeline corresponding to the revised manuscript based on 5,127 BES publications (2000–2025).

## Reproduce the analysis
1. Create the environment: `conda env create -f environment.yml`
2. Activate it: `conda activate bes-discourse`
3. Install a compatible English spaCy/scispaCy model.
4. Place the study dataset at `data/raw/openalex_bes.csv`.
5. Run `python run_pipeline.py --config config/config.yaml`, or execute `notebooks/BES_pipeline.ipynb`.

## Included analyses
OpenAlex/corpus handling; preprocessing; SciBERT 768-D document embeddings; BERTopic with UMAP/HDBSCAN/c-TF-IDF; C_v topic coherence; T1–T4 thematic organization; independent 2-D UMAP; lexical density; MATTR; hedging and boosting; experimental-variable extraction; scientometric/temporal analysis; statistical testing; Random Forest with stratified 5-fold CV; confusion matrix; SHAP and normalized mean absolute SHAP importance; manuscript figures/tables.

## Reproducibility caveat
The package is aligned with the **revised manuscript**. Some historical parameters were not preserved in the manuscript/source materials and are explicitly labeled as reconstructed in `PROVENANCE.md`. They must not be described as historically recovered settings. Empirical values must be recomputed from the actual study corpus.

## Data
Do not redistribute copyrighted article full text unless permitted. A public replication dataset may contain metadata, document identifiers/DOIs, derived variables, topic/domain labels, and legally redistributable text only.

See `MANUSCRIPT_CODE_MAP.md`, `DATA_DICTIONARY.md`, and `PROVENANCE.md`.
