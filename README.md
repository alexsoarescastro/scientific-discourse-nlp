# Scientific Literature and Technological Research Trajectories

Code and supporting documentation for the computational workflow associated with the manuscript:

**Tracing Technological Research Trajectories Through Scientific Literature: A Semantic and Scientometric Approach**

## Scope

The study combines semantic NLP and scientometric analysis to examine longitudinal changes in a scientific field. Bioelectrochemical systems (BES) are used as the empirical case.

The analytical logic is:

**Corpus construction → text preprocessing → SciBERT → BERTopic / UMAP / HDBSCAN → 37 research topics → four higher-order thematic domains → scientometric + temporal analysis → technological research trajectories**

In this project, *technological research trajectories* refer to longitudinal changes in the organization and orientation of research represented in the scientific literature. They are **not** direct measures of technological maturity, technology readiness level, industrial adoption, or commercialization.

## Repository contents

- `run_pipeline.py` — main semantic/topic-modeling workflow.
- `config/config.yaml` — reported model and dimensionality-reduction settings.
- `src/preprocess.py` — text preparation.
- `src/embeddings.py` — SciBERT document embeddings.
- `src/topics.py` — BERTopic with UMAP and HDBSCAN.
- `src/semantic_map.py` — separate 2-D UMAP projection for visualization.
- `src/scientometrics.py` — descriptive scientometric utilities.
- `src/temporal.py` — topic/domain temporal summaries.
- `src/figures.py` — basic plotting helpers.
- `DATA_DICTIONARY.md` — expected input/output fields.
- `MANUSCRIPT_CODE_MAP.md` — correspondence between manuscript methods and code.
- `PROVENANCE.md` — scope and provenance notes.

## Important provenance note

This repository documents the principal computational procedures described in the manuscript. It should not be interpreted as a claim that the public repository alone reproduces every reported result without access to the study corpus and reviewed analytical mappings.

The original publication texts are not redistributed here because access and redistribution may be subject to database, publisher, copyright, and licensing conditions.

The four higher-order thematic domains are an **interpretive aggregation** of the topic structure, not an additional unsupervised clustering step or an automatically inferred maturity scale. A reviewed topic-to-domain mapping should be used for domain-level temporal analyses.

## Installation

Using pip:

```bash
pip install -r requirements.txt
```

or Conda:

```bash
conda env create -f environment.yml
conda activate scientific-literature-trajectories
```

A compatible English spaCy/SciSpaCy model is also required for preprocessing.

## Running

Place the authorized study corpus at:

```text
data/raw/bes_corpus.csv
```

Then run:

```bash
python run_pipeline.py --config config/config.yaml
```

## Data availability

Bibliographic data were obtained primarily from OpenAlex, with complementary metadata from Crossref when available. Source documents are not redistributed through this repository. Access remains subject to the terms and availability conditions of the original providers.

## License

MIT License. See `LICENSE`.
