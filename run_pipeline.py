import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils import load_config, ensure_dirs, set_seed
from src.preprocess import preprocess_dataframe
from src.embeddings import SciBERTEmbedder
from src.topics import fit_topics, topic_table
from src.semantic_map import make_2d_umap
from src.scientometrics import annual_publications
from src.temporal import annual_topic_counts
from src.figures import fig_annual_publications

def main(config_path):
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    set_seed(cfg["seed"])

    raw = Path(cfg["paths"]["raw_csv"])
    if not raw.exists():
        raise FileNotFoundError(
            f"{raw} not found. Add the study corpus before running empirical analyses."
        )

    df = pd.read_csv(raw)
    df = preprocess_dataframe(df)

    emb_path = Path(cfg["paths"]["embeddings_npy"])
    if emb_path.exists():
        embeddings = np.load(emb_path)
    else:
        embedder = SciBERTEmbedder(
            cfg["scibert"]["model_name"],
            cfg["scibert"]["max_length"],
        )
        embeddings = embedder.encode(df["text_semantic"].fillna("").tolist())
        np.save(emb_path, embeddings)

    model, topics, probabilities = fit_topics(
        df["text_semantic"].fillna("").tolist(), embeddings, cfg
    )
    df["topic"] = topics
    model.save(cfg["paths"]["topic_model"], serialization="pickle")

    topic_table(model).to_csv(
        Path(cfg["paths"]["output_tables"]) / "topics.csv", index=False
    )
    annual_publications(df).to_csv(
        Path(cfg["paths"]["output_tables"]) / "annual_publications.csv", index=False
    )
    annual_topic_counts(df).to_csv(
        Path(cfg["paths"]["output_tables"]) / "annual_topic_counts.csv", index=False
    )

    z = make_2d_umap(embeddings, cfg)
    df = pd.concat([df.reset_index(drop=True), z], axis=1)

    # Higher-order thematic domains (T1–T4) are an interpretive aggregation
    # of the recovered topic structure. They are not generated here by an
    # automatic classifier. If a reviewed topic-to-domain mapping is supplied,
    # merge it downstream before domain-level temporal analysis.

    df.to_csv(cfg["paths"]["processed_csv"], index=False)
    fig_annual_publications(
        df, Path(cfg["paths"]["output_figures"]) / "annual_publications.png"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
