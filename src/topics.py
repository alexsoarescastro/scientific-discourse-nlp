import numpy as np
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

def build_topic_model(cfg):
    c = cfg["bertopic"]
    umap_model = UMAP(
        n_neighbors=c["umap_n_neighbors"],
        n_components=c["umap_n_components"],
        min_dist=c["umap_min_dist"],
        metric=c["umap_metric"],
        random_state=cfg["seed"],
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=c["hdbscan_min_cluster_size"],
        metric=c["hdbscan_metric"],
        cluster_selection_method=c["hdbscan_cluster_selection_method"],
        prediction_data=True,
    )
    vectorizer = CountVectorizer(stop_words="english")
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        min_topic_size=c["min_topic_size"],
        top_n_words=c["top_n_words"],
        calculate_probabilities=True,
        verbose=True,
    )

def fit_topics(texts, embeddings, cfg):
    model = build_topic_model(cfg)
    topics, probabilities = model.fit_transform(texts, embeddings)
    return model, np.asarray(topics), probabilities

def topic_table(topic_model):
    return topic_model.get_topic_info()
