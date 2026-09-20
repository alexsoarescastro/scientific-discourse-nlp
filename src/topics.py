import numpy as np, pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

def build_topic_model(cfg):
    c = cfg["bertopic"]
    umap_model = UMAP(
        n_neighbors=c["umap_n_neighbors"], n_components=c["umap_n_components"],
        min_dist=c["umap_min_dist"], metric=c["umap_metric"],
        random_state=cfg["seed"]
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=c["hdbscan_min_cluster_size"],
        metric=c["hdbscan_metric"],
        cluster_selection_method=c["hdbscan_cluster_selection_method"],
        prediction_data=True
    )
    vectorizer = CountVectorizer(stop_words="english")
    return BERTopic(
        umap_model=umap_model, hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        min_topic_size=c["min_topic_size"],
        top_n_words=c["top_n_words"],
        calculate_probabilities=True,
        verbose=True
    )

def fit_topics(texts, embeddings, cfg):
    model = build_topic_model(cfg)
    topics, probs = model.fit_transform(texts, embeddings)
    return model, np.asarray(topics), probs

def cv_coherence(topic_model, tokenized_docs, topn=10):
    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=2)
    topics = []
    topic_ids = [t for t in topic_model.get_topics().keys() if t != -1]
    for tid in topic_ids:
        topics.append([w for w, _ in topic_model.get_topic(tid)[:topn]])
    cm = CoherenceModel(
        topics=topics, texts=tokenized_docs, dictionary=dictionary,
        coherence="c_v"
    )
    per_topic = cm.get_coherence_per_topic()
    return pd.DataFrame({"topic": topic_ids, "c_v": per_topic})

def topic_table(topic_model):
    return topic_model.get_topic_info()

# Domain assignment must be reviewed against the 37 recovered topics.
# This is intentionally explicit rather than pretending an unsupervised model
# automatically knows T1–T4.
DEFAULT_DOMAIN_KEYWORDS = {
    "T1": ["current","power","mfc","ohmic","reactor","anode","cathode","performance"],
    "T2": ["biofilm","electron transfer","extracellular","microbial","substrate","ph"],
    "T3": ["hydrogen","mec","catalyst","electrocatalysis","her","nimo","potential"],
    "T4": ["electrosynthesis","mes","carbon","co2","acetate","glycerol","value-added","chemical"],
}

def assign_topic_domains(topic_model, keyword_map=DEFAULT_DOMAIN_KEYWORDS):
    mapping = {}
    for tid in [x for x in topic_model.get_topics() if x != -1]:
        words = " ".join(w.lower() for w,_ in topic_model.get_topic(tid))
        scores = {d: sum(k in words for k in kws) for d,kws in keyword_map.items()}
        mapping[tid] = max(scores, key=scores.get) if max(scores.values()) > 0 else None
    return mapping
