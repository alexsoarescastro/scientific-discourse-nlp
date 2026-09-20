import re, numpy as np, pandas as pd, spacy
from collections import Counter

def load_lexicon(path):
    with open(path, encoding="utf-8") as f:
        return {x.strip().lower() for x in f if x.strip() and not x.startswith("#")}

def mattr(tokens, window=50):
    tokens = [t.lower() for t in tokens if t.strip()]
    n = len(tokens)
    if n == 0: return np.nan
    if n <= window: return len(set(tokens))/n
    vals = [len(set(tokens[i:i+window]))/window for i in range(n-window+1)]
    return float(np.mean(vals))

def count_lexicon(text, lexicon):
    # unigram + phrase matching, case-insensitive
    low = text.lower()
    total = 0
    for item in lexicon:
        total += len(re.findall(r"(?<!\w)" + re.escape(item) + r"(?!\w)", low))
    return total

def compute_indicators(text, nlp, hedges, boosters, mattr_window=50):
    doc = nlp(text)
    words = [t.text.lower() for t in doc if not t.is_space and not t.is_punct]
    content = [
        t for t in doc if (t.pos_ in {"NOUN","PROPN","VERB","ADJ"})
        and not t.is_space and not t.is_punct
    ]
    nsent = max(1, sum(1 for _ in doc.sents))
    return {
        "lexical_density": len(content)/len(words) if words else np.nan,
        "mattr": mattr(words, mattr_window),
        "hedging_frequency": count_lexicon(text, hedges)/nsent,
        "boosting_frequency": count_lexicon(text, boosters)/nsent,
        "n_tokens": len(words),
        "n_sentences": nsent,
    }

def add_linguistic_features(df, cfg, hedge_path="resources/hedges.txt",
                            booster_path="resources/boosters.txt"):
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError("Install an English spaCy/scispaCy model.")
    hedges, boosters = load_lexicon(hedge_path), load_lexicon(booster_path)
    feats = df["text_minimal"].map(
        lambda x: compute_indicators(x, nlp, hedges, boosters,
                                     cfg["linguistics"]["mattr_window"])
    )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(feats.tolist())], axis=1)
