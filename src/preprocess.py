import re, unicodedata, pandas as pd
import spacy

def get_nlp(model="en_core_web_sm"):
    try:
        return spacy.load(model, disable=["ner"])
    except OSError:
        raise RuntimeError("Install a compatible English spaCy/scispaCy model before preprocessing.")

def normalize_unicode(text):
    return unicodedata.normalize("NFKC", str(text or ""))

def minimal_text(text):
    text = normalize_unicode(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def semantic_text(text, nlp):
    """Normalized representation for semantic/thematic analysis.
    Keeps scientific alphabetic tokens; avoids destructive removal of chemical expressions."""
    doc = nlp(minimal_text(text))
    toks = []
    for t in doc:
        if t.is_space or t.is_punct or t.is_stop:
            continue
        if t.like_num:
            continue
        lemma = (t.lemma_ or t.text).strip().lower()
        if lemma and len(lemma) > 1:
            toks.append(lemma)
    return " ".join(toks)

def preprocess_dataframe(df, text_col="full_text", fallback_cols=("abstract","title")):
    nlp = get_nlp()
    def choose(r):
        x = r.get(text_col, "")
        if isinstance(x, str) and x.strip(): return x
        return " ".join(str(r.get(c,"") or "") for c in fallback_cols).strip()
    out = df.copy()
    out["text_minimal"] = out.apply(choose, axis=1).map(minimal_text)
    out["text_semantic"] = out["text_minimal"].map(lambda x: semantic_text(x, nlp))
    out["has_full_text"] = out.get(text_col, pd.Series("", index=out.index)).fillna("").astype(str).str.strip().ne("")
    return out
