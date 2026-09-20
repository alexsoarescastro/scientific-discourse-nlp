import requests, pandas as pd, time
from tqdm import tqdm

OPENALEX = "https://api.openalex.org/works"

def reconstruct_abstract(inv):
    if not isinstance(inv, dict) or not inv:
        return ""
    positions = []
    for token, idxs in inv.items():
        for i in idxs:
            positions.append((i, token))
    return " ".join(t for _, t in sorted(positions))

def query_openalex(search_terms, year_min=2000, year_max=2025, mailto=None,
                   per_page=200, max_records=None):
    """Metadata collector. Review/lock the exact search strategy before publication."""
    rows, cursor = [], "*"
    filt = f"from_publication_date:{year_min}-01-01,to_publication_date:{year_max}-12-31"
    while cursor:
        params = {
            "search": search_terms,
            "filter": filt,
            "per-page": per_page,
            "cursor": cursor,
        }
        if mailto: params["mailto"] = mailto
        r = requests.get(OPENALEX, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        for w in payload["results"]:
            rows.append({
                "id": w.get("id"),
                "doi": w.get("doi"),
                "title": w.get("title") or "",
                "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
                "publication_year": w.get("publication_year"),
                "cited_by_count": w.get("cited_by_count", 0),
                "type": w.get("type"),
                "language": w.get("language"),
                "source": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            })
            if max_records and len(rows) >= max_records:
                return pd.DataFrame(rows)
        cursor = payload.get("meta", {}).get("next_cursor")
        if not payload["results"]: break
        time.sleep(0.1)
    return pd.DataFrame(rows)

def filter_corpus(df, year_min=2000, year_max=2025):
    out = df.copy()
    out = out[out.publication_year.between(year_min, year_max)]
    if "language" in out:
        out = out[(out.language.isna()) | (out.language.eq("en"))]
    return out.drop_duplicates(subset=["doi","title"], keep="first")
