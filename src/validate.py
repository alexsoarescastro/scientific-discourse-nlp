from pathlib import Path
import pandas as pd, numpy as np

REQUIRED_BASE=["title","publication_year","cited_by_count"]
REQUIRED_TEXT=["full_text","abstract"]

def validate_input(df):
    problems=[]
    for c in REQUIRED_BASE:
        if c not in df: problems.append(f"Missing required column: {c}")
    if not any(c in df for c in REQUIRED_TEXT):
        problems.append("Need at least one text column: full_text or abstract")
    if "publication_year" in df:
        bad=df[~df.publication_year.between(2000,2025, inclusive="both")]
        if len(bad): problems.append(f"{len(bad)} rows outside 2000–2025.")
    return problems

def coverage_table(df, cols):
    rows=[]
    for c in cols:
        if c in df:
            n=df[c].notna().sum()
            rows.append({"variable":c,"available_n":int(n),"missing_n":int(len(df)-n),
                         "coverage_pct":100*n/len(df) if len(df) else np.nan})
    return pd.DataFrame(rows)
