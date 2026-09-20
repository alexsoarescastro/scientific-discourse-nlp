import ast, pandas as pd, numpy as np
from collections import Counter
import networkx as nx

def _listify(x):
    if isinstance(x, list): return x
    if pd.isna(x): return []
    if isinstance(x, str):
        try:
            y=ast.literal_eval(x)
            return y if isinstance(y,list) else [x]
        except Exception:
            return [z.strip() for z in x.split(";") if z.strip()]
    return []

def annual_publications(df):
    return df.groupby("publication_year").size().rename("n_publications").reset_index()

def domain_by_year(df, domain_col="technology_domain"):
    return df.groupby(["publication_year",domain_col]).size().rename("n").reset_index()

def citation_rate(df, current_year=2026):
    out=df.copy()
    age=(current_year-out["publication_year"]).clip(lower=1)
    out["citation_rate_per_year"]=out["cited_by_count"].fillna(0)/age
    return out

def country_counts(df, countries_col="countries"):
    c=Counter()
    for xs in df[countries_col].map(_listify):
        c.update(set(xs))
    return pd.DataFrame(c.items(),columns=["country","publications"]).sort_values("publications",ascending=False)

def coauthorship_country_network(df, countries_col="countries"):
    G=nx.Graph()
    for xs in df[countries_col].map(_listify):
        xs=sorted(set(xs))
        for x in xs: G.add_node(x)
        for i in range(len(xs)):
            for j in range(i+1,len(xs)):
                a,b=xs[i],xs[j]
                G.add_edge(a,b,weight=G.get_edge_data(a,b,{}).get("weight",0)+1)
    return G

def edge_table(G):
    return pd.DataFrame(
        [{"source":u,"target":v,**d} for u,v,d in G.edges(data=True)]
    )
