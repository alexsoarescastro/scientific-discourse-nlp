import pandas as pd

PERIODS = [
    (2000,2008,"2000–2008"),
    (2009,2016,"2009–2016"),
    (2017,2025,"2017–2025"),
]

def add_period(df):
    out=df.copy()
    def p(y):
        for lo,hi,label in PERIODS:
            if lo <= y <= hi: return label
        return None
    out["period"]=out["publication_year"].map(p)
    return out

def thematic_temporal_table(df):
    d=add_period(df)
    tab=d.groupby(["period","thematic_domain"],dropna=False).size().rename("n").reset_index()
    tab["share_within_period"]=tab.groupby("period")["n"].transform(lambda x:x/x.sum())
    return tab
