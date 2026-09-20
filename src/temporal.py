import pandas as pd

def annual_topic_counts(df, topic_col="topic"):
    return (
        df.groupby(["publication_year", topic_col], dropna=False)
          .size().rename("n").reset_index()
    )

def annual_domain_counts(df, domain_col="thematic_domain"):
    return (
        df.groupby(["publication_year", domain_col], dropna=False)
          .size().rename("n").reset_index()
    )

def annual_domain_shares(df, domain_col="thematic_domain"):
    tab = annual_domain_counts(df, domain_col=domain_col)
    tab["share_within_year"] = (
        tab.groupby("publication_year")["n"].transform(lambda x: x / x.sum())
    )
    return tab
