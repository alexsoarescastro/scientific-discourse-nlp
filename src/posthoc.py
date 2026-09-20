import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def epsilon_squared_kruskal(H, n, k):
    return (H-k+1)/(n-k) if n>k else np.nan

def eta_squared_anova(groups):
    vals=np.concatenate(groups)
    grand=np.mean(vals)
    ssb=sum(len(g)*(np.mean(g)-grand)**2 for g in groups)
    sst=sum((vals-grand)**2)
    return ssb/sst if sst else np.nan

def pairwise_mannwhitney(df,value,group="thematic_domain"):
    d=df[[value,group]].dropna()
    levels=sorted(d[group].unique())
    rows=[]
    for i,a in enumerate(levels):
        for b in levels[i+1:]:
            x=d.loc[d[group]==a,value]; y=d.loc[d[group]==b,value]
            u,p=stats.mannwhitneyu(x,y,alternative="two-sided")
            # rank-biserial correlation
            rbc=1-(2*u)/(len(x)*len(y))
            rows.append({"group1":a,"group2":b,"U":u,"p":p,"rank_biserial":rbc,
                         "n1":len(x),"n2":len(y)})
    out=pd.DataFrame(rows)
    if len(out):
        out["p_fdr"]=multipletests(out.p,method="fdr_bh")[1]
    return out
