import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def group_test(df, value, group="thematic_domain"):
    x = df[[value,group]].dropna()
    groups = [g[value].values for _,g in x.groupby(group)]
    if len(groups) < 2: return {"variable":value, "test":None}
    lev = stats.levene(*groups)
    normal = all((len(a) >= 3 and stats.shapiro(a[:5000]).pvalue > .05) for a in groups)
    if normal and lev.pvalue > .05:
        res = stats.f_oneway(*groups); test="ANOVA"
    elif normal:
        try:
            res = stats.f_oneway(*groups, equal_var=False); test="Welch ANOVA"
        except TypeError:
            res = stats.kruskal(*groups); test="Kruskal-Wallis"
    else:
        res = stats.kruskal(*groups); test="Kruskal-Wallis"
    return {"variable":value,"test":test,"statistic":res.statistic,"p":res.pvalue,"n":len(x)}

def spearman_with_ci(x, y, n_boot=2000, seed=42):
    d = pd.DataFrame({"x":x,"y":y}).dropna()
    rho,p = stats.spearmanr(d.x,d.y)
    rng=np.random.default_rng(seed); boots=[]
    n=len(d)
    if n >= 4:
        for _ in range(n_boot):
            idx=rng.integers(0,n,n)
            r,_=stats.spearmanr(d.x.iloc[idx],d.y.iloc[idx])
            if np.isfinite(r): boots.append(r)
    lo,hi=(np.percentile(boots,[2.5,97.5]) if boots else (np.nan,np.nan))
    return {"rho":rho,"p":p,"ci_low":lo,"ci_high":hi,"n":n}

def fdr_table(results, pcol="p"):
    out=pd.DataFrame(results)
    mask=out[pcol].notna()
    out.loc[mask,"p_fdr"]=multipletests(out.loc[mask,pcol], method="fdr_bh")[1]
    return out
