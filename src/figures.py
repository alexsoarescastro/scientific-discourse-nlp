from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def fig_temporal(df, out):
    x=df.groupby(["publication_year","technology_domain"]).size().reset_index(name="n")
    fig,ax=plt.subplots(figsize=(9,5))
    for dom,g in x.groupby("technology_domain"):
        ax.plot(g.publication_year,g.n,label=dom)
    ax.set(xlabel="Publication year",ylabel="Number of publications")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out,dpi=600); plt.close(fig)

def fig_umap(zdf, labels, out):
    fig,ax=plt.subplots(figsize=(8,6))
    cats=pd.Series(labels).astype("category")
    for cat in cats.cat.categories:
        m=(cats==cat).values
        ax.scatter(zdf.loc[m,"UMAP-1"],zdf.loc[m,"UMAP-2"],s=8,alpha=.55,label=cat)
    ax.set(xlabel="UMAP-1",ylabel="UMAP-2")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out,dpi=600); plt.close(fig)

def fig_coherence(coh, out, top=10):
    d=coh.nlargest(top,"c_v").sort_values("c_v")
    fig,ax=plt.subplots(figsize=(8,5))
    ax.barh(d.topic.astype(str),d.c_v)
    ax.set(xlabel="Topic coherence (Cᵥ)",ylabel="Topic")
    fig.tight_layout(); fig.savefig(out,dpi=600); plt.close(fig)

def fig_shap(imp, out, top=10):
    d=imp.nlargest(top,"normalized_mean_abs_shap").sort_values("normalized_mean_abs_shap")
    fig,ax=plt.subplots(figsize=(9,5.5))
    ax.barh(d.base_feature,d.normalized_mean_abs_shap)
    ax.set(xlabel="Normalized mean absolute SHAP importance",ylabel="")
    for y,v in enumerate(d.normalized_mean_abs_shap):
        ax.text(v,y,f" {v:.2f}",va="center")
    fig.tight_layout(); fig.savefig(out,dpi=600); plt.close(fig)

def fig_confusion(cm, out):
    fig,ax=plt.subplots(figsize=(5.5,5))
    im=ax.imshow(cm.values,aspect="auto")
    ax.set_xticks(range(len(cm.columns)),cm.columns)
    ax.set_yticks(range(len(cm.index)),cm.index)
    ax.set(xlabel="Predicted",ylabel="Observed")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(cm.iloc[i,j]),ha="center",va="center")
    fig.colorbar(im,ax=ax)
    fig.tight_layout(); fig.savefig(out,dpi=600); plt.close(fig)
