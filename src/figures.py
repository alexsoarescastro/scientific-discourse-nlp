from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def fig_annual_publications(df, out):
    x = df.groupby("publication_year").size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x["publication_year"], x["n"])
    ax.set(xlabel="Publication year", ylabel="Number of publications")
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)

def fig_umap(zdf, labels, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    cats = pd.Series(labels).astype("category")
    for cat in cats.cat.categories:
        mask = (cats == cat).values
        ax.scatter(
            zdf.loc[mask, "UMAP-1"],
            zdf.loc[mask, "UMAP-2"],
            s=8, alpha=.55, label=cat
        )
    ax.set(xlabel="UMAP-1", ylabel="UMAP-2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)
