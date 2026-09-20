import pandas as pd
from umap import UMAP

def make_2d_umap(embeddings, cfg):
    c = cfg["visual_umap"]
    reducer = UMAP(
        n_neighbors=c["n_neighbors"], min_dist=c["min_dist"],
        metric=c["metric"], n_components=2, random_state=cfg["seed"]
    )
    z = reducer.fit_transform(embeddings)
    return pd.DataFrame({"UMAP-1":z[:,0], "UMAP-2":z[:,1]})
