import argparse, numpy as np, pandas as pd
from pathlib import Path
from src.utils import load_config, ensure_dirs, set_seed
from src.preprocess import preprocess_dataframe
from src.embeddings import SciBERTEmbedder
from src.topics import fit_topics, cv_coherence, topic_table, assign_topic_domains
from src.semantic_map import make_2d_umap
from src.linguistics import add_linguistic_features
from src.experimental import add_experimental_features
from src.ml import cross_validate_rf
from src.shap_analysis import global_shap_importance, aggregate_onehot_importance
from src.figures import fig_temporal, fig_umap, fig_coherence, fig_shap, fig_confusion

def main(config):
    cfg=load_config(config); ensure_dirs(cfg); set_seed(cfg["seed"])
    raw=Path(cfg["paths"]["raw_csv"])
    if not raw.exists():
        raise FileNotFoundError(
            f"{raw} not found. Add the real BES corpus before running empirical analyses."
        )
    df=pd.read_csv(raw)
    df=preprocess_dataframe(df)

    emb_path=Path(cfg["paths"]["embeddings_npy"])
    if emb_path.exists():
        E=np.load(emb_path)
    else:
        e=SciBERTEmbedder(cfg["scibert"]["model_name"],cfg["scibert"]["max_length"])
        E=e.encode(df.text_semantic.fillna("").tolist())
        np.save(emb_path,E)

    model,topics,probs=fit_topics(df.text_semantic.fillna("").tolist(),E,cfg)
    df["topic"]=topics
    model.save(cfg["paths"]["topic_model"], serialization="pickle")

    domains=assign_topic_domains(model)
    df["thematic_domain"]=df["topic"].map(domains)

    coh=cv_coherence(model,[x.split() for x in df.text_semantic.fillna("")],
                     cfg["bertopic"]["top_n_words"])
    coh.to_csv(Path(cfg["paths"]["output_tables"])/"topic_coherence_cv.csv",index=False)
    topic_table(model).to_csv(Path(cfg["paths"]["output_tables"])/"topics.csv",index=False)

    z=make_2d_umap(E,cfg)
    df=pd.concat([df.reset_index(drop=True),z],axis=1)
    df=add_linguistic_features(df,cfg)
    df=add_experimental_features(df)
    df.to_csv(cfg["paths"]["processed_csv"],index=False)

    figs=Path(cfg["paths"]["output_figures"])
    tabs=Path(cfg["paths"]["output_tables"])
    if "technology_domain" in df:
        fig_temporal(df,figs/"fig2_temporal.png")
    fig_umap(z,df["thematic_domain"].fillna("Unassigned"),figs/"fig4a_umap.png")
    fig_coherence(coh,figs/"fig4b_coherence.png")

    pipe,metrics,cm,report,idx,num,cat=cross_validate_rf(df,cfg)
    pd.DataFrame([metrics]).to_csv(tabs/"rf_cv_metrics.csv",index=False)
    cm.to_csv(tabs/"rf_confusion_matrix.csv")
    report.to_csv(tabs/"rf_classification_report.csv")
    fig_confusion(cm,figs/"rf_confusion_matrix.png")

    d=df.loc[idx]
    X=d[num+cat]
    imp=global_shap_importance(pipe,X)
    agg=aggregate_onehot_importance(imp)
    imp.to_csv(tabs/"shap_transformed_features.csv",index=False)
    agg.to_csv(tabs/"shap_normalized_importance.csv",index=False)
    fig_shap(agg,figs/"fig5_shap.png")
    print(metrics)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="config/config.yaml")
    args=p.parse_args()
    main(args.config)
