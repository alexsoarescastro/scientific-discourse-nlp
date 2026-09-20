import numpy as np, pandas as pd, shap

def transformed_feature_names(pipe):
    pre = pipe.named_steps["preprocess"]
    return list(pre.get_feature_names_out())

def global_shap_importance(pipe, X):
    """TreeSHAP on the fitted RF after preprocessing.
    Multiclass SHAP arrays are collapsed across samples/classes by mean absolute value."""
    pre=pipe.named_steps["preprocess"]
    model=pipe.named_steps["model"]
    Xt=pre.transform(X)
    names=transformed_feature_names(pipe)
    explainer=shap.TreeExplainer(model)
    sv=explainer.shap_values(Xt)
    if isinstance(sv,list):
        arr=np.stack(sv,axis=-1)   # samples, features, classes
    else:
        arr=np.asarray(sv)
    if arr.ndim==3:
        mean_abs=np.mean(np.abs(arr),axis=(0,2))
    elif arr.ndim==2:
        mean_abs=np.mean(np.abs(arr),axis=0)
    else:
        raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
    norm=mean_abs/mean_abs.sum()
    return pd.DataFrame({
        "feature":names,
        "mean_abs_shap":mean_abs,
        "normalized_mean_abs_shap":norm
    }).sort_values("normalized_mean_abs_shap",ascending=False)

def aggregate_onehot_importance(imp):
    """Optional aggregation of one-hot substrate levels into their source variable."""
    out=imp.copy()
    def base(x):
        x=x.replace("num__","").replace("cat__","")
        if x.startswith("substrate_type_"): return "substrate_type"
        return x
    out["base_feature"]=out.feature.map(base)
    g=out.groupby("base_feature",as_index=False)[["mean_abs_shap","normalized_mean_abs_shap"]].sum()
    g["normalized_mean_abs_shap"]=g["mean_abs_shap"]/g["mean_abs_shap"].sum()
    return g.sort_values("normalized_mean_abs_shap",ascending=False)
