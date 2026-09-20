import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

DEFAULT_NUMERIC = [
    "lexical_density","mattr","hedging_frequency","boosting_frequency",
    "cited_by_count","current_density","power_density","hydrogen_efficiency",
    "applied_potential","pH"
]
DEFAULT_CATEGORICAL = ["substrate_type"]

def build_pipeline(cfg, numeric, categorical):
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer([("num",num,numeric),("cat",cat,categorical)])
    rf = RandomForestClassifier(
        n_estimators=cfg["random_forest"]["n_estimators"],
        class_weight=cfg["random_forest"]["class_weight"],
        random_state=cfg["seed"], n_jobs=cfg["random_forest"]["n_jobs"]
    )
    return Pipeline([("preprocess",pre),("model",rf)])

def cross_validate_rf(df, cfg, target="thematic_domain",
                      numeric=None, categorical=None):
    numeric = [c for c in (numeric or DEFAULT_NUMERIC) if c in df.columns]
    categorical = [c for c in (categorical or DEFAULT_CATEGORICAL) if c in df.columns]
    d = df.dropna(subset=[target]).copy()
    X, y = d[numeric+categorical], d[target].astype(str)
    pipe = build_pipeline(cfg,numeric,categorical)
    cv = StratifiedKFold(
        n_splits=cfg["random_forest"]["cv_folds"], shuffle=True,
        random_state=cfg["seed"]
    )
    pred = cross_val_predict(pipe,X,y,cv=cv,n_jobs=1)
    metrics = {
        "accuracy": accuracy_score(y,pred),
        "balanced_accuracy": balanced_accuracy_score(y,pred),
        "macro_precision": precision_score(y,pred,average="macro",zero_division=0),
        "macro_recall": recall_score(y,pred,average="macro",zero_division=0),
        "macro_f1": f1_score(y,pred,average="macro",zero_division=0),
    }
    labels=sorted(y.unique())
    cm=pd.DataFrame(confusion_matrix(y,pred,labels=labels),index=labels,columns=labels)
    report=pd.DataFrame(classification_report(y,pred,output_dict=True,zero_division=0)).T
    pipe.fit(X,y)
    return pipe, metrics, cm, report, d.index, numeric, categorical
