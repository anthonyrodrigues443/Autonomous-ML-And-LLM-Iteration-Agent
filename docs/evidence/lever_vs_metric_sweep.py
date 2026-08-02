"""Does each canonical lever move a THRESHOLD metric and a RANKING metric equally?
Deterministic, no LLM. If the gap is structural it shows on every dataset."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from iterate.adapters.data.tabular import load_csv
from iterate.core.scoring import score

def from_csv(path, tgt):
    ds = load_csv(path, target=tgt)
    Xt = pd.get_dummies(ds.train_features)
    Xh = pd.get_dummies(ds.test_features).reindex(columns=Xt.columns, fill_value=0)
    return Xt, ds.train_target, Xh, ds.test_target

def from_sklearn(loader):
    d = loader(); X, y = pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target)
    return (*train_test_split(X, y, test_size=0.2, random_state=42, stratify=y),)[0:1]+(None,)

def bc():
    d = load_breast_cancer()
    X, y = pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target)
    Xt,Xh,yt,yh = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return Xt,yt,Xh,yh

def synth(n=4000, imb=0.08, seed=0):
    rng=np.random.RandomState(seed)
    X=pd.DataFrame(rng.randn(n,12), columns=[f"f{i}" for i in range(12)])
    logit = X["f0"]*1.4 + X["f1"]*0.9 - X["f2"]*0.7 + rng.randn(n)*0.8
    thr=np.quantile(logit, 1-imb); y=pd.Series((logit>thr).astype(int))
    Xt,Xh,yt,yh=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    return Xt,yt,Xh,yh

DATASETS = {
    "churn (27% pos)":        lambda: from_csv("examples/churn_tabular/data.clean.csv","Churn"),
    "adult (26% pos)":        lambda: from_csv("examples/adult_income.csv","income"),
    "breast_cancer (63%)":    bc,
    "synth sev-imb (8%)":     lambda: synth(imb=0.08),
    "synth mild-imb (35%)":   lambda: synth(imb=0.35, seed=3),
}

def metrics(m, Xt, yt, Xh, yh, thresh=None):
    p = m.predict_proba(Xh)[:,1]
    pred = (p >= thresh).astype(int) if thresh is not None else m.predict(Xh)
    return score("classification", yh, pred, y_proba=p, include=("average_precision",))

def best_threshold(m, Xt, yt):
    p = m.predict_proba(Xt)[:,1]
    best, bt = -1, 0.5
    for t in np.linspace(0.05,0.95,37):
        f = score("classification", yt, (p>=t).astype(int))["f1"]
        if f > best: best, bt = f, t
    return bt

LEVERS = ["imbalance: class_weight=balanced", "threshold: tuned on train",
          "model-swap: RandomForest", "model-swap: LogisticRegression",
          "hyperparam: deeper/more iters"]

print(f"{'dataset':22} {'lever':32} {'Δf1':>8} {'Δavg_prec':>10} {'Δroc_auc':>9}")
print("="*86)
agg = {l: {"f1":[], "ap":[], "auc":[]} for l in LEVERS}
for name, get in DATASETS.items():
    Xt,yt,Xh,yh = get()
    base_m = HistGradientBoostingClassifier(random_state=42).fit(Xt,yt)
    b = metrics(base_m, Xt,yt,Xh,yh)
    variants = [
        (LEVERS[0], HistGradientBoostingClassifier(random_state=42, class_weight="balanced").fit(Xt,yt), None),
        (LEVERS[1], base_m, best_threshold(base_m, Xt, yt)),
        (LEVERS[2], RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xt,yt), None),
        (LEVERS[3], LogisticRegression(max_iter=2000).fit(Xt.fillna(Xt.median()), yt), None),
        (LEVERS[4], HistGradientBoostingClassifier(random_state=42, max_iter=400, max_depth=8, learning_rate=0.05).fit(Xt,yt), None),
    ]
    for label, m, th in variants:
        Xh2 = Xh.fillna(Xt.median()) if "Logistic" in label else Xh
        v = metrics(m, Xt,yt,Xh2,yh, th)
        d1, dap, dauc = v["f1"]-b["f1"], v["average_precision"]-b["average_precision"], v["roc_auc"]-b["roc_auc"]
        agg[label]["f1"].append(d1); agg[label]["ap"].append(dap); agg[label]["auc"].append(dauc)
        print(f"{name:22} {label:32} {d1:>+8.4f} {dap:>+10.4f} {dauc:>+9.4f}")
    print("-"*86)

print("\nMEAN ABSOLUTE MOVEMENT ACROSS ALL 5 DATASETS")
print(f"{'lever':34} {'|Δf1|':>8} {'|Δavg_prec|':>12} {'|Δroc_auc|':>11}  verdict")
for l in LEVERS:
    f1=np.mean(np.abs(agg[l]["f1"])); ap=np.mean(np.abs(agg[l]["ap"])); auc=np.mean(np.abs(agg[l]["auc"]))
    rank_mean=(ap+auc)/2
    verdict = "THRESHOLD-ONLY" if f1 > 3*max(rank_mean,1e-9) else "moves both"
    print(f"{l:34} {f1:>8.4f} {ap:>12.4f} {auc:>11.4f}  {verdict}")
