"""HAL parts of RQ2 (configuration level), RQ5 (distinctness) and U1 (early warning). Pre-registered.

Outputs: results/tables/rq2_hal_config.csv, rq5_hal_correlations.csv, u1_hal.csv, results/figures/rq5_hal_heatmap.pdf
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import statkit as S  # noqa: E402

RES = ROOT / "results"


def main_model_tokens(usage_json):
    u = json.loads(usage_json)
    tin = sum(v.get("prompt_tokens", 0) for k, v in u.items() if "embedding" not in k)
    tout = sum(v.get("completion_tokens", 0) for k, v in u.items() if "embedding" not in k)
    return tin, tout


def rq2_config(d, cfg):
    rows = []
    v = d[d.excluded.fillna("") == ""]
    agg = v.groupby("config").agg(scaffold=("scaffold", "first"), mean_T=("T", "mean"), ARR=("ARR_psi", "mean"),
                                  SBF=("SBF_url", "mean"), TSR=("success", "mean"),
                                  calls_per_task=("n_llm_calls", "mean"))
    tok = cfg.set_index("config").usage_json.apply(main_model_tokens)
    agg["tokens_in_per_task"] = [tok[c][0] / 300 for c in agg.index]
    agg["tokens_out_per_task"] = [tok[c][1] / 300 for c in agg.index]
    agg["tokens_per_task"] = agg.tokens_in_per_task + agg.tokens_out_per_task
    agg["reported_cost"] = cfg.set_index("config").total_cost_reported.reindex(agg.index)
    agg.to_csv(RES / "tables" / "rq2_hal_config_values.csv")
    for sc, g in agg.groupby("scaffold"):
        for m in ("mean_T", "ARR", "SBF", "TSR", "calls_per_task"):
            if g[m].isna().all():
                continue
            for y in ("tokens_per_task", "reported_cost"):
                rows.append({"scaffold": sc, "metric": m, "outcome": y, "K": len(g), "spearman": S.spearman(g[m], g[y])})
    return pd.DataFrame(rows)


def rq5_distinct(d):
    v = d[(d.excluded.fillna("") == "") & (d.scaffold == "browser-use")].copy()
    v["logT"] = np.log(v["T"])
    cols = ["SER_H", "ARR_psi", "SBF_url", "T", "success", "B_unique", "B_longest_repeat", "B_trigram_loop"]
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rows.append({"level": "trajectory", "a": a, "b": b, "spearman": S.spearman(v[a], v[b]),
                         "partial_given_logT": S.partial_spearman(v[a], v[b], v.logT) if "T" not in (a, b) else np.nan})
    c = v.groupby("config")[cols].mean()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rows.append({"level": "configuration (K=%d)" % len(c), "a": a, "b": b, "spearman": S.spearman(c[a], c[b]),
                         "partial_given_logT": np.nan})
    out = pd.DataFrame(rows)
    heat(v[["SER_H", "ARR_psi", "SBF_url", "T", "success"]].corr(method="spearman"))
    return out


def heat(c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    lab = ["SER", "ARR", "SBF", "T", "S"]
    ax.set_xticks(range(5), lab)
    ax.set_yticks(range(5), lab)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, fraction=.046)
    fig.tight_layout()
    fig.savefig(RES / "figures" / "rq5_hal_heatmap.pdf")


def u1(d, k=10, n_splits=100):
    v = d[(d.excluded.fillna("") == "") & (d.scaffold == "browser-use") & (d["T"] > k)].copy()
    y = 1 - v.success.values
    rows = []
    sites = v.website.values
    us = np.unique(sites)
    for feat in ("pre_ARR_psi", "pre_SBF_url"):
        auc = roc_auc_score(y, v[feat])
        W = S.cluster_weights(len(us), offset=7)
        idx = {s: i for i, s in enumerate(us)}
        si = np.array([idx[s] for s in sites])
        bt = []
        for i in range(S.B):
            w = W[i][si]
            m = w > 0
            if len(np.unique(y[m])) < 2:
                continue
            bt.append(roc_auc_score(y[m], v[feat].values[m], sample_weight=w[m]))
        lo, hi = S.pct_ci(bt)
        rows.append({"dataset": "HAL Browser-Use", "feature": feat, "n": len(v), "failure_rate": y.mean(),
                     "auroc": auc, "lo": lo, "hi": hi})
    # early-termination rule on pre_ARR_psi: threshold maximising Youden J on half of the tasks
    tasks = v.task_id.unique()
    r = S.rng(8)
    saved, lost, term = [], [], []
    for s in range(n_splits):
        tr = set(r.choice(tasks, len(tasks) // 2, replace=False))
        a, b = v[v.task_id.isin(tr)], v[~v.task_id.isin(tr)]
        best, thr = -1, None
        for t in np.unique(a.pre_ARR_psi):
            pred = a.pre_ARR_psi >= t
            tpr = (pred & (a.success == 0)).sum() / max(1, (a.success == 0).sum())
            fpr = (pred & (a.success == 1)).sum() / max(1, (a.success == 1).sum())
            if tpr - fpr > best:
                best, thr = tpr - fpr, t
        stop = b.pre_ARR_psi >= thr
        saved.append(((b["T"] - k) * stop).sum() / b["T"].sum())
        lost.append((stop & (b.success == 1)).sum() / max(1, (b.success == 1).sum()))
        term.append(stop.mean())
    rows.append({"dataset": "HAL Browser-Use", "feature": "early stop rule pre_ARR_psi", "n": len(v),
                 "share_steps_saved": np.mean(saved), "sd_saved": np.std(saved),
                 "share_successes_lost": np.mean(lost), "sd_lost": np.std(lost), "share_terminated": np.mean(term)})
    return pd.DataFrame(rows)


def main():
    d = pd.read_csv(RES / "per_trajectory_hal.csv", low_memory=False)
    cfg = pd.read_csv(ROOT / "data" / "processed" / "hal_configs.csv")
    cfg["config"] = cfg.config.str.removesuffix("_UPLOAD")
    a = rq2_config(d, cfg)
    a.to_csv(RES / "tables" / "rq2_hal_config.csv", index=False)
    b = rq5_distinct(d)
    b.to_csv(RES / "tables" / "rq5_hal_correlations.csv", index=False)
    c = u1(d)
    c.to_csv(RES / "tables" / "u1_hal.csv", index=False)
    pd.set_option("display.width", 200)
    print(a.round(3).to_string())
    print(b.round(3).to_string())
    print(c.round(3).to_string())


if __name__ == "__main__":
    main()
