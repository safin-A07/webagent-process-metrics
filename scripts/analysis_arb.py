"""AgentRewardBench analyses: RQ2 (per-trajectory cost regression), RQ3 (agreement with expert looping labels and
LLM judges, incremental validity), RQ5 (abstraction robustness, distinctness), U1 (early warning), loop-erased L*.
Pre-registered in notes/preregistration.md.

Outputs: results/tables/rq2_arb_*.csv, rq3_*.csv, rq5_arb_*.csv, u1_arb.csv, lstar_arb.csv; figures rq3_roc.pdf,
         rq2_cost_vs_arr.pdf
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import loaders as L  # noqa: E402
import statkit as S  # noqa: E402

RES = ROOT / "results"
SCORES = ["ARR_url", "ARR_ax", "SBF_url", "ARR_plus_SBF", "ARR_url_noscroll", "ARR_psi", "T", "B_longest_repeat",
          "B_trigram_loop"]
N_SPLITS = 100


def load():
    d = pd.read_csv(RES / "per_trajectory_arb.csv", low_memory=False)
    d = d[d.excluded.fillna("") == ""].copy()
    d["ARR_plus_SBF"] = d.ARR_url + d.SBF_url
    d["logT"] = np.log(d["T"])
    d["task_key"] = d.benchmark + "/" + d.task_id
    return d


# ------------------------------------------------------------------------------------------------ bootstrap helpers
def task_boot(d, fn, offset=0, b=S.B):
    tasks = d.task_key.unique()
    groups = {t: g.index.values for t, g in d.groupby("task_key")}
    r = S.rng(offset)
    out = []
    for _ in range(b):
        pick = r.choice(tasks, len(tasks), replace=True)
        idx = np.concatenate([groups[t] for t in pick])
        try:
            out.append(fn(d.loc[idx]))
        except ValueError:
            out.append(np.nan)
    return S.pct_ci(out)


# ------------------------------------------------------------------------------------------------ RQ3
def rq3(d):
    v = d[d.loop.notna()].copy()
    v["loop"] = v.loop.astype(int)
    rows = []
    for s in SCORES:
        auc = roc_auc_score(v.loop, v[s])
        ap = average_precision_score(v.loop, v[s])
        lo, hi = task_boot(v, lambda x: roc_auc_score(x.loop, x[s]), offset=11)
        alo, ahi = task_boot(v, lambda x: average_precision_score(x.loop, x[s]), offset=12)
        r_pb = float(np.corrcoef(v.loop, v[s])[0, 1])
        rows.append({"score": s, "n": len(v), "prevalence": v.loop.mean(), "auroc": auc, "auroc_lo": lo,
                     "auroc_hi": hi, "auprc": ap, "auprc_lo": alo, "auprc_hi": ahi, "point_biserial": r_pb})
    main = pd.DataFrame(rows)
    # held-out thresholds (grouped by task, stratified by benchmark) and incremental validity over log T
    r = S.rng(13)
    bench_tasks = {b: g.task_key.unique() for b, g in v.groupby("benchmark")}
    split_rows, inc = [], []
    for k in range(N_SPLITS):
        train = set()
        for b, ts in bench_tasks.items():
            train |= set(r.choice(ts, len(ts) // 2, replace=False))
        tr, te = v[v.task_key.isin(train)], v[~v.task_key.isin(train)]
        for s in ["ARR_url", "ARR_ax", "SBF_url", "ARR_plus_SBF", "ARR_psi", "B_trigram_loop", "T"]:
            ths = np.unique(tr[s])
            f1s = [f1_score(tr.loop, tr[s] >= t, zero_division=0) for t in ths]
            t = ths[int(np.argmax(f1s))]
            p = te[s] >= t
            split_rows.append({"split": k, "score": s, "threshold": t, "f1": f1_score(te.loop, p, zero_division=0),
                               "precision": precision_score(te.loop, p, zero_division=0),
                               "recall": recall_score(te.loop, p, zero_division=0)})
        base = LogisticRegression(max_iter=1000).fit(tr[["logT"]], tr.loop)
        a0 = roc_auc_score(te.loop, base.predict_proba(te[["logT"]])[:, 1])
        for feats in (["logT", "ARR_url"], ["logT", "ARR_url", "SBF_url"], ["logT", "ARR_psi"],
                      ["logT", "B_trigram_loop"], ["logT", "ARR_ax"]):
            m = LogisticRegression(max_iter=1000).fit(tr[feats], tr.loop)
            a1 = roc_auc_score(te.loop, m.predict_proba(te[feats])[:, 1])
            inc.append({"split": k, "model": "+".join(feats[1:]), "auroc_logT": a0, "auroc_model": a1,
                        "delta": a1 - a0})
    sp = pd.DataFrame(split_rows)
    held = sp.groupby("score").agg(f1=("f1", "mean"), f1_sd=("f1", "std"), precision=("precision", "mean"),
                                   recall=("recall", "mean"), thr_median=("threshold", "median")).reset_index()
    inc = pd.DataFrame(inc)
    incs = inc.groupby("model").agg(auroc_logT=("auroc_logT", "mean"), auroc_model=("auroc_model", "mean"),
                                    delta_mean=("delta", "mean"), delta_lo=("delta", lambda x: np.quantile(x, .025)),
                                    delta_hi=("delta", lambda x: np.quantile(x, .975)),
                                    share_positive=("delta", lambda x: (x > 0).mean())).reset_index()
    a = main.set_index("score")
    h3 = bool(a.loc["ARR_url", "auroc_lo"] >= 0.70 and incs.set_index("model").loc["ARR_url", "delta_lo"] > 0)
    # LLM judges on the same trajectories, evaluated on all labelled trajectories and on the same held-out halves
    judges = L.arb_judges()
    jrows = []
    keyed = v.assign(jk=list(zip(v.benchmark, v.config, v.task_id)))
    for j in sorted({jj for x in judges.values() for jj in x}):
        pred = keyed.jk.map(lambda k: judges.get(k, {}).get(j))
        ok = pred.notna()
        if ok.sum() == 0:
            continue
        y, p = keyed.loop[ok], pred[ok].astype(int)
        jrows.append({"detector": f"LLM judge: {j}", "n": int(ok.sum()), "f1": f1_score(y, p),
                      "precision": precision_score(y, p), "recall": recall_score(y, p),
                      "accuracy": float((y == p).mean())})
    # ARR detector at the median held-out threshold, evaluated on the same trajectories as the judges
    thr = held.set_index("score").loc["ARR_url", "thr_median"]
    p = (v.ARR_url >= thr).astype(int)
    jrows.append({"detector": f"ARR_url >= {thr:.3f} (median held-out threshold)", "n": len(v),
                  "f1": f1_score(v.loop, p), "precision": precision_score(v.loop, p),
                  "recall": recall_score(v.loop, p), "accuracy": float((v.loop == p).mean())})
    return main, held, incs, pd.DataFrame(jrows), h3, v


def roc_fig(v):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for s, lab, c in (("ARR_url", "ARR ($\\phi_{URL}$)", "#2a6fbb"), ("ARR_ax", "ARR ($\\phi_{URL+AX}$)", "#5fa8d3"),
                      ("ARR_psi", "unique-action ratio", "#999999"), ("T", "trajectory length $T$", "#d1495b"),
                      ("SBF_url", "SBF", "#edae49")):
        f, t, _ = roc_curve(v.loop, v[s])
        ax.plot(f, t, label=f"{lab} ({roc_auc_score(v.loop, v[s]):.2f})", color=c, lw=1.2)
    ax.plot([0, 1], [0, 1], color="grey", lw=.6, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "figures" / "rq3_roc.pdf")


# ------------------------------------------------------------------------------------------------ RQ2
def within(y, X, groups):
    """Two-way fixed effects by alternating demeaning (task and agent)."""
    y, X = y.copy(), X.copy()
    for _ in range(50):
        for g in groups:
            y = y - y.groupby(g).transform("mean")
            X = X - X.groupby(g).transform("mean")
    return y, X


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    resid = y.values - X.values @ beta
    r2 = 1 - (resid ** 2).sum() / ((y.values - y.values.mean()) ** 2).sum()
    return beta, r2


def rq2(d):
    v = d[d.success.notna() & (d.tokens_total > 0)].copy()
    v["S"] = v.success.astype(float)
    v["logC"] = np.log(v.tokens_total)
    groups = [v.task_key, v.config]
    base_cols, full_cols = ["S", "logT"], ["S", "logT", "ARR_url", "SBF_url"]

    def fit(x):
        g = [x.task_key, x.config]
        y, X = within(x.logC, x[full_cols], g)
        b_full, r2_full = ols(y, X)
        _, r2_base = ols(y, X[base_cols])
        return b_full, r2_full - r2_base, r2_full, r2_base

    b, dr2, r2f, r2b = fit(v)
    # task-cluster bootstrap (B = 1000)
    tasks = v.task_key.unique()
    grp = {t: g.index.values for t, g in v.groupby("task_key")}
    r = S.rng(21)
    bs = []
    for _ in range(S.B):
        pick = r.choice(tasks, len(tasks), replace=True)
        idx = np.concatenate([grp[t] for t in pick])
        x = v.loc[idx].copy()
        # duplicated tasks become separate fixed-effect groups
        x["task_key"] = np.repeat([f"{t}#{i}" for i, t in enumerate(pick)], [len(grp[t]) for t in pick])
        x = x.reset_index(drop=True)
        bb, ddr2, *_ = fit(x)
        bs.append(np.r_[bb, ddr2])
    bs = np.array(bs)
    rows = []
    for i, c in enumerate(full_cols):
        lo, hi = S.pct_ci(bs[:, i])
        rows.append({"term": c, "coef": b[i], "ci_lo": lo, "ci_hi": hi})
    lo, hi = S.pct_ci(bs[:, -1])
    rows.append({"term": "delta_within_R2 (ARR,SBF beyond S, logT)", "coef": dr2, "ci_lo": lo, "ci_hi": hi})
    rows.append({"term": "within_R2_full", "coef": r2f})
    rows.append({"term": "within_R2_base", "coef": r2b})
    rows.append({"term": "n_trajectories", "coef": len(v)})
    reg = pd.DataFrame(rows)
    h2 = bool(any(reg.loc[reg.term == t, "ci_lo"].iloc[0] > 0 or reg.loc[reg.term == t, "ci_hi"].iloc[0] < 0
                  for t in ("ARR_url", "SBF_url")) and dr2 >= 0.01)
    # Spearman correlations (raw)
    cor = [{"metric": m, "outcome": "log tokens", "spearman": S.spearman(v[m], v.logC)}
           for m in ("ARR_url", "SBF_url", "T", "ARR_psi", "omega_tokens")]
    # omega distribution per agent
    om = v.groupby("config").agg(n=("omega_tokens", "size"), omega_mean=("omega_tokens", "mean"),
                                 omega_median=("omega_tokens", "median"),
                                 share_tokens_flagged=("omega_tokens", lambda x: np.nan),
                                 tokens_total=("tokens_total", "sum")).reset_index()
    for i, c in enumerate(om.config):
        g = v[v.config == c]
        om.loc[i, "share_tokens_flagged"] = (g.omega_tokens * g.tokens_total).sum() / g.tokens_total.sum()
    allrow = {"config": "all", "n": len(v), "omega_mean": v.omega_tokens.mean(), "omega_median": v.omega_tokens.median(),
              "share_tokens_flagged": (v.omega_tokens * v.tokens_total).sum() / v.tokens_total.sum(),
              "tokens_total": v.tokens_total.sum()}
    om = pd.concat([om, pd.DataFrame([allrow])])
    return reg, pd.DataFrame(cor), om, h2, v


def cost_fig(v):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agent = {"Qwen": "Qwen2.5-VL-72B", "claude": "Claude 3.7 Sonnet", "gpt-4o": "GPT-4o", "Llama": "Llama-3.3-70B"}
    plt.rcParams.update({"pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    for c, g in v.groupby("config"):
        ax.scatter(g.ARR_url, g.tokens_total / 1000, s=5, alpha=.4,
                   label=next((n for k, n in agent.items() if k in c), c))
    ax.set_yscale("log")
    ax.set_xlabel("ARR ($\\phi_{\\mathrm{URL}}$)")
    ax.set_ylabel("Tokens per trajectory (thousands)")
    ax.legend(fontsize=6, frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(RES / "figures" / "rq2_cost_vs_arr.pdf")


# ------------------------------------------------------------------------------------------------ RQ5
def rq5(d):
    rows = []
    for a, b in (("ARR_url", "ARR_ax"), ("SBF_url", "SBF_ax"), ("ARR_url", "ARR_url_noscroll"),
                 ("ARR_url", "ARR_psi")):
        rho = S.spearman(d[a], d[b])
        lo, hi = task_boot(d, lambda x: S.spearman(x[a], x[b]), offset=31, b=300)
        rows.append({"a": a, "b": b, "spearman": rho, "lo": lo, "hi": hi, "robust_rho_ge_0.8": rho >= 0.8,
                     "mean_a": d[a].mean(), "mean_b": d[b].mean()})
    rob = pd.DataFrame(rows)
    cols = ["ARR_url", "SBF_url", "T", "success", "ARR_psi", "B_trigram_loop"]
    v = d[d.success.notna()]
    cr = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            cr.append({"a": a, "b": b, "spearman": S.spearman(v[a], v[b]),
                       "partial_given_logT": S.partial_spearman(v[a], v[b], v.logT) if "T" not in (a, b) else np.nan})
    agents = d.groupby("config").agg(n=("T", "size"), success=("success", "mean"), T=("T", "median"),
                                     ARR_url=("ARR_url", "mean"), ARR_ax=("ARR_ax", "mean"), SBF_url=("SBF_url", "mean"),
                                     SBF_ax=("SBF_ax", "mean"), loop_rate=("loop", "mean"),
                                     truncated=("truncated", "mean")).reset_index()
    return rob, pd.DataFrame(cr), agents


# ------------------------------------------------------------------------------------------------ U1 and L*
def u1(d, k=10):
    v = d[(d["T"] > k) & d.success.notna()].copy()
    y = 1 - v.success.astype(int)
    rows = []
    for f in ("pre_ARR_url", "pre_SBF_url", "pre_ARR_psi"):
        lo, hi = task_boot(v.assign(y=y), lambda x: roc_auc_score(x.y, x[f]), offset=41, b=500)
        rows.append({"dataset": "AgentRewardBench", "feature": f, "n": len(v), "failure_rate": y.mean(),
                     "auroc": roc_auc_score(y, v[f]), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def lstar(d):
    """Loop-erased path length vs raw length on successful runs (exact states, phi_URL)."""
    v = d[d.success == 1]
    rows = []
    for col, lab in (("loop_erased_len", "phi_URL"), ("loop_erased_len_ax", "phi_URL+AX")):
        # leave-one-agent-out minimum per task, raw T vs loop-erased length
        g = v.groupby("task_key")
        lo_raw, lo_le = [], []
        for _, x in g:
            for i in range(len(x)):
                other = x.config.values != x.config.values[i]
                if other.any():
                    lo_raw.append(x["T"].values[other].min())
                    lo_le.append(x[col].values[other].min())
        rows.append({"abstraction": lab, "n_success": len(v), "median_T": v["T"].median(),
                     "median_loop_erased": v[col].median(), "mean_ratio_LE_over_T": (v[col] / v["T"]).mean(),
                     "share_shorter": (v[col] < v["T"]).mean(), "n_loo": len(lo_raw),
                     "median_LOO_min_T": float(np.median(lo_raw)) if lo_raw else np.nan,
                     "median_LOO_min_LE": float(np.median(lo_le)) if lo_le else np.nan})
    return pd.DataFrame(rows)


def main():
    d = load()
    print("ARB trajectories:", len(d), "labelled loop:", d.loop.notna().sum(), flush=True)
    main_, held, incs, judges, h3, v = rq3(d)
    main_.to_csv(RES / "tables" / "rq3_auroc.csv", index=False)
    held.to_csv(RES / "tables" / "rq3_heldout.csv", index=False)
    incs.to_csv(RES / "tables" / "rq3_incremental.csv", index=False)
    judges.to_csv(RES / "tables" / "rq3_judges.csv", index=False)
    roc_fig(v)
    reg, cor, om, h2, v2 = rq2(d)
    reg.to_csv(RES / "tables" / "rq2_arb_regression.csv", index=False)
    cor.to_csv(RES / "tables" / "rq2_arb_spearman.csv", index=False)
    om.to_csv(RES / "tables" / "rq2_arb_omega.csv", index=False)
    cost_fig(v2)
    rob, cr, agents = rq5(d)
    rob.to_csv(RES / "tables" / "rq5_arb_robustness.csv", index=False)
    cr.to_csv(RES / "tables" / "rq5_arb_correlations.csv", index=False)
    agents.to_csv(RES / "tables" / "arb_agents.csv", index=False)
    u = u1(d)
    u.to_csv(RES / "tables" / "u1_arb.csv", index=False)
    ls = lstar(d)
    ls.to_csv(RES / "tables" / "lstar_arb.csv", index=False)
    pd.set_option("display.width", 220)
    for name, t in (("RQ3 AUROC", main_), ("RQ3 held-out", held), ("RQ3 incremental", incs), ("RQ3 judges", judges),
                    ("RQ2 regression", reg), ("RQ2 spearman", cor), ("RQ2 omega", om), ("RQ5 robustness", rob),
                    ("RQ5 correlations", cr), ("agents", agents), ("U1", u), ("L* loop-erased", ls)):
        print("\n==", name)
        print(t.round(3).to_string())
    print("\nH2 supported:", h2, " H3 supported:", h3)


def only_lstar():
    d = load()
    ls = lstar(d)
    ls.to_csv(RES / "tables" / "lstar_arb.csv", index=False)
    print(ls.round(3).to_string())


def only_rq3():
    """Re-run RQ3 alone (used after the LLM-judge download completed)."""
    main_, held, incs, judges, h3, v = rq3(load())
    main_.to_csv(RES / "tables" / "rq3_auroc.csv", index=False)
    held.to_csv(RES / "tables" / "rq3_heldout.csv", index=False)
    incs.to_csv(RES / "tables" / "rq3_incremental.csv", index=False)
    judges.to_csv(RES / "tables" / "rq3_judges.csv", index=False)
    roc_fig(v)
    print(judges.round(3).to_string())
    print("H3 supported:", h3)


if __name__ == "__main__":
    {"rq3": only_rq3, "lstar": only_lstar}.get(sys.argv[1] if sys.argv[1:] else "", main)()
