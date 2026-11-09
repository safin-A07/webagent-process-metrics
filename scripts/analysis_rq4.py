"""RQ4 Optimal path without demonstrations (HAL). Pre-registered in notes/preregistration.md.

Estimators per task q and scored configuration m (pool = successful runs of the same scaffold):
  L_H        human reference length (Online-Mind2Web, copied into HAL records)
  L_minus_m  min T over successful runs of all other configurations                     (Eq. 13)
  L_minus_f  min T over successful runs of configurations of other model families      (family = base model)
  L_pool     min T over successful runs of all configurations (one common reference, includes m)
Deviation (data audit): the loop-erased estimator needs the state before every action, which HAL Browser-Use does
not log; it is evaluated on AgentRewardBench instead (analysis_arb.py).
Outputs: results/tables/rq4_*.csv, results/figures/rq4_estimators.pdf
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import statkit as S  # noqa: E402

RES = ROOT / "results"
EST = ["L_minus_m", "L_minus_f", "L_pool"]


def estimators(d):
    """Return one row per successful trajectory (config m, task q) with all estimators."""
    succ = d[(d.excluded.fillna("") == "") & (d.success == 1)][["scaffold", "config", "family", "task_id", "website",
                                                                  "level", "L_H", "T"]]
    rows = []
    for (sc, q), g in succ.groupby(["scaffold", "task_id"]):
        Ts, cs, fs = g["T"].values, g.config.values, g.family.values
        for i in range(len(g)):
            other = cs != cs[i]
            ofam = fs != fs[i]
            rows.append({"scaffold": sc, "task_id": q, "config": cs[i], "family": fs[i],
                         "website": g.website.values[i], "level": g.level.values[i], "L_H": g.L_H.values[i],
                         "T": Ts[i], "L_minus_m": Ts[other].min() if other.any() else np.nan,
                         "L_minus_f": Ts[ofam].min() if ofam.any() else np.nan, "L_pool": Ts.min()})
    return pd.DataFrame(rows)


def ser(T, L):
    return np.where(np.isfinite(L), L / np.maximum(T, L), np.nan)


def summarise(e, d):
    out, strata, rank = [], [], []
    for sc, g in e.groupby("scaffold"):
        sites = sorted(g.website.unique())
        si = {s: i for i, s in enumerate(sites)}
        W = S.cluster_weights(len(sites), offset=4)
        configs = sorted(g.config.unique())
        ci = {c: i for i, c in enumerate(configs)}
        g = g.assign(SER_H=ser(g["T"].values, g.L_H.values.astype(float)))
        for est in EST:
            g[f"SER_{est}"] = ser(g["T"].values, g[est].values.astype(float))
        n_succ = len(g)
        for est in EST:
            cov = g[est].notna().mean()
            ok = g[est].notna()
            # task-level agreement: one value per task (L_pool is task-level; LOAO averaged within task)
            tl = g[ok].groupby("task_id").agg(Lh=("L_H", "first"), Le=(est, "mean"), website=("website", "first"))
            rho = S.spearman(tl.Lh, tl.Le)
            lr = np.log(g.loc[ok, est] / g.loc[ok, "L_H"])
            # rankings by mean SER on successful runs, same covered trajectories for both estimators
            sub = g[ok]

            def cfg_means(col, w=None):
                num = np.zeros((len(configs), len(sites)))
                den = np.zeros((len(configs), len(sites)))
                np.add.at(num, (sub.config.map(ci).values, sub.website.map(si).values), sub[col].values)
                np.add.at(den, (sub.config.map(ci).values, sub.website.map(si).values), 1)
                if w is None:
                    return num.sum(1) / den.sum(1)
                return (w @ num.T) / (w @ den.T)
            a, b = cfg_means("SER_H"), cfg_means(f"SER_{est}")
            keep = np.isfinite(a) & np.isfinite(b)
            tau = S.kendall_b(a[keep], b[keep])
            A, Bm = cfg_means("SER_H", W), cfg_means(f"SER_{est}", W)
            bt = [S.kendall_b(A[i][keep], Bm[i][keep]) for i in range(S.B)]
            lo, hi = S.pct_ci(bt)
            # rho CI by website bootstrap at task level
            tsite = tl.website.map(si).values
            rb = []
            r = S.rng(5)
            for i in range(200):  # 200 resamples for the task-level rho (cheap enough; documented)
                pick = r.integers(0, len(sites), len(sites))
                m = np.isin(tsite, pick)
                cnt = np.bincount(pick, minlength=len(sites))[tsite]
                rr = np.repeat(np.arange(len(tl)), cnt)
                rb.append(S.spearman(tl.Lh.values[rr], tl.Le.values[rr]) if len(rr) > 3 else np.nan)
            out.append({"scaffold": sc, "estimator": est, "n_success_traj": n_succ, "coverage": cov,
                        "spearman_rho_task": rho, "rho_lo": S.pct_ci(rb)[0], "rho_hi": S.pct_ci(rb)[1],
                        "n_tasks": len(tl), "logratio_mean": lr.mean(), "loa_lo": lr.mean() - 1.96 * lr.std(),
                        "loa_hi": lr.mean() + 1.96 * lr.std(), "ratio_median": float(np.exp(lr.median())),
                        "share_below_LH": float((g.loc[ok, est] < g.loc[ok, "L_H"]).mean()),
                        "tau_b_SER_ranking": tau, "tau_lo": lo, "tau_hi": hi, "K": int(keep.sum()),
                        "H4_replaceable": bool(tau >= 0.8 and cov >= 0.80)})
            for lv, h in g[ok].groupby("level"):
                l2 = np.log(h[est] / h.L_H)
                strata.append({"scaffold": sc, "estimator": est, "level": lv, "n": len(h),
                               "ratio_median": float(np.exp(l2.median())), "logratio_mean": l2.mean(),
                               "coverage": float(g[g.level == lv][est].notna().mean())})
            rank.append(pd.DataFrame({"scaffold": sc, "estimator": est, "config": np.array(configs)[keep],
                                      "SER_H": a[keep], "SER_est": b[keep]}))
        # how often does the human reference exceed the shortest agent path (granularity check)
    return pd.DataFrame(out), pd.DataFrame(strata), pd.concat(rank)


def main():
    d = pd.read_csv(RES / "per_trajectory_hal.csv", low_memory=False)
    e = estimators(d)
    e.to_csv(RES / "rq4_estimates_per_trajectory.csv", index=False)
    out, strata, rank = summarise(e, d)
    out.to_csv(RES / "tables" / "rq4_summary.csv", index=False)
    strata.to_csv(RES / "tables" / "rq4_by_level.csv", index=False)
    rank.to_csv(RES / "tables" / "rq4_rankings.csv", index=False)
    # agent T vs human L_H for successful runs (granularity / shortcut check)
    g = e.assign(r=np.log(e["T"] / e.L_H))
    gran = g.groupby("scaffold").agg(n=("r", "size"), median_T_over_LH=("r", lambda x: float(np.exp(x.median()))),
                                     share_T_below_LH=("r", lambda x: float((x < 0).mean())))
    gran.to_csv(RES / "tables" / "rq4_agent_vs_human_length.csv")
    pd.set_option("display.width", 220)
    print(out.round(3).to_string())
    print(strata.round(3).to_string())
    print(gran)
    figure(e)


def figure(e):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, sc in zip(axes, ("browser-use", "seeact")):
        g = e[e.scaffold == sc].dropna(subset=["L_minus_m"])
        tl = g.groupby("task_id").agg(Lh=("L_H", "first"), Lm=("L_minus_m", "mean"))
        ax.scatter(tl.Lh + np.random.default_rng(42).uniform(-.2, .2, len(tl)), tl.Lm, s=8, alpha=.6, color="#2a6fbb")
        m = max(tl.Lh.max(), tl.Lm.max())
        ax.plot([1, m], [1, m], color="grey", lw=.8, ls="--")
        ax.set_xlabel("Human reference length (steps)")
        ax.set_ylabel("Leave-one-out estimate (steps)")
        ax.set_title({"browser-use": "Browser-Use", "seeact": "SeeAct"}[sc], fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "figures" / "rq4_estimators.pdf")


if __name__ == "__main__":
    main()
