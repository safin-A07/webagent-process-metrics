"""RQ1 Discriminability beyond success (HAL, within scaffold). Pre-registered in notes/preregistration.md.

Outputs: results/tables/rq1_config_metrics.csv, rq1_kendall.csv, rq1_pairs.csv, rq1_summary.csv
         results/figures/rq1_tsr_vs_metrics.pdf
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import statkit as S  # noqa: E402

RES = ROOT / "results"
MARGIN = 0.10
METRICS = {"browser-use": [("SER_H", +1), ("ARR_psi", -1), ("SBF_url", -1)],
           "seeact": [("SER_H", +1), ("ARR_psi", -1)]}


def config_table(d, scaffold):
    """Config means with website-cluster bootstrap. Returns table and bootstrap arrays."""
    d = d[d.scaffold == scaffold].copy()
    configs = sorted(d.config.unique())
    sites = sorted(d.website.unique())
    ci_, si_ = {c: i for i, c in enumerate(configs)}, {s: i for i, s in enumerate(sites)}
    W = S.cluster_weights(len(sites), offset=1)
    out, boot = [], {}

    def agg(mask, col):
        x = d.loc[mask]
        num = np.zeros((len(configs), len(sites)))
        den = np.zeros((len(configs), len(sites)))
        np.add.at(num, (x.config.map(ci_).values, x.website.map(si_).values), x[col].astype(float).values)
        np.add.at(den, (x.config.map(ci_).values, x.website.map(si_).values), 1)
        point = num.sum(1) / den.sum(1)
        bs = (W @ num.T) / (W @ den.T)  # (B, configs)
        return point, bs

    valid = d.excluded.fillna("") == ""
    succ = valid & (d.success == 1)
    pt, bs = agg(np.ones(len(d), bool), "success")
    tab = pd.DataFrame({"config": configs, "TSR": pt, "n_tasks": d.groupby("config").size().reindex(configs).values})
    boot["TSR"] = bs
    for col, _ in METRICS[scaffold]:
        for name, mask in (("succ", succ), ("all", valid)):
            p, b = agg(mask, col)
            tab[f"{col}_{name}"] = p
            boot[f"{col}_{name}"] = b
            lo, hi = np.nanquantile(b, 0.025, axis=0), np.nanquantile(b, 0.975, axis=0)
            tab[f"{col}_{name}_lo"], tab[f"{col}_{name}_hi"] = lo, hi
    lo, hi = np.nanquantile(boot["TSR"], 0.025, axis=0), np.nanquantile(boot["TSR"], 0.975, axis=0)
    tab["TSR_lo"], tab["TSR_hi"] = lo, hi
    tab["n_success_traj"] = d[succ].groupby("config").size().reindex(configs).fillna(0).astype(int).values
    tab["share_truncated"] = d[valid].groupby("config").truncated.mean().reindex(configs).values
    tab["median_T"] = d[valid].groupby("config").T.median().reindex(configs).values
    tab.insert(1, "scaffold", scaffold)
    return tab, boot


def kendall_rows(tab, boot, scaffold):
    rows = []
    for col, sign in METRICS[scaffold]:
        for sub in ("succ", "all"):
            m = f"{col}_{sub}"
            tau = S.kendall_b(tab.TSR, sign * tab[m])
            bt = [S.kendall_b(boot["TSR"][i], sign * boot[m][i]) for i in range(S.B)]
            lo, hi = S.pct_ci(bt)
            verdict = "disagree" if hi < 0.8 else ("agree" if lo >= 0.8 else "inconclusive")
            rows.append({"scaffold": scaffold, "metric": col, "subset": sub, "orientation": "higher is better" if sign > 0
                         else "lower is better", "K": len(tab), "tau_b": tau, "ci_lo": lo, "ci_hi": hi,
                         "verdict_H1a": verdict})
    return rows


def pair_rows(d, scaffold):
    d = d[d.scaffold == scaffold]
    piv_s = d.pivot_table(index="task_id", columns="config", values="success", aggfunc="first")
    valid = d[d.excluded.fillna("") == ""]
    rows = []
    for k, (a, b) in enumerate(combinations(sorted(d.config.unique()), 2)):
        both = piv_s[[a, b]].dropna()
        diff, p_tost = S.tost_paired_binary(both[a], both[b], MARGIN)
        p_mcn = S.mcnemar_exact(both[a], both[b])
        row = {"scaffold": scaffold, "config_a": a, "config_b": b, "n_tasks": len(both), "TSR_diff": diff,
               "p_tost": p_tost, "tsr_equivalent": p_tost < 0.05, "p_mcnemar": p_mcn}
        for col, _ in METRICS[scaffold]:
            pv = valid.pivot_table(index="task_id", columns="config", values=col, aggfunc="first")
            if a not in pv or b not in pv:
                continue
            x = pv[[a, b]].dropna()
            dd = (x[a] - x[b]).values
            row[f"{col}_n"] = len(dd)
            row[f"{col}_meandiff"] = float(dd.mean()) if len(dd) else np.nan
            row[f"{col}_HL"] = S.hodges_lehmann(dd)
            row[f"{col}_p"] = S.paired_permutation(dd, offset=k)
        rows.append(row)
    pr = pd.DataFrame(rows)
    for col, _ in METRICS[scaffold]:
        pr[f"{col}_p_holm"] = S.holm(pr[f"{col}_p"].values)
        pr[f"{col}_sig"] = pr[f"{col}_p_holm"] < 0.05
    pr["mcnemar_p_holm"] = S.holm(pr.p_mcnemar.values)
    return pr


def main():
    d = pd.read_csv(RES / "per_trajectory_hal.csv", low_memory=False)
    tabs, ks, prs = [], [], []
    for sc in ("browser-use", "seeact"):
        tab, boot = config_table(d, sc)
        tabs.append(tab)
        ks += kendall_rows(tab, boot, sc)
        prs.append(pair_rows(d, sc))
        print(sc, "done", flush=True)
    tab = pd.concat(tabs)
    tab.to_csv(RES / "tables" / "rq1_config_metrics.csv", index=False)
    kt = pd.DataFrame(ks)
    kt.to_csv(RES / "tables" / "rq1_kendall.csv", index=False)
    pr = pd.concat(prs)
    pr.to_csv(RES / "tables" / "rq1_pairs.csv", index=False)
    summ = []
    for sc in ("browser-use", "seeact"):
        p = pr[pr.scaffold == sc]
        for col, _ in METRICS[sc]:
            summ.append({"scaffold": sc, "metric": col, "pairs": len(p), "tsr_equivalent": int(p.tsr_equivalent.sum()),
                         "tsr_mcnemar_sig_holm": int((p.mcnemar_p_holm < 0.05).sum()),
                         "metric_sig_holm": int(p[f"{col}_sig"].sum()),
                         "equivalent_and_metric_sig": int((p.tsr_equivalent & p[f"{col}_sig"]).sum())})
    pd.DataFrame(summ).to_csv(RES / "tables" / "rq1_summary.csv", index=False)
    print(kt.to_string())
    print(pd.DataFrame(summ).to_string())
    figure(tab)


def figure(tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (col, lab) in zip(axes, [("SER_H_succ", "SER (successful runs)"), ("ARR_psi_succ", "ARR (successful runs)"),
                                     ("SBF_url_succ", "SBF (successful runs)")]):
        for sc, mk, colr in (("browser-use", "o", "#2a6fbb"), ("seeact", "s", "#d1495b")):
            t = tab[tab.scaffold == sc]
            if col not in t or t[col].isna().all():
                continue
            ax.errorbar(t.TSR, t[col], yerr=[t[col] - t[col + "_lo"], t[col + "_hi"] - t[col]], fmt=mk, ms=4,
                        color=colr, ecolor=colr, alpha=.8, elinewidth=.6, label=sc)
        ax.set_xlabel("Task success rate")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "figures" / "rq1_tsr_vs_metrics.pdf")


if __name__ == "__main__":
    main()
