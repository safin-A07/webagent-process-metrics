"""SER by difficulty stratum (HAL, Online-Mind2Web levels: easy <= 5, medium 6-10, hard >= 11 human steps).

Answers the coverage-bias concern of RQ4: the leave-one-configuration-out estimator is undefined on tasks that no
other configuration solves, which could bias SER towards easy tasks. For each scaffold and level we report TSR,
mean SER under the human reference (successful runs and all runs), the coverage of L_minus_m, and mean SER on the
covered successful runs under both references, plus the range of configuration means.
Inputs: results/per_trajectory_hal.csv, results/rq4_estimates_per_trajectory.csv
Outputs: results/tables/ser_by_level.csv, paper/tables/strata.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LEVELS = ["easy", "medium", "hard"]


def main():
    d = pd.read_csv(RES / "per_trajectory_hal.csv", low_memory=False)
    d = d[d.excluded.fillna("") == ""]
    e = pd.read_csv(RES / "rq4_estimates_per_trajectory.csv")
    e["SER_m"] = np.where(e.L_minus_m.notna(), e.L_minus_m / np.maximum(e["T"], e.L_minus_m), np.nan)
    e["SER_H"] = e.L_H / np.maximum(e["T"], e.L_H)
    rows = []
    for (sc, lv), g in d.groupby(["scaffold", "level"]):
        s = g[g.success == 1]
        es = e[(e.scaffold == sc) & (e.level == lv)]
        cov = es[es.L_minus_m.notna()]
        cfg = s.groupby("config").SER_H.mean()
        rows.append({"scaffold": sc, "level": lv, "n_runs": len(g), "n_tasks": g.task_id.nunique(),
                     "TSR": g.success.mean(), "n_success": len(s),
                     "SER_H_success": s.SER_H.mean(), "SER_H_all": g.SER_H.mean(),
                     "SER_H_cfg_min": cfg.min(), "SER_H_cfg_max": cfg.max(),
                     "coverage_minus_m": len(cov) / len(es) if len(es) else np.nan,
                     "SER_H_covered": cov.SER_H.mean(), "SER_minus_m_covered": cov.SER_m.mean()})
    t = pd.DataFrame(rows)
    t["level"] = pd.Categorical(t.level, LEVELS, ordered=True)
    t = t.sort_values(["scaffold", "level"])
    t.to_csv(RES / "tables" / "ser_by_level.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    tex(t)


def tex(t):
    f = lambda x: f"{x:.2f}"  # noqa: E731
    name = {"browser-use": "Browser-Use", "seeact": "SeeAct"}
    body = "\n".join(
        f"{name[r.scaffold]} & {r.level} & {r.n_tasks} & {100 * r.TSR:.1f} & {r.n_success} & {f(r.SER_H_success)} "
        f"[{f(r.SER_H_cfg_min)}, {f(r.SER_H_cfg_max)}] & {f(r.SER_H_all)} & {f(r.coverage_minus_m)} & "
        f"{f(r.SER_H_covered)} & {f(r.SER_minus_m_covered)} \\\\" for r in t.itertuples())
    (ROOT / "paper" / "tables" / "strata.tex").write_text(r"""\begin{table}[t]
\centering
\caption{SER by difficulty level on HAL (levels from the human reference length: easy $\le 5$, medium 6--10,
hard $\ge 11$ steps). TSR in \%. SER$_{\mathrm{H}}$: mean SER under $\hat L^{*}_{\mathrm{H}}$ on successful runs, with
the range of configuration means, and on all runs. Cov.: share of successful runs for which $\hat L^{*}_{-m}$ is
defined. The last two columns compare both references on the covered successful runs.}
\label{tab:strata}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llcccccccc@{}}
\toprule
& & & & & \multicolumn{2}{c}{SER$_{\mathrm{H}}$} & & \multicolumn{2}{c}{Covered runs} \\
\cmidrule(lr){6-7}\cmidrule(l){9-10}
Scaffold & Level & Tasks & TSR & Succ. & Successful [range] & All & Cov. & SER$_{\mathrm{H}}$ & SER$_{-m}$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""", encoding="utf-8")


if __name__ == "__main__":
    main()
