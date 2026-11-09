"""Generate the LaTeX results tables of Section 8 directly from results/tables/*.csv (no hand-copied numbers).

Usage: python -I scripts/make_tables.py      -> paper/tables/*.tex
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(exist_ok=True)

NAMES = {"SER_H": "SER", "ARR_psi": "ARR$^{\\psi}$", "SBF_url": "SBF", "ARR_url": "ARR ($\\phi_{\\mathrm{URL}}$)",
         "ARR_ax": "ARR ($\\phi_{\\mathrm{URL+AX}}$)", "ARR_plus_SBF": "ARR + SBF",
         "ARR_url_noscroll": "ARR (scroll/wait exempt)", "T": "Length $T$",
         "B_longest_repeat": "Longest repeat / $T$", "B_trigram_loop": "Trigram loop",
         "L_minus_m": "$\\hat L^{*}_{-m}$", "L_minus_f": "$\\hat L^{*}_{-f}$", "L_pool": "$\\hat L^{*}_{\\mathrm{pool}}$"}


def f(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def ci(x, lo, hi, d=2):
    return f"{f(x, d)} [{f(lo, d)}, {f(hi, d)}]"


def write(name, body):
    (OUT / name).write_text(body, encoding="utf-8")
    print("wrote", name)


def rq1():
    k = pd.read_csv(T / "rq1_kendall.csv")
    s = pd.read_csv(T / "rq1_summary.csv")
    rows = []
    for sc in ("browser-use", "seeact"):
        for m in ("SER_H", "ARR_psi", "SBF_url"):
            a = k[(k.scaffold == sc) & (k.metric == m) & (k.subset == "succ")]
            b = k[(k.scaffold == sc) & (k.metric == m) & (k.subset == "all")]
            p = s[(s.scaffold == sc) & (s.metric == m)]
            if a.empty:
                continue
            a, b, p = a.iloc[0], b.iloc[0], p.iloc[0]
            rows.append(f"{'Browser-Use' if sc == 'browser-use' else 'SeeAct'} ({int(a.K)}) & {NAMES[m]} & "
                        f"{ci(a.tau_b, a.ci_lo, a.ci_hi)} & {ci(b.tau_b, b.ci_lo, b.ci_hi)} & {int(p.pairs)} & "
                        f"{int(p.tsr_equivalent)} & {int(p.metric_sig_holm)} & {int(p.equivalent_and_metric_sig)} \\\\")
    body = r"""\begin{table}[t]
\centering
\caption{RQ1: agreement between configuration rankings by TSR and by each process metric (Kendall's $\tau_b$,
metric oriented so that higher is better, 95\% website-cluster bootstrap CI), and pairwise tests within each
scaffold. Equiv.: pairs whose success rates are equivalent by TOST ($\Delta = 10$ points); Sig.: pairs whose
metric differs (permutation test, Holm); Both: equivalent in success and different in the metric. ARR$^{\psi}$ is
the state-free ARR (equal to the unique-action ratio), the only form computable on HAL.}
\label{tab:rq1}
\footnotesize
\begin{tabular}{@{}llccrrrr@{}}
\toprule
Scaffold ($K$) & Metric & $\tau_b$ (successful runs) & $\tau_b$ (all runs) & Pairs & Equiv. & Sig. & Both \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("rq1.tex", body)


def rq4():
    s = pd.read_csv(T / "rq4_summary.csv")
    g = pd.read_csv(T / "rq4_agent_vs_human_length.csv")
    rows = []
    for _, r in s.iterrows():
        rows.append(f"{'Browser-Use' if r.scaffold == 'browser-use' else 'SeeAct'} & {NAMES[r.estimator]} & "
                    f"{f(r.coverage)} & {ci(r.spearman_rho_task, r.rho_lo, r.rho_hi)} & {f(r.ratio_median)} & "
                    f"{f(r.logratio_mean)} [{f(r.loa_lo)}, {f(r.loa_hi)}] & {ci(r.tau_b_SER_ranking, r.tau_lo, r.tau_hi)} & "
                    f"{'yes' if r.H4_replaceable else 'no'} \\\\")
    body = r"""\begin{table}[t]
\centering
\caption{RQ4: agent-based estimators of the optimal path length against the human reference
$\hat L^{*}_{\mathrm{H}}$ on Online-Mind2Web (successful runs). Coverage: share of successful runs for which the
estimator is defined. $\rho$: task-level Spearman correlation with $\hat L^{*}_{\mathrm{H}}$. Ratio: median of
$\hat L^{*}/\hat L^{*}_{\mathrm{H}}$. Bias [LoA]: mean and 95\% limits of agreement of the log ratio. $\tau_b$:
agreement of configuration rankings by mean SER. H4: $\tau_b \ge 0.8$ and coverage $\ge 0.80$.}
\label{tab:rq4}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}llcccccc@{}}
\toprule
Scaffold & Estimator & Coverage & $\rho$ [CI] & Ratio & Bias [LoA] & $\tau_b$ [CI] & H4 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("rq4.tex", body)


def rq3():
    a = pd.read_csv(T / "rq3_auroc.csv")
    h = pd.read_csv(T / "rq3_heldout.csv").set_index("score")
    rows = []
    for _, r in a.iterrows():
        hf = h.loc[r.score, "f1"] if r.score in h.index else float("nan")
        rows.append(f"{NAMES.get(r.score, r.score)} & {ci(r.auroc, r.auroc_lo, r.auroc_hi)} & "
                    f"{ci(r.auprc, r.auprc_lo, r.auprc_hi)} & {f(r.point_biserial)} & {f(hf)} \\\\")
    inc = pd.read_csv(T / "rq3_incremental.csv")
    MOD = {"ARR_ax": NAMES["ARR_ax"], "ARR_psi": NAMES["ARR_psi"], "ARR_url": NAMES["ARR_url"],
           "ARR_url+SBF_url": "ARR + SBF", "B_trigram_loop": "Trigram loop"}
    irows = [f"$\\log T$ + {MOD.get(r.model, r.model)} & {f(r.auroc_logT)} & {f(r.auroc_model)} & "
             f"{f(r.delta_mean, 3)} [{f(r.delta_lo, 3)}, {f(r.delta_hi, 3)}] \\\\" for _, r in inc.iterrows()]
    j = pd.read_csv(T / "rq3_judges.csv")
    JD = {"LLM judge: gpt-4o-noscreen": "LLM judge: GPT-4o",
          "LLM judge: claude-3.7-sonnet-noscreen": "LLM judge: Claude 3.7 Sonnet",
          "LLM judge: llama-3.3-70b-noscreen": "LLM judge: Llama-3.3-70B",
          "LLM judge: qwen-2.5-vl-noscreen": "LLM judge: Qwen2.5-VL"}

    def jname(x):
        if x in JD:
            return JD[x]
        return "ARR ($\\phi_{\\mathrm{URL}}$) $\\ge$ " + x.split(">= ")[1].split(" ")[0] + " (rule)"
    jrows = [f"{jname(r.detector)} & {int(r.n)} & {f(r.f1)} & {f(r.precision)} & "
             f"{f(r.recall)} \\\\" for _, r in j.iterrows()]
    body = r"""\begin{table}[t]
\centering
\caption{RQ3: prediction of the expert repetition-cycle label on AgentRewardBench (prevalence """ + \
        f"{a.prevalence.iloc[0]:.2f}, $n={int(a.n.iloc[0])}$" + r"""). Top: each score alone, with 95\%
task-cluster bootstrap CIs and the mean held-out F1 over 100 task splits. Middle: out-of-sample gain in AUROC when
the score is added to $\log T$ (mean and 2.5--97.5\% range over the splits). Bottom: binary detectors on the same
trajectories.}
\label{tab:rq3}
\footnotesize
\begin{tabular}{@{}lcccc@{}}
\toprule
Score & AUROC [CI] & AUPRC [CI] & $r_{pb}$ & Held-out F1 \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
Model & AUROC ($\log T$) & AUROC (model) & \multicolumn{2}{c}{$\Delta$AUROC [range]} \\
\midrule
""" + "\n".join(r.replace("] \\\\", "] \\\\").replace(" & " + r.split(" & ")[-1], " & \\multicolumn{2}{c}{" + r.split(" & ")[-1].replace(" \\\\", "") + "} \\\\") for r in irows) + r"""
\midrule
Detector & $n$ & F1 & Precision & Recall \\
\midrule
""" + "\n".join(jrows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("rq3.tex", body)


def rq2():
    r = pd.read_csv(T / "rq2_arb_regression.csv").set_index("term")
    om = pd.read_csv(T / "rq2_arb_omega.csv")
    lab = {"S": "Success $S$", "logT": "$\\log T$", "ARR_url": "ARR ($\\phi_{\\mathrm{URL}}$)", "SBF_url": "SBF"}
    rows = [f"{lab[t]} & {ci(r.loc[t, 'coef'], r.loc[t, 'ci_lo'], r.loc[t, 'ci_hi'], 3)} \\\\" for t in lab]
    d = r.loc["delta_within_R2 (ARR,SBF beyond S, logT)"]
    rows.append(f"$\\Delta$ within-$R^2$ from (ARR, SBF) & {ci(d.coef, d.ci_lo, d.ci_hi, 3)} \\\\")
    rows.append(f"Within-$R^2$ ($S$, $\\log T$) / full & {f(r.loc['within_R2_base', 'coef'], 3)} / "
                f"{f(r.loc['within_R2_full', 'coef'], 3)} \\\\")
    rows.append(f"Trajectories & {int(r.loc['n_trajectories', 'coef'])} \\\\")
    agent = {"Qwen": "Qwen2.5-VL-72B", "claude": "Claude 3.7 Sonnet", "gpt-4o": "GPT-4o", "Llama": "Llama-3.3-70B",
             "all": "All agents"}
    lbl = lambda c: next(v for k, v in agent.items() if k in c)  # noqa: E731
    orows = [f"{lbl(c.config)} & {int(c.n)} & {f(c.omega_mean, 3)} & "
             f"{f(c.omega_median, 3)} & {f(c.share_tokens_flagged, 3)} \\\\" for _, c in om.iterrows()]
    body = r"""\begin{table}[t]
\centering
\caption{RQ2 on AgentRewardBench. Top: fixed-effects regression of log tokens per trajectory (Eq.~\ref{eq:regression},
task and agent fixed effects; 95\% task-cluster bootstrap CIs). Bottom: redundancy-attributed cost share $\omega$
with per-step token costs; the last column is the share of all tokens of the agent spent on flagged steps.}
\label{tab:rq2}
\footnotesize
\begin{tabular}{@{}lc@{}}
\toprule
Term & Coefficient [CI] \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\medskip
\begin{tabular}{@{}lcccc@{}}
\toprule
Agent & $n$ & Mean $\omega$ & Median $\omega$ & Tokens on flagged steps \\
\midrule
""" + "\n".join(orows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("rq2.tex", body)


if __name__ == "__main__":
    import sys
    for name in (sys.argv[1:] or ["rq1", "rq4", "rq3", "rq2"]):
        globals()[name]()
