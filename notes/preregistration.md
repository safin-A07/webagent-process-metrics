> **Note added at code release (9 Nov 2026):** the text below is unchanged. "Supervisor feedback" refers to internal review comments; the file `IMPLEMENTATION_README.md` mentioned below is the authors' working log, and every deviation listed there is reproduced in `DEVIATIONS.md`.

# Pre-registered analysis plan (written 2026-11-06, before any metric was computed)

This file fixes hypotheses, decision thresholds, data scope and tests **before** SER/ARR/SBF were
computed on any real trajectory. It answers supervisor feedback A1 ("add hypotheses with direction and
decision thresholds", "power analysis") and B (statistics). It is binding: deviations must be listed in
`IMPLEMENTATION_README.md` under "Deviations" with a reason.

What had been seen before writing this (honest disclosure): field presence per source (data audit), the
leaderboard success counts of three HAL configurations (DeepSeek-R1 BU 76/300, GPT-4.1 BU 109/300,
Gemini-2.0-Flash SeeAct 80/300), the SeeAct `exit_by` counts of one configuration, and the
AgentRewardBench label marginals (looping Yes 711 / No 697 rows; success 395/1012/1 rows). No process
metric, cost, correlation or ranking had been computed.

## 0. Data scope (fixed)

| Source | Unit | Used for | Known limits (from data audit) |
| --- | --- | --- | --- |
| HAL Browser-Use (BU), all configs that decrypt | trajectory = (config, task) | RQ1, RQ2 (config level), RQ4, RQ5, U1 | URL per model step; **no per-step or per-trajectory tokens** (run totals only) |
| HAL SeeAct, all configs that decrypt | trajectory | RQ1 (TSR, SER, ARR_noState only), RQ4 (min-T estimators only) | **no URL per step**, no per-step tokens |
| AgentRewardBench (ARB) cleaned/, 1,302 trajectories | trajectory | RQ2 (per trajectory), RQ3, RQ5, U1 | URL + accessibility tree + per-step tokens |
| ARB LLM judges: gpt-4o-noscreen, claude-3.7-sonnet-noscreen, llama-3.3-70b-noscreen, qwen-2.5-vl-noscreen | trajectory | RQ3 comparison | binary `<loop>` answer only |

- L_H (human reference length) is taken from the `reference_length` field that HAL copies from
  Online-Mind2Web into every record; it is cross-checked against the gated Online-Mind2Web file if the
  user's HF login becomes available.
- SeeAct has no URL, so SBF, φ-based ARR, loop-erased L* and ω are **not** computed for SeeAct. For SeeAct we
  report ARR_noState (an action is redundant if the identical action string was executed earlier in the
  episode), clearly labelled as a different, weaker quantity.
- `seeact_gpt-5_reasoning_effort_high_UPLOAD_updated.zip` (12.8 GB) is not used; the non-updated file is.

## 1. Normalisation (Section 5.2 of the paper, plus additions)

1. Each executed browser action is one step. Browser-Use: actions in `model_actions` of a model step,
   truncated to the number of entries in `action_results` (proposed but not executed actions are dropped).
   `done` is the terminal answer and not counted. SeeAct: each entry of `take_actions_and_elements` whose
   action is not a terminal answer.
2. Browser-Use state before the k-th action of model step i is `urls[i]`; the state after the last action
   of step i is `urls[i+1]` (or the last URL for the final step). Intermediate actions inside one model step
   are assigned `urls[i]` (no navigation is observable between them); this is stated as a limitation.
3. Action abstraction ψ = (type, target, value). Browser-Use targets elements by **index**; fallback:
   target = index (indices are per-page, so the pair (state, index) is still meaningful for ARR). ARB:
   target = BrowserGym bid; value = normalised typed text.
4. Scroll/wait sensitivity (feedback A4): ARR and SBF are also computed with scroll, wait and
   `extract_content`/no-op actions removed from the redundancy check (they still count in T). Reported as
   "ARR (excl. scroll/wait)".
5. Step-limit truncation: flagged (BU max 30 model steps; SeeAct `exit_by` step limit; ARB `truncated`).
6. Exclusions: trajectories with T = 0 or unparseable logs, counted by reason.

## 2. Hypotheses, tests and decision rules

All CIs: 95 % percentile cluster bootstrap, B = 1,000, seed 42. Clusters: **website** for HAL (tasks are
nested in 136 websites, so this also accounts for task clustering), **task** for ARB.

### RQ1 Discriminability (HAL; analyses run **within scaffold**, BU and SeeAct separately)

- Metrics per config: TSR; mean SER (L_H), ARR, SBF on **successful** trajectories (primary, feedback B
  "condition on successful runs") and on all trajectories (secondary, truncated runs reported separately).
- H1a (direction: disagreement). Kendall τb between config rankings by TSR and by each process metric.
  Decision: "rankings disagree" iff the upper CI bound of τb < 0.8; "agree" iff the lower bound ≥ 0.8;
  otherwise "inconclusive".
- H1b. For every config pair within a scaffold: TSR equivalence by **TOST** on paired task outcomes, margin
  Δ = 10 percentage points, α = 0.05 per side; metric difference by **paired permutation test**
  (10,000 sign flips, all tasks both configs attempted), Holm-corrected within each metric family, effect
  size = Hodges–Lehmann estimate. Supported iff ≥ 1 pair is TSR-equivalent and metric-different.
- Power note: with n = 300 paired tasks and a discordance rate ≤ 0.40, the SE of the TSR difference is
  ≤ 0.0365, so TOST at Δ = 10 pp can declare equivalence only when |observed difference| ≲ 4 pp; at
  Δ = 5 pp it practically never can. This is why Δ = 10 pp is the primary margin.

### RQ2 Cost grounding

- ARB (per trajectory, real per-step tokens). Primary outcome: log total tokens (input + output);
  secondary: log cost with one fixed price table (API models only; open-weight models are reported with
  tokens only). Model (feedback F):
  log C_ij = α_i + β_j + γ S_ij + θ log T_ij + δ1 ARR_ij + δ2 SBF_ij + ε_ij (task and agent fixed effects).
  H2 supported iff the CI of δ1 or δ2 excludes 0 **and** the within-R² increases by ≥ 0.01 when
  (ARR, SBF) are added to (S, log T).
- HAL (no per-trajectory tokens): configuration-level Spearman ρ between tokens per task (from run totals)
  and mean T, ARR, SBF, BU only; exploratory (n ≈ 19). Reported with and without configurations the
  leaderboard flags as over-reporting cost.
- Redundancy-attributed cost share ω (renamed from "wasted cost share"): distribution on ARB with per-step
  costs; on HAL BU only the constant-cost version (fraction of flagged steps), labelled as such. ω is not a
  counterfactual saving.

### RQ3 Agreement with human judgement (ARB)

- Label: expert `trajectory_looping`. Trajectories with several annotations: majority vote; ties excluded
  from the primary analysis and counted.
- Scores: ARR (φ_URL), ARR (φ_URL+AX), SBF, ARR+SBF; baselines: T, unique-action ratio (1 − #unique ψ / T),
  longest repeated contiguous action subsequence / T, and a trigram loop detector (any ψ-trigram ≥ 2 times).
- Report AUROC **and AUPRC**, point-biserial r. Threshold: max F1 on a random half of **tasks** (split
  within each benchmark), evaluated on the other half, 100 splits; mean ± sd of held-out F1/precision/recall.
- Incremental validity: out-of-sample ΔAUROC of logistic(label ~ log T + ARR) over logistic(label ~ log T),
  same splits.
- H3 supported iff ARR AUROC lower CI bound ≥ 0.70 and the ΔAUROC CI excludes 0.
- Judges: held-out F1 of the ARR detector vs each LLM judge's `<loop>` answer on the same trajectories.

### RQ4 Optimal path without demonstrations (HAL)

- Estimators per task: L_H; L̂₋ₘ leave-one-config-out min T over successful runs (within scaffold);
  L̂₋fam leaving out the whole **model family** (base model, ignoring reasoning/thinking setting) across
  both scaffolds; L̂_pool fixed pool (min over all configs of the scaffold, one common reference);
  L̂_LE loop-erased minimum (BU only: remove cycles in the URL state sequence of each successful run).
- Report coverage; Spearman ρ with L_H; log-ratio Bland–Altman (mean bias and 95 % limits of agreement);
  ratio by difficulty stratum (`level`); Kendall τb between config rankings by mean SER (successful runs)
  under L_H and under each estimator.
- H4 decision: an estimator "can replace L_H for ranking" iff τb ≥ 0.8 **and** coverage ≥ 0.80.
- Assumptions stated: the estimator approaches L* only if some agent eventually finds an optimal path;
  judge false positives can bias it downwards; the min is a noisy extreme statistic.

### RQ5 Robustness and distinctness

- ARB: trajectory-level Spearman of ARR and SBF under φ_URL vs φ_URL+AX; "robust" iff ρ ≥ 0.8. The four
  ARB agents are ranked descriptively only (K = 4 is too small for τb inference, feedback B).
- Distinctness: Spearman matrix of SER, ARR, SBF, T, S at trajectory level (HAL BU, ARB) and config level
  (HAL BU); partial Spearman controlling for log T. "Distinct" iff |ρ| < 0.8 between metric pairs.

### U1 Downstream utility (feedback B "early warning")

- Prefix features ARR and SBF on the first k = 10 steps, trajectories with T > 10 (HAL BU, ARB); AUROC for
  eventual failure. Early-termination rule thresholded on one half of tasks: steps saved and successes
  lost on the other half (100 splits).

### Not performed by the assistant

- The human efficiency study (feedback B) needs real human annotators; an annotation kit is prepared
  (`scripts/make_annotation_kit.py`). If the authors do not carry it out, it is reported as a limitation.
