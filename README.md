# Cost-grounded process-level metrics for LLM web agents

Code, tests, pre-registered analysis plan and derived results for the paper

> Md Safin Hossain and Jasem Chawdhory. *Beyond the Destination: Cost-Grounded Process-Level Evaluation Metrics
> for LLM-Based Web Navigation Agents.* Manuscript in preparation, 2026.

The paper defines process metrics that are computed post hoc from released web-agent logs, for successful and
failed runs:

| Metric | Definition |
| --- | --- |
| **SER** (Step Efficiency Ratio) | `L* / max(T, L*)`; `SER_S = S * SER` equals SPL with path length in actions |
| **ARR** (Action Redundancy Rate) | share of steps whose (abstract state, abstract action) pair occurred earlier |
| **SBF** (State Backtracking Frequency) | share of steps that return to an earlier state other than the one being left |
| **omega** (redundancy-attributed cost share) | share of token cost spent on redundant or backtracking steps |

plus agent-based estimators of the optimal path length `L*` (leave-one-configuration-out, leave-one-family-out,
fixed pool, loop-erased).

## Repository layout

```
code/procmetrics.py        reference implementation of the metrics (incl. the one-pass O(T) version, Algorithm 1)
code/test_procmetrics.py   tests: worked example, Propositions 1-4, one-pass == reference
code/statkit.py            bootstrap, rank correlations, permutation tests, TOST
code/loaders.py            common trajectory format
scripts/                   download, decryption, extraction, metric computation, analyses, tables
notes/preregistration.md   hypotheses and decision rules, fixed before any metric was computed
DEVIATIONS.md              every deviation from the pre-registered plan, with reasons
results/                   derived per-trajectory metric values, result tables (CSV) and figures (PDF)
paper/tables/              LaTeX tables generated from results/tables by scripts/make_tables.py
```

## Quick check

```bash
pip install -r requirements.txt
python code/test_procmetrics.py
```

## Reproducing the results

The raw data are **not** included (see "Data" below). With the data in `data/raw/`, run from the repository root:

```bash
python -I scripts/download_data.py            # Online-Mind2Web (gated: accept terms, `huggingface-cli login`), HAL traces, AgentRewardBench
export HAL_DECRYPT_PASSWORD=...               # the password published in HAL's hal-harness repository (hal/utils/decrypt.py)
python -I scripts/hal_decrypt.py              # decrypt the HAL archives into data/raw/hal/
python -I scripts/extract_hal.py              # compact HAL cache (data/processed/)
python -I scripts/extract_arb.py              # compact AgentRewardBench cache
python -I scripts/compute_metrics.py          # results/per_trajectory_{hal,arb}.csv
python -I scripts/analysis_rq1.py             # RQ1
python -I scripts/analysis_rq4.py             # RQ4
python -I scripts/analysis_hal_extra.py       # RQ2 (HAL, configuration level), RQ5, early warning (HAL)
python -I scripts/analysis_arb.py             # RQ2, RQ3, RQ5, early warning, loop-erased L* (AgentRewardBench)
python -I scripts/analysis_strata.py          # SER by difficulty level
python -I scripts/hal_cost_flags.py           # HAL span audit and cost sensitivity
python -I scripts/make_tables.py              # LaTeX tables
python -I scripts/checksums.py                # SHA-256 of every input file
```

`-I` runs Python in isolated mode, which is recommended because the downloaded files are untrusted input.
All resampling uses seed 42. The analyses are CPU-only post-processing; the full pipeline ran on a 2-core laptop
with 12 GB RAM.

## Data

All analysed data are released by their original authors; this repository contains only derived
per-trajectory metric values, aggregate results and SHA-256 checksums of the input files
(`results/tables/checksums.csv`).

| Source | Version used | Terms |
| --- | --- | --- |
| HAL traces for Online-Mind2Web (Hugging Face `agent-evals/hal_traces`) | commit `e7dcedc82b4f4bc819a170fd6616bdb44841c71e` | Encrypted by the provider to limit contamination; traces are not redistributed here. |
| Online-Mind2Web tasks (`osunlp/Online-Mind2Web`) | as copied into the HAL records | CC-BY-4.0, gated; only task identifiers, websites, levels and reference lengths appear in the derived files. |
| AgentRewardBench (`McGill-NLP/agent-reward-bench`) | commit `b6d17e646009d6cb63d5dd7be78807b680693f61` | See `DATA_NOTICE.md`. |

## Licence

Code: MIT (see `LICENSE`). Derived data: subject to the terms of the original datasets (`DATA_NOTICE.md`).

## Citation

See `CITATION.cff`. Please also cite the HAL, Online-Mind2Web and AgentRewardBench papers when using the derived
data.
