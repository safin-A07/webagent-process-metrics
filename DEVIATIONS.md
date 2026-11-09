# Deviations from the pre-registered plan

The plan in `notes/preregistration.md` was written after the data audit and before any process metric was
computed. Every deviation is listed here with its reason; the paper states the same deviations.

1. **HAL ARR is state-free only.** The HAL logs do not record the state before each action, so ARR on HAL is the
   state-free variant, which equals one minus the unique-action ratio.
2. **HAL SBF is a lower bound.** HAL Browser-Use records one URL per model call, so SBF is computed on that
   subsequence of states. SeeAct records no URL, so SBF is not computed for SeeAct.
3. **RQ2 per-trajectory regression uses AgentRewardBench only.** HAL reports tokens per run only, so HAL enters
   RQ2 at the configuration level. The primary outcome is tokens, because two of the four AgentRewardBench agents
   are open-weight models without a list price.
4. **Loop-erased L\* is exploratory and AgentRewardBench-only.** It needs the state before every action, which
   HAL does not log. The phi_URL version is not meaningful (it erases actions that do not change the URL, such
   as typing), so the phi_URL+AX version is reported.
5. **RQ4 coverage definition.** Coverage is the share of the scored configuration's successful runs for which
   the estimator is defined, because SER is computed on successful runs. Family exclusion is done within a
   scaffold.
6. **Fewer bootstrap resamples in three places** (to save time; stated in the code): 200 for the task-level rho
   CI of RQ4, 300 for the AgentRewardBench RQ5 robustness CIs, 500 for the AgentRewardBench early-warning CI.
   All other intervals use 1,000.
7. **RQ3 was run once before all LLM-judge files were downloaded.** The reported numbers come from the rerun on
   all 4 x 1,302 judge files.
8. **L_H source.** The gated Online-Mind2Web file was not downloaded. L_H is taken from the `reference_length`
   field that HAL copies into every record; the copies agree across all 36 configurations.
9. **Analyses added after review (not pre-registered, reported as such):**
   - SER by difficulty level (`scripts/analysis_strata.py`), to check the coverage bias of the
     leave-one-configuration-out estimator.
   - A span audit of the HAL logs (`scripts/hal_cost_flags.py`) that identifies configurations whose model
     calls are logged twice, and a sensitivity analysis of the configuration-level cost correlations with and
     without them.
