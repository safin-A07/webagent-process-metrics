"""Which HAL configurations are affected by the leaderboard's token over-reporting, and does it change RQ2?

HAL's notice (5 Oct 2026) says that nested trace spans counted the same API response twice, so affected rows are
inflated by 2x, and that models logged as a single span are unaffected; it does not name the affected rows.
We identify them from the released logs: a configuration is flagged as double-logged when (almost) every model
call is recorded both by a framework-level span (langchain.Llm.*, litellm.completion) and by a provider-level
span (openai.*.create / parse, anthropic.*Messages.create), i.e. min/max of the two counts >= 0.9.

Sensitivity of the configuration-level RQ2 correlations: (A) tokens as released, (B) tokens of flagged
configurations halved, (C) flagged configurations excluded.
Inputs: data/raw/hal/*/*.json (decrypted traces), results/tables/rq2_hal_config_values.csv
Outputs: results/tables/hal_span_audit.csv, results/tables/rq2_hal_config_flags.csv
Run with: python -I scripts/hal_cost_flags.py [--reuse]   (--reuse skips the slow log scan)
"""
import collections
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import statkit as S  # noqa: E402

RES = ROOT / "results" / "tables"
PROVIDER = ("openai.chat.completions.create", "openai.beta.chat.completions.parse", "anthropic.AsyncMessages.create",
            "anthropic.Messages.create")


def op_name(c):
    return (c.get("op_name") or "").split("/op/")[-1].split(":")[0]


def audit():
    rows = []
    for f in sorted((ROOT / "data" / "raw" / "hal").glob("*/*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        ops = collections.Counter(op_name(c) for c in d.get("raw_logging_results") or [])
        wrapper = sum(v for k, v in ops.items() if k.startswith("langchain.Llm.") or k == "litellm.completion")
        provider = sum(v for k, v in ops.items() if k in PROVIDER)
        share = min(wrapper, provider) / max(wrapper, provider) if wrapper and provider else 0.0
        usage = d.get("total_usage") or {}
        rows.append({"config": f.stem.removesuffix("_UPLOAD"), "wrapper_spans": wrapper, "provider_spans": provider,
                     "double_share": round(share, 3), "flagged_double": share >= 0.9,
                     "n_usage_keys": sum(1 for k in usage if "embedding" not in k)})
        print(rows[-1], flush=True)
        del d
    a = pd.DataFrame(rows)
    a.to_csv(RES / "hal_span_audit.csv", index=False)
    return a


def sensitivity(a):
    v = pd.read_csv(RES / "rq2_hal_config_values.csv", index_col="config")
    flag = a.set_index("config").flagged_double.reindex(v.index).fillna(False).astype(bool)
    rows = []
    for variant in ("A_as_released", "B_flagged_halved", "C_flagged_excluded"):
        w = v.copy()
        if variant.startswith("B"):
            # logged calls are double-counted too (two spans per API response), so halve them as well
            w.loc[flag, ["tokens_per_task", "reported_cost", "calls_per_task"]] /= 2
        if variant.startswith("C"):
            w = w[~flag]
        for sc, g in w.groupby("scaffold"):
            for m in ("mean_T", "ARR", "SBF", "TSR", "calls_per_task"):
                if g[m].isna().all():
                    continue
                for y in ("tokens_per_task", "reported_cost"):
                    rows.append({"variant": variant, "scaffold": sc, "metric": m, "outcome": y, "K": len(g),
                                 "spearman": S.spearman(g[m], g[y]) if len(g) >= 3 else float("nan")})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rq2_hal_config_flags.csv", index=False)
    return out


def main():
    a = pd.read_csv(RES / "hal_span_audit.csv") if "--reuse" in sys.argv else audit()
    pd.set_option("display.width", 200)
    v = pd.read_csv(RES / "rq2_hal_config_values.csv", index_col="config")
    a["scaffold"] = a.config.map(v.scaffold)
    print(a.groupby("scaffold").flagged_double.agg(["sum", "size"]))
    out = sensitivity(a)
    print(out[out.outcome == "tokens_per_task"].pivot_table(index=["scaffold", "metric"], columns="variant",
                                                             values="spearman").round(2))
    print(out[out.outcome == "tokens_per_task"].groupby(["variant", "scaffold"]).K.first())


if __name__ == "__main__":
    main()
