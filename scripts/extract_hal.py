"""Extract the fields needed for the analysis from the decrypted HAL traces (one pass over ~3.5 GB of JSON).

Input : data/raw/hal/<run>/<run>.json   (decrypted by scripts/hal_decrypt.py)
Output: data/processed/hal_trajectories.jsonl.gz   one line per (configuration, task)
        data/processed/hal_configs.csv              one line per configuration (run-level usage, cost, accuracy)

Nothing is computed here except counting; all raw strings needed later are kept verbatim.
Usage: python -I scripts/extract_hal.py
"""
import csv
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAL = ROOT / "data" / "raw" / "hal"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

EL_RE = re.compile(r"tag_name='([^']*)', xpath='([^']*)'")


def model_of(run: str) -> str:
    return run.split("_", 1)[1].removesuffix("_UPLOAD")


def family_of(model: str) -> str:
    """Base model, ignoring reasoning/thinking settings (pre-registered definition, RQ4)."""
    m = re.sub(r"_thinking_high_4096$", "", model)
    m = re.sub(r"_reasoning_effort_(high|medium|minimal|low)$", "", m)
    return m


def bu_actions(rec):
    out = []
    for a in rec.get("model_actions") or []:
        a = dict(a or {})
        el = a.pop("interacted_element", None)
        name = next(iter(a), None)
        params = a.get(name) if name else None
        tag = xpath = None
        if isinstance(el, str):
            m = EL_RE.search(el)
            if m:
                tag, xpath = m.group(1), m.group(2)
        out.append({"name": name, "params": params, "tag": tag, "xpath": xpath})
    return out


def main():
    runs = sorted(p for p in HAL.iterdir() if p.is_dir())
    cfg_rows = []
    with gzip.open(OUT / "hal_trajectories.jsonl.gz", "wt", encoding="utf-8") as fo:
        for run in runs:
            files = list(run.glob("*.json"))
            if len(files) != 1:
                print(f"SKIP {run.name}: {len(files)} json files", flush=True)
                continue
            d = json.loads(files[0].read_text(encoding="utf-8"))
            scaffold = "browser-use" if run.name.startswith("browser-use") else "seeact"
            model = model_of(run.name)
            calls = Counter()
            first, last = {}, {}
            for c in d.get("raw_logging_results") or []:
                t = c.get("weave_task_id") or (c.get("attributes") or {}).get("weave_task_id")
                if not t:
                    continue
                calls[t] += 1
                s, e = c.get("started_at"), c.get("ended_at")
                if s and (t not in first or s < first[t]):
                    first[t] = s
                if e and (t not in last or e > last[t]):
                    last[t] = e
            res = d.get("results") or {}
            succ = set(res.get("successful_tasks") or [])
            n_traj = 0
            for tid, r in (d.get("raw_eval_results") or {}).items():
                row = {
                    "config": run.name.removesuffix("_UPLOAD"), "scaffold": scaffold, "model": model,
                    "family": family_of(model), "task_id": r.get("task_id", tid), "website": r.get("website"),
                    "task_text": r.get("confirmed_task"),
                    "level": r.get("level"), "reference_length": r.get("reference_length"),
                    "predicted_label": r.get("predicted_label"), "in_successful_tasks": tid in succ,
                    "error": r.get("error"), "n_llm_calls": calls.get(tid, 0),
                    "t_start": first.get(tid), "t_end": last.get(tid),
                }
                if scaffold == "browser-use":
                    row.update({
                        "urls": r.get("urls") or [], "actions": bu_actions(r),
                        "action_names": r.get("action_names") or [],
                        "n_action_results": len(r.get("action_results") or []),
                        "action_errors": [("error=None" not in str(x)) for x in (r.get("action_results") or [])],
                        "n_model_steps": len(r.get("model_thoughts") or []),
                        "step_errors": [e is not None for e in (r.get("errors") or [])],
                    })
                    has = bool(row["actions"])
                else:
                    row.update({
                        "actions": [x.get("actions") for x in (r.get("take_actions_and_elements") or [])],
                        "elements": [x.get("elements") for x in (r.get("take_actions_and_elements") or [])],
                        "num_step": r.get("num_step"), "exit_by": r.get("exit_by"),
                    })
                    has = bool(row["actions"])
                n_traj += has
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            usage = d.get("total_usage") or {}
            cfg_rows.append({
                "config": run.name.removesuffix("_UPLOAD"), "scaffold": scaffold, "model": model,
                "family": family_of(model), "date": (d.get("config") or {}).get("date"),
                "max_steps": ((d.get("config") or {}).get("agent_args") or {}).get("max_steps"),
                "n_tasks": len(d.get("raw_eval_results") or {}), "n_with_actions": n_traj,
                "accuracy": res.get("accuracy"), "n_successful": len(succ), "total_cost_reported": res.get("total_cost"),
                "usage_json": json.dumps(usage), "n_logged_calls": sum(calls.values()),
            })
            print(f"ok {run.name}: {cfg_rows[-1]['n_tasks']} tasks, {n_traj} with actions", flush=True)
            del d
    with open(OUT / "hal_configs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cfg_rows[0]))
        w.writeheader()
        w.writerows(cfg_rows)
    print(f"done: {len(cfg_rows)} configurations")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
