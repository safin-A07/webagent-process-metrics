"""Compute per-trajectory process metrics and baselines (step 4 of the pipeline in README.md).

Usage: python -I scripts/compute_metrics.py [hal] [arb]
Output: results/per_trajectory_hal.csv, results/per_trajectory_arb.csv, results/tables/exclusions.csv
"""
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import loaders as L  # noqa: E402
import procmetrics as M  # noqa: E402

RES = ROOT / "results"
K_PREFIX = 10  # pre-registered prefix length for the early-warning analysis (U1)


def base_metrics(actions):
    T = len(actions)
    r = M.arr_psi(actions)
    r_ns = M.arr_psi(actions, exclude=L.SCROLL_WAIT)
    return {
        "T": T, "logT": None,
        "ARR_psi": sum(r) / T, "ARR_psi_noscroll": sum(r_ns) / T, "n_redundant_psi": sum(r),
        "B_unique": M.unique_action_ratio(actions), "B_longest_repeat": M.longest_repeat_ratio(actions),
        "B_trigram_loop": M.ngram_loop(actions, 3),
        "share_scroll_wait": sum(a[0] in L.SCROLL_WAIT for a in actions) / T,
    }


def hal_rows(excl):
    for tr in L.hal_trajectories():
        key = {k: tr[k] for k in ("scaffold", "config", "model", "family", "task_id", "website", "level", "L_H",
                                  "success", "truncated", "n_noop", "n_llm_calls")}
        acts = tr["actions"]
        if not acts:
            excl[(tr["scaffold"], "no environment action")] += 1
            yield {**key, "excluded": "no environment action", "T": 0}
            continue
        row = {**key, "excluded": "", **base_metrics(acts)}
        T = row["T"]
        row["SER_H"] = tr["L_H"] / max(T, tr["L_H"]) if tr["L_H"] else None
        row["SER_S_H"] = row["SER_H"] * tr["success"] if row["SER_H"] is not None else None
        pre = acts[:K_PREFIX]
        row["pre_ARR_psi"] = sum(M.arr_psi(pre)) / len(pre)
        if tr["states_url"] is not None:  # Browser-Use: URL sequence at model-step granularity
            urls = [u for u in tr["states_url"]]
            nb = M.backtracks_in_sequence(urls)
            row.update(SBF_url=nb / T, n_backtracks=nb, n_url_states=len(urls), n_distinct_urls=len(set(urls)),
                       pre_SBF_url=M.backtracks_in_sequence(urls[:K_PREFIX + 1]) / len(pre),
                       loop_erased_len=len(M.loop_erase(urls)) - 1)
            row["omega_const_upper"] = min(1.0, (row["n_redundant_psi"] + nb) / T)
        else:
            row.update(exit_by=tr.get("exit_by"))
        yield row


def arb_rows(excl):
    for tr in L.arb_trajectories():
        if "parse_error" in tr:
            excl[("arb", "unparseable log")] += 1
            yield {"source": "arb", "excluded": "unparseable log: " + tr["parse_error"], "path": tr["path"]}
            continue
        key = {k: tr[k] for k in ("config", "benchmark", "exp", "task_id", "success", "loop", "has_label", "n_annot",
                                  "truncated", "n_noop")}
        key["optimality"] = "|".join(tr["optimality"] or [])
        acts = tr["actions"]
        tin, tout = tr["total_tokens"]
        key.update(tokens_in=tin, tokens_out=tout, tokens_total=tin + tout)
        if not acts:
            excl[("arb", "no environment action")] += 1
            yield {**key, "excluded": "no environment action", "T": 0}
            continue
        row = {**key, "excluded": "", **base_metrics(acts)}
        T = row["T"]
        su, sa = tr["states_url"], tr["states_ax"]
        r_u, r_a = M.arr_state(su, acts), M.arr_state(sa, acts)
        r_u_ns = M.arr_state(su, acts, exclude=L.SCROLL_WAIT)
        b_u, b_a = M.backtrack_flags_seq(su), M.backtrack_flags_seq(sa)
        row.update(ARR_url=sum(r_u) / T, ARR_ax=sum(r_a) / T, ARR_url_noscroll=sum(r_u_ns) / T,
                   SBF_url=sum(b_u) / T, SBF_ax=sum(b_a) / T, n_backtracks=sum(b_u),
                   n_redundant_url=sum(r_u))
        # coarse SBF as in HAL Browser-Use is identical here because ARB records one state per action
        # cost attribution with real per-step tokens: c_t = n_in + n_out (model-agnostic token cost)
        c = [a + b for a, b in tr["tokens"]]
        flag = [max(x, y) for x, y in zip(r_u, b_u)]
        tot = tin + tout
        row["omega_tokens"] = (sum(ci for ci, f in zip(c, flag) if f) / tot) if tot else None
        row["omega_tokens_steps_only"] = (sum(ci for ci, f in zip(c, flag) if f) / sum(c)) if sum(c) else None
        row["omega_const"] = sum(flag) / T
        row["step_tokens_in"] = sum(a for a, _ in tr["tokens"])
        row["step_tokens_out"] = sum(b for _, b in tr["tokens"])
        pre = acts[:K_PREFIX]
        row["pre_ARR_url"] = sum(M.arr_state(su[:K_PREFIX + 1], pre)) / len(pre)
        row["pre_SBF_url"] = sum(M.backtrack_flags_seq(su[:K_PREFIX + 1])) / len(pre)
        row["pre_ARR_psi"] = sum(M.arr_psi(pre)) / len(pre)
        row["loop_erased_len"] = len(M.loop_erase(su)) - 1
        row["loop_erased_len_ax"] = len(M.loop_erase(sa)) - 1
        yield row


def write(rows, path):
    rows = list(rows)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main(args):
    which = args or ["hal", "arb"]
    excl = Counter()
    if "hal" in which:
        print("hal rows:", write(hal_rows(excl), RES / "per_trajectory_hal.csv"), flush=True)
    if "arb" in which:
        print("arb rows:", write(arb_rows(excl), RES / "per_trajectory_arb.csv"), flush=True)
    mode = "a" if (RES / "tables" / "exclusions.csv").exists() and len(which) == 1 else "w"
    with open(RES / "tables" / "exclusions.csv", mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["source", "reason", "n"])
        for (s, why), n in sorted(excl.items()):
            w.writerow([s, why, n])
    print(dict(excl))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1:])
