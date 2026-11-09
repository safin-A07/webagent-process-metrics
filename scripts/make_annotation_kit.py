"""Annotation kit for the human efficiency study.

Samples 120 pairs of SUCCESSFUL HAL Browser-Use runs on the same task by two different configurations, shows
both action sequences side by side (with URLs), and asks two or more annotators, independently:
  Q1 Which run reaches the goal more efficiently?  A / B / about equal
  Q2 How much unnecessary or repeated activity does run A contain?  1 (none) .. 5 (a lot)
  Q3 Same for run B.
The metric values are NOT shown, so the judgement is blind to SER/ARR/SBF. Order A/B is randomised.

Usage:  python -I scripts/make_annotation_kit.py           -> results/annotation_kit/{kit.html, sheet_<name>.csv, key.csv}
        python -I scripts/make_annotation_kit.py score     -> Cohen's kappa between annotators and agreement with SER
"""
import csv
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import loaders as L  # noqa: E402

OUT = ROOT / "results" / "annotation_kit"
N_PAIRS = 120
ANNOTATORS = ["annotator1", "annotator2"]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    trajs = {(t["config"], t["task_id"]): t for t in L.hal_trajectories()
             if t["scaffold"] == "browser-use" and t["success"] == 1 and t["actions"]}
    by_task = {}
    for (c, q) in trajs:
        by_task.setdefault(q, []).append(c)
    r = np.random.default_rng(42)
    tasks = sorted(q for q, cs in by_task.items() if len(cs) >= 2)
    pick = r.choice(tasks, size=min(N_PAIRS, len(tasks)), replace=False)
    key, blocks = [], []
    for i, q in enumerate(pick, 1):
        a, b = r.choice(sorted(by_task[q]), size=2, replace=False)
        ta, tb = trajs[(a, q)], trajs[(b, q)]
        key.append({"pair": i, "task_id": q, "A": a, "B": b, "T_A": len(ta["actions"]), "T_B": len(tb["actions"]),
                    "L_H": ta["L_H"]})

        def fmt(t):
            rows = [f"<li>{html.escape(str(x[0]))} {html.escape(str(x[1] or ''))[:90]} "
                    f"{html.escape(str(x[2] or ''))[:60]}</li>" for x in t["actions"]]
            urls = "<br>".join(html.escape(u or "") for u in dict.fromkeys(t["states_url"] or []))
            return f"<ol>{''.join(rows)}</ol><details><summary>pages visited</summary><small>{urls}</small></details>"
        blocks.append(f"<h3>Pair {i}</h3><p><b>Task:</b> {html.escape(ta.get('task_text') or q)}</p>"
                      f"<table><tr><th>Run A</th><th>Run B</th></tr><tr><td>{fmt(ta)}</td><td>{fmt(tb)}</td></tr></table>")
    (OUT / "kit.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Efficiency annotation</title>"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px}td{vertical-align:top;"
        "width:50%;border:1px solid #ccc;padding:6px}table{border-collapse:collapse;width:100%}</style>"
        "<h1>Efficiency annotation</h1><p>Both runs completed the task. For each pair answer in your own sheet: "
        "Q1 which run reaches the goal more efficiently (A, B or equal); Q2/Q3 how much unnecessary or repeated "
        "activity run A / run B contains (1 = none, 5 = a lot). Work alone; do not discuss pairs.</p>"
        + "".join(blocks), encoding="utf-8")
    pd.DataFrame(key).to_csv(OUT / "key.csv", index=False)
    for name in ANNOTATORS:
        with open(OUT / f"sheet_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pair", "Q1_more_efficient(A/B/equal)", "Q2_waste_A(1-5)", "Q3_waste_B(1-5)"])
            for i in range(1, len(key) + 1):
                w.writerow([i, "", "", ""])
    print(f"kit with {len(key)} pairs written to {OUT}")


def score():
    from sklearn.metrics import cohen_kappa_score
    key = pd.read_csv(OUT / "key.csv")
    sheets = {n: pd.read_csv(OUT / f"sheet_{n}.csv") for n in ANNOTATORS}
    q1 = {n: s.iloc[:, 1].astype(str).str.strip().str.upper() for n, s in sheets.items()}
    if any((q == "NAN").all() for q in q1.values()):
        print("sheets are not filled in yet")
        return
    a, b = q1[ANNOTATORS[0]], q1[ANNOTATORS[1]]
    ok = (a != "NAN") & (b != "NAN")
    print("Cohen's kappa Q1:", cohen_kappa_score(a[ok], b[ok]))
    agree = a[ok] == b[ok]
    ser_pref = np.where(key.T_A < key.T_B, "A", np.where(key.T_A > key.T_B, "B", "EQUAL"))
    print("agreement of the consensus with the shorter run (higher SER):",
          float((a[ok & agree].values == ser_pref[(ok & agree).values]).mean()), "on", int((ok & agree).sum()), "pairs")


if __name__ == "__main__":
    score() if sys.argv[1:] == ["score"] else build()
