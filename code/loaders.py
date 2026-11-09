"""Loaders: convert HAL (Browser-Use, SeeAct) and AgentRewardBench logs to one common trajectory format.

Common format (one dict per trajectory):
  source, scaffold, config, model, family, task_id, website, level, L_H, success, truncated,
  n_noop                number of model steps without an executed environment action (excluded from T)
  actions               list of (type, target, value) tuples: the environment actions a_1..a_T
  states_url            canonical URLs; ARB: s_0..s_T (exact, one per action);
                        HAL Browser-Use: one per *model step* (coarser than actions, see below); SeeAct: None
  states_ax             ARB only: (url, accessibility-tree hash) per state
  tokens                ARB only: (n_in, n_out) of the model call that produced each action
  total_tokens          ARB only: all calls of the episode, including the final answer call

Normalisation rules (paper Sec. 5.2 + notes/preregistration.md):
  - T counts environment actions. Terminal answers (Browser-Use `done`, ARB `send_msg_to_user`/`report_infeasible`)
    and model steps without an action (Browser-Use empty action, SeeAct "No Operation") are not counted.
  - Browser-Use logs flatten the actions of all model steps into one list and record one URL per model step, so the
    URL before each individual action cannot be recovered when a model step issues several actions (data audit,
    6 Nov 2026). For HAL Browser-Use the state sequence is therefore the URL sequence at model-step granularity.
  - Browser-Use targets elements by index; target = xpath of the interacted element (available for ~96 % of
    index actions), falling back to the index.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# ---------------------------------------------------------------- URL canonicalisation (Eq. 2)
VOLATILE = re.compile(r"^(utm_.*|sid|session.*|sessionid|phpsessid|jsessionid|gclid|fbclid|msclkid|_ga|_gl|ref|ref_.*|"
                      r"trk|tracking.*|cache.*|_|ts|timestamp|nonce|token|csrf.*)$", re.I)


def canon_url(u: str | None) -> str | None:
    if not u:
        return None
    try:
        p = urlsplit(u.strip())
    except ValueError:
        return u
    q = sorted((k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not VOLATILE.match(k))
    path = p.path.rstrip("/") or "/"
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), path, urlencode(q), ""))


# ---------------------------------------------------------------- action types (for sensitivity analyses)
SCROLL_WAIT = {"scroll", "scroll_down", "scroll_up", "scroll_to_text", "wait", "noop", "extract_content",
               "get_dropdown_options", "save_pdf", "hover"}

# ---------------------------------------------------------------- HAL


def hal_trajectories():
    """Yield common-format trajectories from data/processed/hal_trajectories.jsonl.gz (scripts/extract_hal.py)."""
    with gzip.open(PROC / "hal_trajectories.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            base = {k: r[k] for k in ("scaffold", "config", "model", "family", "task_id", "website", "level")}
            base.update(source="hal", task_text=r.get("task_text"), L_H=r["reference_length"], success=int(r["predicted_label"] == 1),
                        n_llm_calls=r["n_llm_calls"], hal_error=r.get("error"))
            if r["scaffold"] == "browser-use":
                acts, noop = [], 0
                for a in r["actions"]:
                    name = a["name"]
                    if name is None:
                        noop += 1
                        continue
                    if name == "done":
                        continue
                    p = a["params"] if isinstance(a["params"], dict) else {}
                    target = a["xpath"] or (f"idx:{p['index']}" if "index" in p else None)
                    value = None
                    for k in ("text", "url", "query", "keys", "option", "goal", "page_id", "amount"):
                        if k in p and p[k] is not None:
                            value = str(p[k]).strip().lower()
                            break
                    typ = {"click_element_by_index": "click_element"}.get(name, name)
                    acts.append((typ, target, value))
                urls = [canon_url(u) for u in r["urls"]]
                # step limit: 30 model steps (config max_steps); flagged when the run used all of them without `done`
                done = any(a["name"] == "done" for a in r["actions"])
                base.update(actions=acts, states_url=urls, n_noop=noop,
                            truncated=(not done) and len(r["urls"]) >= 30)
            else:
                acts, noop = [], 0
                for s in r["actions"]:
                    if not s or s.strip() == "No Operation":
                        noop += 1
                        continue
                    if s.startswith("Failed to "):
                        body = s[len("Failed to "):].split(" because ")[0]
                        typ = body.split(" ")[0]
                        acts.append(("failed_" + typ.lower(), body, None))
                    else:
                        left, _, right = s.rpartition(" -> ")
                        typ, _, val = right.partition(":")
                        acts.append((typ.strip().lower(), left.strip(), val.strip().lower() or None))
                ex = (r.get("exit_by") or "")
                base.update(actions=acts, states_url=None, n_noop=noop, exit_by=ex,
                            truncated="step limit" in ex)
            yield base


# ---------------------------------------------------------------- AgentRewardBench
AX_LINE = re.compile(r"^\s*\[([^\]]+)\]\s+(\S+)\s*'([^']*)'", re.M)
FUNC = re.compile(r"^\s*(\w+)\((.*)\)\s*$", re.S)
TERMINAL = {"send_msg_to_user", "report_infeasible"}


def ax_index(axtree: str | None) -> dict:
    return {m.group(1): (m.group(2), m.group(3)) for m in AX_LINE.finditer(axtree or "")}


def ax_hash(axtree_pruned: str | None) -> str:
    """h(ax(s)): accessibility tree with element ids, StaticText content and digits removed (dynamic text)."""
    t = re.sub(r"\[[^\]]+\]\s*", "", axtree_pruned or "")
    t = re.sub(r"StaticText '[^']*'", "StaticText", t)
    t = re.sub(r"\d+", "#", t)
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]


def parse_bg_action(a: str):
    m = FUNC.match(a or "")
    if not m:
        return None, []
    name, args = m.group(1), m.group(2)
    try:
        vals = list(ast.literal_eval("(" + args + ",)")) if args.strip() else []
    except (ValueError, SyntaxError):
        vals = [args]
    return name, vals


def arb_labels():
    """Expert labels per trajectory (majority vote over annotators; ties -> None). Key: (benchmark, model, exp, task)."""
    import csv
    votes = {}
    with open(RAW / "agent_reward_bench" / "data" / "annotations.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = (row["benchmark"], row["model_name"], row["exp_name"], row["task_id"])
            votes.setdefault(k, []).append(row)

    def maj(rows, col, pos):
        c = Counter(r[col] for r in rows)
        yes, no = c.get(pos, 0), sum(v for kk, v in c.items() if kk != pos and kk not in ("", "Unsure"))
        if yes == no:
            return None
        return int(yes > no)

    out = {}
    for k, rows in votes.items():
        out[k] = {"n_annot": len(rows), "loop": maj(rows, "trajectory_looping", "Yes"),
                  "success": maj(rows, "trajectory_success", "Successful"),
                  "side_effect": maj(rows, "trajectory_side_effect", "Yes"),
                  "optimality": [r["trajectory_optimality"] for r in rows]}
    return out


def arb_compact(path: Path) -> dict:
    """Parse one raw AgentRewardBench trajectory (~23 MB) into the few fields the metrics need.
    Element targets are resolved to (role, name) here, while the step's accessibility tree is in memory."""
    try:
        import orjson
        d = orjson.loads(path.read_bytes())
    except ImportError:
        d = json.loads(path.read_text(encoding="utf-8"))
    rel = path.relative_to(RAW / "agent_reward_bench" / "cleaned").parts  # benchmark/model/exp/task.json
    steps = []
    for st in d.get("steps") or []:
        stats = st.get("stats") or {}
        name, vals = parse_bg_action(st.get("action"))
        role = nm = None
        if name and vals and name in BID_ACTIONS:
            role, nm = ax_index(st.get("axtree")).get(str(vals[0]), (None, None))
        steps.append({"url": st.get("url"), "action": st.get("action"), "role": role, "name": nm,
                      "axh": ax_hash(st.get("axtree_pruned")), "n_in": stats.get("input_tokens") or 0,
                      "n_out": stats.get("output_tokens") or 0})
    si = d.get("summary_info") or {}
    return {"benchmark": rel[0], "model": rel[1], "exp": rel[2], "task": rel[3][:-5],
            "truncated": bool(si.get("truncated")), "cum_cost": si.get("stats.cum_cost"), "steps": steps}


BID_ACTIONS = {"click", "dblclick", "hover", "fill", "select_option", "press", "focus", "clear", "check", "uncheck"}


def arb_trajectory(c: dict, labels: dict):
    """Common-format trajectory from a compact AgentRewardBench record (see arb_compact)."""
    bench, model, exp, task = c["benchmark"], c["model"], c["exp"], c["task"]
    lab = labels.get((bench, model, exp, task))
    steps = c["steps"]
    acts, s_url, s_ax, toks, noop = [], [], [], [], 0
    tot_in = tot_out = 0
    for i, st in enumerate(steps):
        n_in, n_out = st["n_in"], st["n_out"]
        tot_in += n_in
        tot_out += n_out
        name, vals = parse_bg_action(st["action"])
        if name is None or name in TERMINAL:
            continue
        typ, target, value = name, None, None
        if name in BID_ACTIONS:
            target = f"{st['role']}|{st['name']}" if st["role"] else (f"bid:{vals[0]}" if vals else None)
            if len(vals) > 1:
                value = str(vals[1]).strip().lower()
        elif name in ("goto", "tab_focus", "keyboard_press", "keyboard_type", "scroll"):
            value = ",".join(str(v) for v in vals).lower()
        nxt = steps[i + 1] if i + 1 < len(steps) else st
        if not s_url:
            s_url.append(canon_url(st["url"]))
            s_ax.append((s_url[0], st["axh"]))
        acts.append((typ, target, value))
        u = canon_url(nxt["url"])
        s_url.append(u)
        s_ax.append((u, nxt["axh"]))
        toks.append((n_in, n_out))
    return {
        "source": "arb", "scaffold": "browsergym", "config": model, "model": model, "family": model,
        "benchmark": bench, "exp": exp, "task_id": task, "website": bench, "level": None, "L_H": None,
        "success": None if lab is None else lab["success"], "loop": None if lab is None else lab["loop"],
        "has_label": lab is not None, "n_annot": None if lab is None else lab["n_annot"],
        "optimality": None if lab is None else lab["optimality"],
        "truncated": c["truncated"], "n_noop": noop, "actions": acts, "states_url": s_url,
        "states_ax": s_ax, "tokens": toks, "total_tokens": (tot_in, tot_out),
        "cum_cost_reported": c["cum_cost"],
    }


def arb_trajectories():
    """Read the compact cache written by scripts/extract_arb.py."""
    labels = arb_labels()
    with gzip.open(PROC / "arb_compact.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if "parse_error" in c:
                yield {"source": "arb", "parse_error": c["parse_error"], "path": c["path"]}
                continue
            yield arb_trajectory(c, labels)


def arb_judges():
    """LLM-judge <loop> answers. Returns {(benchmark, model, task): {judge: 0/1/None}}."""
    out = {}
    base = RAW / "agent_reward_bench" / "judgments"
    for p in base.rglob("*.json"):
        rel = p.relative_to(base).parts  # benchmark/model/judge/task.json
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            txt = d["response"]["choices"][0]["message"]["content"]
        except Exception:
            txt = ""
        m = re.search(r"<loop>\s*(Yes|No)\s*</loop>", txt or "", re.I)
        out.setdefault((rel[0], rel[1], rel[3][:-5]), {})[rel[2]] = None if not m else int(m.group(1).lower() == "yes")
    return out
