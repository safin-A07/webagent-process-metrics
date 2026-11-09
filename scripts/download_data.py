"""Download the three public data sources into data/raw/.

Usage:  python -I scripts/download_data.py [om2w] [hal] [arb_meta] [arb_traj] [arb_judges]
        (no argument = all four, in this order)

- om2w      osunlp/Online-Mind2Web task file (gated: needs `hf auth login` by the user)
- hal       HAL traces for Online-Mind2Web: browser-use_* and seeact_* zips from agent-evals/hal_traces.
            seeact_gpt-5_reasoning_effort_high_UPLOAD_updated.zip (12.8 GB) is skipped unless --with-updated.
- arb_meta  McGill-NLP/agent-reward-bench README.md + data/ (expert annotations)
- arb_traj  McGill-NLP/agent-reward-bench cleaned/ trajectories (~30 GB); screenshots/ are not downloaded
- arb_judges  judgments/ of 4 LLM judges (ARB_JUDGES, ~1 GB), for the RQ3 judge comparison

Every downloaded file is logged with its size to data/raw/download_log.csv.
"""
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "data" / ".hf_cache"))

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

HAL_REPO = "agent-evals/hal_traces"
OM2W_REPO = "osunlp/Online-Mind2Web"
ARB_REPO = "McGill-NLP/agent-reward-bench"
HAL_UPDATED = "seeact_gpt-5_reasoning_effort_high_UPLOAD_updated.zip"
# Four LLM judges released by AgentRewardBench, one per judge model family (accessibility tree, no screenshot
# where available). Chosen before looking at their accuracy.
ARB_JUDGES = ["gpt-4o-noscreen", "claude-3.7-sonnet-noscreen", "llama-3.3-70b-noscreen", "qwen-2.5-vl-noscreen"]

api = HfApi()
LOG = RAW / "download_log.csv"


def log(repo, fname, path):
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "repo", "file", "bytes"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), repo, fname, Path(path).stat().st_size])


def fetch(repo, files, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for i, fname in enumerate(files, 1):
        target = dest / fname
        if target.exists():
            print(f"[{i}/{len(files)}] skip (exists) {fname}", flush=True)
            continue
        for attempt in range(1, 6):
            try:
                path = hf_hub_download(repo, fname, repo_type="dataset", local_dir=dest)
                log(repo, fname, path)
                print(f"[{i}/{len(files)}] ok {fname}", flush=True)
                break
            except Exception as e:  # network errors: retry, then report and continue
                print(f"[{i}/{len(files)}] attempt {attempt} failed {fname}: {type(e).__name__}: {e}", flush=True)
                # HTTP 429 = Hugging Face rate limit for anonymous IPs: wait 5 minutes before retrying
                time.sleep(300 if "429" in str(e) else 10 * attempt)
        else:
            print(f"FAILED {fname}", flush=True)


def fetch_parallel(repo, files, dest, workers=3):
    """Like fetch(), with a few threads; each thread backs off 5 minutes on HTTP 429."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [f for f in files if not (dest / f).exists()]
    print(f"{len(files) - len(todo)} already present, {len(todo)} to download", flush=True)

    def one(fname):
        for attempt in range(1, 6):
            try:
                path = hf_hub_download(repo, fname, repo_type="dataset", local_dir=dest)
                log(repo, fname, path)
                return True
            except Exception as e:
                time.sleep(300 if "429" in str(e) else 10 * attempt)
        print(f"FAILED {fname}", flush=True)
        return False

    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 100 == 0:
                print(f"[{i}/{len(todo)}]", flush=True)


def repo_files(repo):
    return api.list_repo_files(repo, repo_type="dataset")


def main(args):
    steps = [a for a in args if not a.startswith("--")] or ["om2w", "hal", "arb_meta", "arb_traj"]
    if "om2w" in steps:
        fetch(OM2W_REPO, [f for f in repo_files(OM2W_REPO) if f.endswith(".json") or f == "README.md"],
              RAW / "online_mind2web")
    if "hal" in steps:
        files = [f for f in repo_files(HAL_REPO) if f.startswith(("browser-use_", "seeact_"))]
        if "--with-updated" not in args:
            files = [f for f in files if f != HAL_UPDATED]
        fetch(HAL_REPO, sorted(files), RAW / "hal_zips")
    if "arb_meta" in steps:
        fetch(ARB_REPO, [f for f in repo_files(ARB_REPO) if f == "README.md" or f.startswith("data/")],
              RAW / "agent_reward_bench")
    if "arb_traj" in steps:
        fetch(ARB_REPO, sorted(f for f in repo_files(ARB_REPO) if f.startswith("cleaned/")),
              RAW / "agent_reward_bench")
    if "arb_judges" in steps:
        # LLM-judge outputs that answer the same four questions as the experts (incl. <loop>), RQ3 comparison.
        # 3 threads: 8 parallel anonymous connections triggered HTTP 429 on 6 Nov 2026, 1 thread was too slow.
        fetch_parallel(ARB_REPO, sorted(f for f in repo_files(ARB_REPO)
                                        if f.startswith("judgments/") and f.split("/")[3] in ARB_JUDGES),
                       RAW / "agent_reward_bench")


if __name__ == "__main__":
    main(sys.argv[1:])
