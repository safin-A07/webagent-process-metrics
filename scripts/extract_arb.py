"""Parse the 1,302 raw AgentRewardBench trajectories (30 GB) once, in parallel, into a compact cache.

Per-file cache: data/processed/arb_cache/<benchmark>/<model>/<exp>/<task>.json (re-runs skip cached files)
Combined:       data/processed/arb_compact.jsonl.gz
Usage: python -I scripts/extract_arb.py [n_processes]
"""
import gzip
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import loaders as L  # noqa: E402

SRC = L.RAW / "agent_reward_bench" / "cleaned"
CACHE = L.PROC / "arb_cache"


def one(p):
    out = CACHE / p.relative_to(SRC)
    if out.exists():
        return str(out)
    try:
        c = L.arb_compact(p)
    except Exception as e:  # unparseable: recorded, counted as exclusion
        c = {"parse_error": f"{type(e).__name__}: {e}", "path": str(p.relative_to(SRC))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(c), encoding="utf-8")
    return str(out)


def main(n):
    files = sorted(SRC.rglob("*.json"))
    done = 0
    with Pool(n) as pool:
        for _ in pool.imap_unordered(one, files, chunksize=2):
            done += 1
            if done % 100 == 0:
                print(f"{done}/{len(files)}", flush=True)
    with gzip.open(L.PROC / "arb_compact.jsonl.gz", "wt", encoding="utf-8") as f:
        for p in files:
            f.write((CACHE / p.relative_to(SRC)).read_text(encoding="utf-8") + "\n")
    print(f"done: {len(files)} trajectories")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
