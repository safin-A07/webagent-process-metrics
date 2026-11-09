"""SHA-256 checksums of every input file (HAL zips, ARB annotations) -> results/tables/checksums.csv.
ARB trajectory files are summarised by one checksum over the sorted list of (path, sha256) to keep the table short.
Usage: python -I scripts/checksums.py
"""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def sha(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


rows = [("hal_zip", p.name, p.stat().st_size, sha(p)) for p in sorted((RAW / "hal_zips").glob("*.zip"))]
rows += [("arb_data", p.name, p.stat().st_size, sha(p)) for p in sorted((RAW / "agent_reward_bench" / "data").glob("*"))]
agg = hashlib.sha256()
n = 0
for p in sorted((RAW / "agent_reward_bench" / "cleaned").rglob("*.json")):
    agg.update(f"{p.relative_to(RAW).as_posix()}:{sha(p)}\n".encode())
    n += 1
rows.append(("arb_cleaned_all", f"{n} files", "", agg.hexdigest()))
with open(ROOT / "results" / "tables" / "checksums.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["kind", "file", "bytes", "sha256"])
    w.writerows(rows)
print(len(rows), "rows")
