"""Decrypt HAL trace archives (data/raw/hal_zips/*.zip -> data/raw/hal/<zip stem>/).

Faithful copy of HAL's official decryption (hal-harness, hal/utils/decrypt.py and
hal/utils/json_encryption.py, commit 002a7495a6906d4ca8f6105f95862fcf896f2fec, 9 Apr 2026):
each zip member is JSON {"encrypted_data", "salt"}; key = PBKDF2-HMAC-SHA256(HAL password (set HAL_DECRYPT_PASSWORD; see hal-harness),
salt, 480000 iterations, 32 bytes) used as a Fernet key. Differences from the official CLI, which
change only where and how files are written, not their content:
  - output goes to a per-zip folder instead of next to the zip, with a path-traversal guard;
  - decrypted bytes are written as-is instead of re-serialised with json.dump(indent=2).

Usage: python -I scripts/hal_decrypt.py [zip ...]   (no argument = every zip in data/raw/hal_zips)
"""
import base64
import os
import json
import sys
from pathlib import Path
from zipfile import ZipFile

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ROOT = Path(__file__).resolve().parents[1]
ZIPS = ROOT / "data" / "raw" / "hal_zips"
OUT = ROOT / "data" / "raw" / "hal"
PASSWORD = os.environ["HAL_DECRYPT_PASSWORD"].encode()  # published in hal-harness, hal/utils/decrypt.py


def fernet(salt_b64: str) -> Fernet:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                     salt=base64.b64decode(salt_b64.encode("utf-8")), iterations=480000)
    return Fernet(base64.urlsafe_b64encode(kdf.derive(PASSWORD)))


def decrypt_zip(zip_path: Path) -> list:
    dest = OUT / zip_path.stem
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    with ZipFile(zip_path) as zf:
        for name in zf.namelist():
            out = (dest / name.replace(".encrypted", "")).resolve()
            if dest.resolve() not in out.parents:
                raise ValueError(f"unsafe member path {name!r} in {zip_path.name}")
            if out.exists():
                written.append(out)
                continue
            with zf.open(name) as f:
                blob = json.load(f)
            data = fernet(blob["salt"]).decrypt(base64.b64decode(blob["encrypted_data"].encode("utf-8")))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            written.append(out)
    return written


def main(args):
    zips = [Path(a) for a in args] or sorted(ZIPS.glob("*.zip"))
    ok = failed = 0
    for z in zips:
        try:
            files = decrypt_zip(z)
            ok += 1
            print(f"ok   {z.name}: {len(files)} file(s), {sum(f.stat().st_size for f in files) / 1e6:.1f} MB", flush=True)
        except Exception as e:
            failed += 1
            print(f"FAIL {z.name}: {type(e).__name__}: {e}", flush=True)
    print(f"done: {ok} decrypted, {failed} failed")


if __name__ == "__main__":
    main(sys.argv[1:])
