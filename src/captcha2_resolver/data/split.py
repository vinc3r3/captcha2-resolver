"""Train/validation split.

`data/splits.csv` is committed and is the single source of truth, so every model is evaluated on
the same images. Existing assignments are never changed. New labeled images get a split from a hash
of their filename: deterministic, independent of row order, and stable as the dataset grows
(unlike `df.sample(random_state=...)`, which reshuffles everything when one row is added).
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

SPLITS = ("train", "validation")


def hash_split(filename: str, val_fraction: float = 0.1) -> str:
    bucket = int(hashlib.sha1(filename.encode()).hexdigest(), 16) % 10_000
    return "validation" if bucket < val_fraction * 10_000 else "train"


def load(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["filename"]: r["split"] for r in csv.DictReader(f)}


def save(path: Path, splits: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split"])
        for name in sorted(splits):
            w.writerow([name, splits[name]])


def assign(labeled: list[str], existing: dict[str, str], val_fraction: float = 0.1) -> tuple[dict, int]:
    """Keep existing assignments, hash-assign the rest. Returns (splits, n_new)."""
    out = dict(existing)
    new = [n for n in labeled if n not in out]
    for n in new:
        out[n] = hash_split(n, val_fraction)
    return out, len(new)
