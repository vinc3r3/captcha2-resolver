"""Consolidate all RuCaptcha runs into one CSV (no API requests).

Rules (same as the original `make_train_txt.py`):
  * For an image solved in several runs, the latest ready result wins.
  * Answers containing Cyrillic characters are dropped (workers on a Russian layout).
  * Latin letters are lowercased (the captcha is case-insensitive / lowercase only).
  * Answers with whitespace are dropped.

Output columns: filename, rucaptcha_raw, rucaptcha_label (empty when dropped), drop_reason.
"""

from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path


def has_cyrillic(text: str) -> bool:
    return any("CYRILLIC" in unicodedata.name(c, "") for c in text)


def lower_latin(text: str) -> str:
    return "".join(c.lower() if "LATIN" in unicodedata.name(c, "") else c for c in text)


def normalize(text: str) -> tuple[str, str]:
    """-> (label, drop_reason). label is '' when dropped."""
    if has_cyrillic(text):
        return "", "cyrillic"
    label = lower_latin(text.strip())
    if not label or any(c.isspace() for c in label):
        return "", "invalid"
    return label, ""


def consolidate(runs_dir: Path, out: Path) -> dict[str, int]:
    latest: dict[str, str] = {}
    for results in sorted(runs_dir.glob("*/results.json")):  # run ids sort chronologically
        for r in json.loads(results.read_text(encoding="utf-8")):
            if r.get("status") == "ready" and r.get("text"):
                # legacy runs stored absolute "image_path"; current ones store "filename"
                name = r.get("filename") or Path(r["image_path"]).name
                latest[name] = r["text"]
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {"ready": len(latest), "cyrillic": 0, "invalid": 0}
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "rucaptcha_raw", "rucaptcha_label", "drop_reason"])
        for name in sorted(latest):
            label, reason = normalize(latest[name])
            if reason:
                stats[reason] += 1
            w.writerow([name, latest[name], label, reason])
    return stats
