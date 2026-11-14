"""Build data/labels.csv — the single training source of truth — from immutable inputs:

  data/labels/rucaptcha.csv   paid human labels (consolidated runs)          [raw, never edited]
  data/labels/llm.csv         labels annotated by Claude (2nd opinion, weak) [raw, never edited]
  data/corrections.csv        every fix, one row each: filename,before,after,stage,note
  data/splits.csv             frozen split assignment

Corrections are applied in file order and must chain: a row's `before` has to equal the current
label, otherwise the merge fails ("stale correction") — so a re-labeled source can never be
silently overwritten by an old fix. `after` empty = drop the image (unreadable).

Images without a RuCaptcha label are kept with split "unlabeled" (they still have Claude labels and
are useful for semi-supervised work, but they are *not* verified).
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from captcha2_resolver.data import split as split_mod

FIELDS = ["filename", "label", "split", "rucaptcha_raw", "rucaptcha_label",
          "llm_label", "llm_confidence", "corrected", "correction_stage"]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge(images_dir: Path, rucaptcha: Path, llm: Path, corrections: Path, splits: Path,
          out: Path, val_fraction: float = 0.1) -> dict:
    names = sorted(p.name for p in images_dir.glob("*.jpg"))
    ruc = {r["filename"]: r for r in _read(rucaptcha)}
    llm_rows = {r["filename"]: r for r in _read(llm)}

    label = {n: ruc[n]["rucaptcha_label"] for n in names if n in ruc and ruc[n]["rucaptcha_label"]}
    stage: dict[str, str] = {}
    for i, c in enumerate(_read(corrections), start=2):
        n = c["filename"]
        if n not in label:
            raise ValueError(f"corrections.csv:{i}: {n} has no label to correct")
        if label[n] != c["before"]:
            raise ValueError(f"corrections.csv:{i}: stale correction for {n}: "
                             f"expected {c['before']!r}, current label is {label[n]!r}")
        label[n], stage[n] = c["after"], c["stage"]
    label = {n: v for n, v in label.items() if v}  # after == "" -> dropped

    frozen = split_mod.load(splits)
    assigned, n_new = split_mod.assign(sorted(label), frozen, val_fraction)
    if n_new:
        split_mod.save(splits, assigned)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for n in names:
            r, m = ruc.get(n, {}), llm_rows.get(n, {})
            w.writerow({
                "filename": n,
                "label": label.get(n, ""),
                "split": assigned[n] if n in label else "unlabeled",
                "rucaptcha_raw": r.get("rucaptcha_raw", ""),
                "rucaptcha_label": r.get("rucaptcha_label", ""),
                "llm_label": m.get("llm_label", ""),
                "llm_confidence": m.get("llm_confidence", ""),
                "corrected": str(n in stage).lower(),
                "correction_stage": stage.get(n, ""),
            })
    counts = Counter(assigned[n] if n in label else "unlabeled" for n in names)
    return {"images": len(names), "labeled": len(label), "corrected": len(stage),
            "new_split_assignments": n_new, **counts}


def load_labels(path: Path, split: str | None = None) -> list[dict]:
    if not path.exists():  # unlike the optional merge inputs, an absent labels file is a mistake
        raise FileNotFoundError(f"{path} not found: run `c2r data merge` (or `c2r hub pull-dataset`) first")
    rows = [r for r in _read(path) if r["label"]]
    return [r for r in rows if split is None or r["split"] == split]
