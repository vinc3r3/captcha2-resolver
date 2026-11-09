"""Sanity checks on the final labels (ported from the preprocessing notebook).

errors   -> must be fixed (via data/corrections.csv) before training
warnings -> worth a second look at the image, but often legitimate
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from captcha2_resolver.paths import LABEL_LENGTH

# The 20 symbols that make up >99.8% of all label characters. The captcha avoids look-alikes
# (0/o, 1/l/i, h, k, ...). Rare extra symbols (q, s, z, v) in labels are almost always mistakes.
CORE_ALPHABET = "2345678abcdefgmnpwxy"


def check(rows: list[dict], images_dir: Path | None = None,
          length: int = LABEL_LENGTH, alphabet: str = CORE_ALPHABET) -> tuple[dict, dict]:
    """rows: dicts with 'filename' and 'label'. Returns (errors, warnings): name -> list of rows."""
    names = Counter(r["filename"] for r in rows)
    errors = {
        "empty label": [r for r in rows if not r["label"]],
        f"length != {length}": [r for r in rows if r["label"] and len(r["label"]) != length],
        "duplicate filename": [r for r in rows if names[r["filename"]] > 1],
    }
    if images_dir is not None:
        errors["image file missing"] = [r for r in rows if not (images_dir / r["filename"]).is_file()]
    warnings = {
        "symbols outside core alphabet": [r for r in rows if set(r["label"]) - set(alphabet)],
        "all digits": [r for r in rows if r["label"].isdigit()],
        "3+ same symbols in a row": [r for r in rows if re.search(r"(.)\1\1", r["label"])],
    }
    return errors, warnings


def report(errors: dict, warnings: dict, verbose: bool = True) -> int:
    for title, group in (("ERRORS", errors), ("WARNINGS", warnings)):
        print(f"== {title} ==")
        for name, bad in group.items():
            print(f"{name}: {len(bad)}")
            if verbose:
                for r in bad[:50]:
                    print(f"    {r['filename']}\t{r['label']}")
    return sum(len(v) for v in errors.values())


def char_stats(labels: list[str], alphabet: str = CORE_ALPHABET) -> str:
    counts = Counter("".join(labels))
    return "  ".join(f"{c}{'' if c in alphabet else '*'}:{n}" for c, n in counts.most_common())
