"""One-off: rebuild this repo's data/ layout from the original exploration folders, and prove the
result is identical to the labels the published model was trained on.

    uv run --extra dev python scripts/migrate_legacy.py --legacy ~/Desktop/projects

Sources (as they existed during the project):
  img-resolve/data/img/                 the 1999 collected images
  image_d/rucaptcha_runs/*/results.json paid RuCaptcha runs (legacy records use absolute image_path)
  img-resolve/data/labels_claude.csv    labels annotated by Claude for all images (label, high/low confidence)
  img-resolve/preprocessing.ipynb       the manual correction passes, as EDITS / CLAUDE_EDITS dicts
  img-resolve/data/labels_clean.csv     after those passes
  img-resolve/data/labels_final.csv     after the last alphabet fixes = what the model was trained on

Outputs:
  data/images/, data/labels/rucaptcha.csv, data/labels/llm.csv  (gitignored, go to the HF dataset)
  data/corrections.csv, data/splits.csv                          (committed)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from captcha2_resolver.data.merge import load_labels, merge  # noqa: E402
from captcha2_resolver.label.consolidate import consolidate  # noqa: E402

STAGES = [
    # (notebook dict name, stage id, note)
    ("EDITS", "length_fix",
     "RuCaptcha answer had the wrong length (5 or 7 symbols); re-read from the image"),
    ("CLAUDE_EDITS", "llm_review",
     "RuCaptcha and Claude labels disagreed; resolved by checking the image"),
]


def notebook_dicts(nb_path: Path) -> dict[str, dict]:
    nb = json.loads(nb_path.read_text())
    found = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in {s[0] for s in STAGES}):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", type=Path, required=True, help="folder holding img-resolve/, image_d/")
    ap.add_argument("--data", type=Path, default=Path("data"))
    a = ap.parse_args()
    ir, idd, data = a.legacy / "img-resolve", a.legacy / "image_d", a.data

    # 1. images
    (data / "images").mkdir(parents=True, exist_ok=True)
    for p in sorted((ir / "data" / "img").glob("*.jpg")):
        shutil.copy2(p, data / "images" / p.name)

    # 2. paid labels: consolidate the raw runs (same rules as the original make_train_txt.py)
    print("rucaptcha:", consolidate(idd / "rucaptcha_runs", data / "labels" / "rucaptcha.csv"))
    ruc = {r["filename"]: r["rucaptcha_label"]
           for r in csv.DictReader((data / "labels" / "rucaptcha.csv").open(encoding="utf-8"))
           if r["rucaptcha_label"]}

    # 3. Claude (LLM) labels
    with (data / "labels" / "llm.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "llm_label", "llm_confidence"])
        for r in csv.DictReader((ir / "data" / "labels_claude.csv").open(encoding="utf-8")):
            w.writerow([Path(r["filename"]).name, r["label"], r["confidence"]])

    # 4. corrections: replay the notebook passes, then the final alphabet fixes
    dicts = notebook_dicts(ir / "preprocessing.ipynb")
    current = dict(ruc)
    rows = []
    for name, stage, note in STAGES:
        for fn, after in dicts[name].items():
            after = after or ""
            if current[fn] != after:
                rows.append([fn, current[fn], after, stage, note])
                current[fn] = after
    clean = {r["filename"]: r["label"] for r in csv.DictReader((ir / "data" / "labels_clean.csv").open())}
    assert clean == current, "replaying notebook edits does not reproduce labels_clean.csv"
    final = pd.read_csv(ir / "data" / "labels_final.csv", dtype=str, keep_default_na=False)
    for fn, after in zip(final["filename"], final["label"]):
        if current[fn] != after:
            rows.append([fn, current[fn], after, "alphabet_fix",
                         "symbol outside the captcha alphabet (z, v, q); re-read from the image"])
            current[fn] = after
    with (data / "corrections.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "before", "after", "stage", "note"])
        w.writerows(rows)

    # 5. splits: reproduce exactly the split the model was trained/evaluated on
    #    (training notebook: labels_final.sample(frac=1, random_state=42); first 10% = validation)
    shuffled = final.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val = int(round(len(shuffled) * 0.1))
    splits = {fn: ("validation" if i < n_val else "train") for i, fn in enumerate(shuffled["filename"])}
    with (data / "splits.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split"])
        w.writerows(sorted(splits.items()))

    # 6. verify: the new merge reproduces labels_final.csv exactly
    stats = merge(data / "images", data / "labels" / "rucaptcha.csv", data / "labels" / "llm.csv",
                  data / "corrections.csv", data / "splits.csv", data / "labels.csv")
    merged = {r["filename"]: r["label"] for r in load_labels(data / "labels.csv")}
    assert merged == dict(zip(final["filename"], final["label"])), "merge != labels_final.csv"
    assert stats["new_split_assignments"] == 0
    print(json.dumps(stats, indent=2))
    print(f"{len(rows)} corrections; merged labels == labels_final.csv ✔")


if __name__ == "__main__":
    main()
