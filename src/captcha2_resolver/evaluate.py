"""Evaluate an ONNX model on a split: sequence accuracy, char accuracy, latency, mistakes."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from captcha2_resolver.data.merge import load_labels
from captcha2_resolver.infer import CaptchaSolver


def evaluate(model_dir: Path, labels_csv: Path, images_dir: Path, split: str = "validation",
             out_dir: Path | None = None) -> dict:
    rows = load_labels(labels_csv, split)
    if not rows:
        raise ValueError(f"no labeled rows in split {split!r}")
    solver = CaptchaSolver(model_dir)
    solver(str(images_dir / rows[0]["filename"]))  # warm-up, excluded from latency
    mistakes, correct, char_ok, char_total = [], 0, 0, 0
    t0 = time.perf_counter()
    for r in rows:
        pred, conf = solver(str(images_dir / r["filename"]))
        label = r["label"]
        correct += pred == label
        char_ok += sum(a == b for a, b in zip(pred, label))
        char_total += max(len(label), len(pred))
        if pred != label:
            mistakes.append({"filename": r["filename"], "label": label, "pred": pred, "confidence": f"{conf:.3f}"})
    ms = (time.perf_counter() - t0) / len(rows) * 1000
    metrics = {"split": split, "n": len(rows), "sequence_accuracy": correct / len(rows),
               "char_accuracy": char_ok / char_total, "ms_per_image": ms,
               "model_dir": str(model_dir)}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        with (out_dir / f"mistakes_{split}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "label", "pred", "confidence"])
            w.writeheader()
            w.writerows(mistakes)
    metrics["mistakes"] = mistakes
    return metrics
