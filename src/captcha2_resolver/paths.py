"""Default locations. Every CLI command accepts overrides; these are just the conventions."""

from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data"
IMAGES = DATA / "images"                   # all collected images, <YYYYmmdd_HHMMSS>.jpg
RUCAPTCHA_LABELS = DATA / "labels" / "rucaptcha.csv"   # consolidated paid labels (raw + normalized)
LLM_LABELS = DATA / "labels" / "llm.csv"   # labels annotated by Claude (second opinion; weak)
CORRECTIONS = DATA / "corrections.csv"     # committed: every manual / Claude-disagreement fix, with reason
SPLITS = DATA / "splits.csv"               # committed: frozen filename -> split assignment
LABELS = DATA / "labels.csv"               # built by `c2r data merge`: the training source of truth

RUNS = ROOT / "runs" / "rucaptcha"         # raw paid-labeling runs (gitignored)
WORK = ROOT / "work"                       # training work dir (lists, dict, config, checkpoints)
MODELS = ROOT / "models"
PADDLE_EXPORT = MODELS / "paddle"          # exported Paddle inference model
ONNX_EXPORT = MODELS / "onnx"              # model.onnx + dict.txt + inference.yml

LABEL_LENGTH = 6
