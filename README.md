# captcha2-resolver

An end-to-end, reproducible pipeline for training a text-captcha recognizer on **~900 real
labeled images**: web collection, paid human labeling, cross-checking against Claude (LLM) labels with logged
corrections, PaddleOCR PP-OCRv5 fine-tuning, and a 7.6 MB ONNX model that runs on CPU.

| | |
|---|---|
| **Validation accuracy** (all 6 symbols right, n=100) | **98.0%** |
| Character accuracy | 99.67% |
| CPU latency (ONNX Runtime, batch 1) | ~5.7 ms |
| Labels fixed after paid human labeling | 116 / 1004 (11.6%) |
| Claude (LLM) labels that were exactly right | 786 / 1004 (78.3%): worse than paid humans and than a synthetic-only CRNN |

> **Scope and intended use.** This is a study of building OCR on small, noisy, real-world data.
> The captcha source is intentionally not disclosed. The dataset and model are published on the
> Hugging Face Hub with **manually approved, gated access** for research and educational use.
> Don't use this project to get around access controls on live systems.

## Pipeline

```
collect ─▶ label (RuCaptcha + Claude 2nd opinion) ─▶ merge + corrections ─▶ train (PaddleOCR) ─▶ ONNX ─▶ infer
```

Every stage is one `c2r` command that reads files and writes files.
Details: [docs/pipeline.md](docs/pipeline.md). What was tried, including dead ends:
[docs/experiments.md](docs/experiments.md).

## Quick start: inference

```bash
uv sync                                    # inference-only deps: onnxruntime, opencv, numpy
uv run c2r predict path/to/captcha.jpg --hub <user>/captcha2-resolver-onnx   # needs approved access
```

```python
from captcha2_resolver import CaptchaSolver
solve = CaptchaSolver("models/onnx")       # or CaptchaSolver.from_hub("<user>/captcha2-resolver-onnx")
text, confidence = solve("captcha.jpg")
```

## Reproduce from the dataset

```bash
cp .env.example .env                       # HF_TOKEN, HF_DATASET_REPO, ...
uv sync --extra data
uv run c2r hub pull-dataset                # gated: request access on the dataset page first
uv run c2r data merge                      # rebuilds data/labels.csv from sources + corrections
uv run c2r data validate
```

Train on a GPU with [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb), or locally:

```bash
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/   # or: paddlepaddle==3.3.1 (CPU)
uv sync --extra train --extra export
uv run c2r train prepare                   # lists, dict, head-adapted weights, config (add --cpu on a Mac)
uv run c2r train run                       # train -> eval -> export to models/paddle
uv run c2r export onnx --check data/images/<some>.jpg
uv run c2r evaluate                        # metrics + mistakes on the frozen validation split
```

## Build a dataset from scratch

```bash
uv sync --extra collect --extra label --extra data
# .env: CAPTCHA_SOURCE_URL, CAPTCHA_SELECTOR, RUCAPTCHA_API_KEY
uv run c2r collect --target 2000 --interval 5
uv run c2r label rucaptcha --size 10 --dry-run      # see what would be sent
uv run c2r label rucaptcha --size 10                # paid; resumable with --resume runs/rucaptcha/<id>
uv run c2r label consolidate
uv run c2r data merge && uv run c2r data validate   # then fix labels via data/corrections.csv
```

## Publish (gated)

```bash
uv sync --extra data --extra hub
uv run c2r hub push-dataset --dry-run      # stage locally in work/ and inspect first
uv run c2r hub push-dataset                # private -> upload -> gated=manual -> public
uv run c2r hub push-model
```

Then review access requests in each repo's **Settings → Gated user access**.

## Repository layout

```
src/captcha2_resolver/
  collect.py            stage 1: page -> base64 data URI -> jpg
  label/rucaptcha.py    stage 2: paid human labels, resumable runs
  label/consolidate.py         runs -> data/labels/rucaptcha.csv
  data/merge.py         stage 3: sources + corrections + splits -> data/labels.csv
  data/validate.py, data/split.py, data/hub.py
  train/paddle.py       stage 4: PaddleOCR fine-tune (head adaptation)
  export/onnx.py        stage 5: Paddle -> ONNX (+ linear_v2 fix)
  infer.py, evaluate.py stage 6: ONNX Runtime inference + metrics
  cli.py                `c2r`
data/corrections.csv    every label fix, committed
data/splits.csv         frozen train/validation split, committed
hub/                    dataset + model cards (with gating terms)
notebooks/              Colab training wrapper
experiments/            earlier approaches kept for reference (zero-shot Paddle, synthetic CRNN)
scripts/                migration from the original exploration folders, leak check
docs/                   pipeline + experiments log
```

## Development

```bash
uv sync --extra dev && uv run pre-commit install
uv run pytest && uv run ruff check .
```

The pre-commit hook `scripts/check_no_source_leak.sh` blocks commits that contain any pattern from a
local, gitignored `.source-denylist`.

## License

Code: MIT. Dataset and model: CC BY-NC 4.0 with additional gated-access terms (see the Hub cards).
