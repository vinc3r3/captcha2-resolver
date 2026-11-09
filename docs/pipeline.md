# Pipeline

```
 ┌──────────┐   ┌───────────────┐   ┌────────────────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐
 │ collect  │──▶│ label         │──▶│ merge + correct    │──▶│ train      │──▶│ export   │──▶│ infer    │
 │ web page │   │ RuCaptcha     │   │ corrections.csv    │   │ PaddleOCR  │   │ ONNX     │   │ ORT only │
 │ → images │   │ + Claude 2nd  │   │ splits.csv (frozen)│   │ PP-OCRv5 m │   │ + dict   │   │ 5 ms/img │
 └──────────┘   └───────────────┘   └────────────────────┘   └────────────┘   └──────────┘   └──────────┘
   c2r collect    c2r label ...       c2r data merge           c2r train ...    c2r export     c2r predict
                                      c2r data validate                         c2r evaluate
                         └────────── c2r hub push-dataset ──────────┘                └─ c2r hub push-model
```

Each stage reads files and writes files. Nothing is passed in memory between stages, so any stage can
be re-run, inspected or swapped out on its own.

## 1. Collect: `src/captcha2_resolver/collect.py`

The captcha is not a separate image URL. It's **inlined into the HTML as a base64 data URI**, either
as `<img src="data:image/jpeg;base64,…">` or inside an inline CSS `background: url('data:…')`.
So collection is:

1. `GET` the page with a **fresh `requests.Session`**. The server binds the captcha to the session
   cookie. A new session means a new captcha. Reusing one returns the same image.
2. Parse with BeautifulSoup and `select_one(CAPTCHA_SELECTOR)`.
3. Regex the data URI out of `src` or `style`, then `base64.b64decode`.
4. Save as `YYYYmmdd_HHMMSS.jpg`. Skip exact duplicates by SHA-1.
5. Sleep so each iteration takes at least `--interval` seconds. Log errors and keep going. The run
   is restartable because it counts the files already on disk.

If the selector doesn't match, the page is dumped to `data/page_dump.html` (gitignored) so you can
find the right selector. The source URL and selector live only in `.env`.

**Be polite:** keep `--interval` at several seconds, collect only what you need, and respect the
site's terms.

## 2. Label: `label/rucaptcha.py`, `label/consolidate.py`

Paid human labels via RuCaptcha `ImageToTextTask` (~$0.001/image). Runs are folders with
`scope.json` + `results.json`, checkpointed after every task, so `--resume` never pays twice.
Images already sent in any run are never re-sent. `consolidate` keeps the latest answer per image,
drops Cyrillic answers and lowercases the rest.

The second opinion (`data/labels/llm.csv`) is an independent label per image, **annotated by Claude**
(an LLM reading the image, giving a label plus `high`/`low` confidence). Its only job is to *find
disagreements*, never to overrule a human automatically. **Claude is a poor captcha labeler** (78.3%
exact on the verified labels, see [experiments.md](experiments.md#claude-llm-labels-perform-badly)),
but its errors are mostly *different* from the human errors, so disagreements are a cheap way to
find mislabeled images.

## 3. Merge + correct: `data/merge.py`, `data/validate.py`, `data/split.py`

- Raw sources are **immutable**. All fixes are rows in `data/corrections.csv`
  (`before → after`, stage, note). The merge refuses a correction whose `before` doesn't match.
- `data/splits.csv` is **frozen and committed**. New images are assigned by filename hash (stable
  as the dataset grows), and existing assignments never move.
- `validate` hard-fails on empty or wrong-length labels, duplicates and missing files. It warns on
  symbols outside the alphabet and on runs of 3+ identical symbols.

## 4. Train: `train/paddle.py`

`c2r train prepare` writes the PaddleOCR list files and dict, the head-adapted pretrained weights and
the config. `c2r train run` calls the official `tools/train.py`, `eval.py` and `export_model.py`.
Train on a GPU (`notebooks/train_colab.ipynb`). CPU works but takes ~30 s per iteration.

## 5. Export: `export/onnx.py`

`paddle2onnx`, plus the `linear_v2 → matmul + add` rewrite for Paddle ≥ 3.3. `--check IMAGE`
compares ONNX against Paddle outputs on one image.

## 6. Infer: `infer.py`

`CaptchaSolver` depends only on `onnxruntime`, `opencv-python-headless` and `numpy`. Its
preprocessing copies PaddleOCR's `RecResizeImg` exactly. Any deviation (PIL resize, RGB order)
costs accuracy.
