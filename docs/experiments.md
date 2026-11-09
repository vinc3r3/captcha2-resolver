# Experiments log

Everything that was tried, in order, including the dead ends. The numbers come from the saved logs and
notebook outputs of the original exploration folders. Nothing here is reconstructed from memory.

## 0. Zero-shot PaddleOCR (`experiments/paddle_zero_shot/`)

**Question:** can an off-the-shelf OCR model read the captcha without training?

- `test_paddle.py` runs PP-OCRv5 recognition (rec-only or det+rec) on a handful of captchas. An
  optional column-ink crop removes the noisy margins: noise strokes cover ~5–20% of a column's height,
  letters ~30–60%.
- `gen_captchas.py` + `robustness.py` render synthetic captchas with one distortion at a time (rotate,
  warp, lines, …, each at strength 0–1) and measure how much each distortion hurts PaddleOCR.

**Outcome:** no accuracy table was saved from this stage. The project moved on to training on the
target captcha with a restricted alphabet (the pretrained model decodes over an ~18k-symbol dictionary).

## 1. Synthetic data + CRNN from scratch (`experiments/synthetic_crnn/`)

**Question:** can we skip labeling by training on look-alike synthetic captchas?

- `gen.py` renders 300×50 captchas with the same 20-symbol alphabet and Tahoma/Verdana/Arial Bold
  fonts, mimicking the real ones. 50,000 images were generated.
- `train.py`: CNN + BiGRU + CTC, AdamW, OneCycle (max_lr 2e-3), trained on Apple MPS.

| run | epochs | synthetic val seq-acc | synthetic val char-acc |
|---|---|---:|---:|
| v1 (`train_v1.log`) | 2 of 20 (stopped) | 96.9% | 99.1% |
| v2 (`train.log`) | 6 of 12 | **99.9%** | 99.98% |

**Outcome:** near-perfect on synthetic data, but on the 1004 real captchas with verified labels it is
right only **81.4%** of the time (817/1004, fixed-length decoding). That is a clear **synthetic-to-real gap**. The model's predictions
were still useful as a cheap, independent *second opinion* for finding label errors.

## 2. Paid human labels (RuCaptcha)

- 1999 images submitted across 2 runs (a 10-image pilot, then the rest). Cost: $0.001 per image.
- **1078 answered, 921 failed**: 887 `ERROR_ZERO_BALANCE` (the account ran out of credit), 33
  `ERROR_CAPTCHA_UNSOLVABLE`, 1 HTTP 500. That's why only about half of the images have human labels. 74 answers (6.9%) came back in Cyrillic
  look-alikes and were dropped, leaving **1004 human labels**.
- 26 of the 1004 (2.6%) had the wrong length despite `minLength=maxLength=6`.

## 3. Label correction

The human labels were compared against the labels annotated by Claude (`labels/llm.csv`, see below), and every
disagreement or impossible label was checked by looking at the image:

| stage | fixes | what |
|---|---:|---|
| `length_fix` | 25 | 5- or 7-symbol answers (26 were found; 1 was left at 5 symbols and fixed later by `alphabet_fix`) |
| `llm_review` | 88 | human vs LLM disagreement, resolved by eye |
| `alphabet_fix` | 5 | `z`, `v`, `q` outside the alphabet |
| **total** | **118 rows → 116 / 1004 labels changed (11.6%)** | (2 labels were fixed twice) |

See `data/corrections.csv` for the exact rows.

### Claude (LLM) labels perform badly

Every image was also labeled by Claude, an LLM reading the image and reporting a label plus a
`high`/`low` confidence. Scored against the 1004 verified labels:

| labeler | exact (all 6 symbols) | per symbol |
|---|---:|---:|
| paid humans (RuCaptcha, before correction) | 888 / 1004 = **88.4%** | |
| CRNN trained only on synthetic data (§1) | 817 / 1004 = **81.4%** | |
| **Claude** | 786 / 1004 = **78.3%** | 95.8% |
| fine-tuned PP-OCRv5 mobile (final model, validation) | 98 / 100 = **98.0%** | 99.7% |

- Claude labeled worse than a small CRNN that had **never seen a real captcha**, and roughly 2× more
  often wrong than the paid humans.
- Its confidence is barely usable: it called only 33 of 1999 images `high` (27/28 of the verified
  ones were right). On everything else it said `low`, and it was right 77.8% of the time (759/976).
- It was still worth having. Its mistakes rarely match the human mistakes, so *disagreement*
  between the two is a cheap way to find images to re-check. That's where most of the 88
  `llm_review` fixes came from. **Never use LLM labels as ground truth here, only as a second opinion.**
- The 995 `unlabeled` images have only Claude labels. Expect about 1 in 5 of them to be wrong. **Lesson:** even paid human captcha solvers are wrong
about 1 time in 9 on this captcha. Without the correction pass, the validation set itself would be
~11% noisy, which caps any measurable accuracy far below the 98% we got.

Residual: 3 labels still contain `q`/`s`. They are kept so the dataset matches the trained model exactly.

## 4. PP-OCRv5 **server** rec fine-tune (first attempt)

- Fine-tuned on an earlier label snapshot (803 train / 201 val, before the correction pass), on a GPU
  machine, 10 epochs, OneCycle max_lr 7e-4, batch 64.
- 11 mismatches when re-predicting the train split.
- Large model (75.5 MB of weights vs 7.6 MB for mobile, ~10×). We moved on to the mobile variant for
  deployment size and CPU speed.

## 5. PP-OCRv5 **mobile** rec fine-tune: final model

- Data: 904 train / 100 validation (seeded shuffle of the corrected labels, frozen in `data/splits.csv`).
- **Head adaptation** (`train/paddle.py::adapt_pretrained`): keep the pretrained output rows for our
  symbols instead of re-initializing the heads. Before any training this already gives 8% val accuracy
  and 0.63 normalized char similarity (vs 0% / 0.08 with random heads).
- CPU attempt (Mac): ~31 s/iteration → ~3 h ETA for 25 epochs. Abandoned in favor of Colab.
- **Colab T4:** 100 epochs, batch 64, OneCycle max_lr 7e-4, `RecAug` augmentation, fixed 48×320.

## 6. ONNX export

- `paddle2onnx` 2.1 can't convert Paddle ≥ 3.3's fused `pd_op.linear_v2`. Fix: rewrite each one as
  `matmul + add` in the PIR program before conversion (`export/onnx.py::replace_linear_v2`).
- Opset 14, ~7.6 MB.

**Final results** (ONNX Runtime, CPU, Apple Silicon, measured with `c2r evaluate`):

| split | n | sequence acc | char acc | latency |
|---|---:|---:|---:|---:|
| validation | 100 | **98.0%** | 99.67% | ~5.7 ms/img |
| train | 904 | 100.0% | 100.0% | |

Both validation errors are `b`→`6` confusions (`4w5bbx`→`4w5b6x`, `b8m42g`→`b6m42g`).

## Ideas not tried yet

- Pseudo-label the 995 `unlabeled` images with the final model, keep high-confidence ones, retrain.
- Constrain decoding to exactly 6 symbols (the CRNN experiment did this with `decode_fixed`).
- A bigger validation split: with n=100, one error = 1 percentage point.
