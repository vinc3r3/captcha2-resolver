---
license: cc-by-nc-4.0
task_categories:
  - image-to-text
language:
  - en
tags:
  - captcha
  - ocr
  - text-recognition
  - label-noise
pretty_name: captcha2 — 6-symbol text captcha with label provenance
size_categories:
  - 1K<n<10K
dataset_info:
  features:
    - name: image
      dtype: image
    - name: filename
      dtype: string
    - name: label
      dtype: string
    - name: rucaptcha_raw
      dtype: string
    - name: rucaptcha_label
      dtype: string
    - name: llm_label
      dtype: string
    - name: llm_confidence
      dtype: string
    - name: corrected
      dtype: bool
    - name: correction_stage
      dtype: string
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: validation
        path: data/validation-*.parquet
      - split: unlabeled
        path: data/unlabeled-*.parquet
extra_gated_heading: Request access to this dataset
extra_gated_prompt: >-
  This dataset contains real captcha images. Access is granted manually for research and
  educational purposes (OCR, label-noise and annotation-quality studies). By requesting access you
  agree to: not use the data or models trained on it to bypass access controls of any live system;
  not redistribute the images; and cite this dataset in derived work.
extra_gated_fields:
  Name: text
  Affiliation / organization: text
  Intended use: text
  I agree not to use this dataset to circumvent captchas on live systems: checkbox
  I agree not to redistribute the images: checkbox
extra_gated_button_content: Request access
---

# captcha2 — text captcha dataset with full label provenance

{{n_total}} grayscale-ish JPEG captchas (≈300×50 px) of **6 lowercase alphanumeric symbols** from a
~20-symbol alphabet (`2345678abcdefgmnpwxy`; visually ambiguous symbols such as `0/o`, `1/l/i`
are not used), collected from a public web page in October 2026. The source is intentionally not
disclosed.

The interesting part is not the images but **how the labels were made and fixed** — every label
carries its history.

| split | images | label |
|---|---:|---|
| `train` | {{n_train}} | verified |
| `validation` | {{n_val}} | verified (frozen; all reported metrics use it) |
| `unlabeled` | {{n_unlabeled}} | none verified; `llm_label` (Claude) only, so expect ~1 in 5 to be wrong |

## Columns

| column | meaning |
|---|---|
| `image` | the captcha (original JPEG bytes) |
| `filename` | `YYYYmmdd_HHMMSS.jpg`, collection time |
| `label` | final, verified label (empty for `unlabeled`) |
| `rucaptcha_raw` | answer returned by a paid human solver (RuCaptcha `ImageToTextTask`) |
| `rucaptcha_label` | normalized: Latin lowercased; empty if the answer was Cyrillic |
| `llm_label`, `llm_confidence` | label annotated by Claude, used as a second opinion (`high`/`low`); weak: 78.3% exact |
| `corrected` | `true` if `label` differs from `rucaptcha_label` |
| `correction_stage` | `length_fix` / `llm_review` / `alphabet_fix` — see `provenance/corrections.csv` |

## How labels were produced

1. **Paid human labels.** Images were sent to RuCaptcha with `minLength=maxLength=6`.
   ~7% of the returned answers were typed in Cyrillic look-alikes and were discarded; the
   remaining answers are the starting labels.
2. **Second opinion.** Every image was also labeled by **Claude** (an LLM reading the image), with a
   self-reported `high`/`low` confidence. These labels are **not reliable on their own**: only 78.3% of
   them exactly match the verified labels (95.8% per symbol), versus 88.4% for the paid human labels.
   They were used only to find disagreements worth a second look.
3. **Corrections**: {{n_corrected}} of {{n_human}} human labels changed ({{pct_corrected}}), via {{n_corrections}} logged correction rows (before/after + reason):
   - `length_fix` — human answer had 5 or 7 symbols;
   - `llm_review` — human and LLM labels disagreed; resolved by looking at the image;
   - `alphabet_fix` — symbol outside the alphabet (`z`→`2`, `v`→`y`, …).
4. **Sanity checks**: exact length 6, no duplicates, image exists, flags for runs of 3+ identical
   symbols (kept — they are legitimate).

Known residual noise: 3 labels still contain `q` or `s`, which are outside the core alphabet and
are probably errors. They were kept to match the trained model exactly.

## Intended use

OCR fine-tuning on small data, label-noise / annotation-quality research, teaching material.
**Not** for circumventing captchas on live systems.

Pipeline code (collection logic, labeling, training, ONNX export): see the GitHub repository
`captcha2-resolver`.
