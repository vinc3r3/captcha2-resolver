---
license: cc-by-nc-4.0
library_name: onnx
pipeline_tag: image-to-text
tags:
  - captcha
  - ocr
  - paddleocr
  - onnx
  - ctc
datasets:
  - {{dataset_repo}}
extra_gated_heading: Request access to this model
extra_gated_prompt: >-
  Access is granted manually for research and educational purposes. By requesting access you agree
  not to use this model to bypass access controls of any live system and not to redistribute it.
extra_gated_fields:
  Name: text
  Affiliation / organization: text
  Intended use: text
  I agree not to use this model to circumvent captchas on live systems: checkbox
extra_gated_button_content: Request access
---

# captcha2-resolver (ONNX)

A **PP-OCRv5 mobile** text recognizer (SVTR-LCNet backbone + CTC head) fine-tuned on ~900 labeled
6-symbol text captchas, exported to ONNX. ~7.6 MB, CPU-only inference, no Paddle at runtime.

## Results (frozen validation split, n={{n_eval}})

| metric | value |
|---|---:|
| sequence accuracy (all 6 symbols right) | **{{seq_acc}}** |
| character accuracy | {{char_acc}} |
| CPU latency (Apple Silicon, batch 1, ONNX Runtime) | {{latency}} ms |

The remaining errors are `b`↔`6` confusions.

## Usage

```python
# pip install "captcha2-resolver @ git+https://github.com/vinc3r3/captcha2-resolver" huggingface_hub
from captcha2_resolver import CaptchaSolver

solve = CaptchaSolver.from_hub("{{model_repo}}")
text, confidence = solve("captcha.jpg")
```

Raw ONNX: input `x` float32 `[N, 3, 48, W≤320]`, BGR, aspect-preserving resize to height 48,
scaled to `[-1, 1]`, right zero-padded. Output: softmax `[N, T, 1 + len(dict)]`; greedy CTC with
class 0 = blank, class *i* = line *i−1* of `dict.txt`.

## Training

Fine-tuned from the official PP-OCRv5 mobile rec weights. The pretrained CTC/NRTR head rows for the
captcha's symbols were kept, not re-initialized: this alone gives 8% accuracy before training.
100 epochs, batch 64, OneCycle (max_lr 7e-4), RecAug, 48×320 input, one T4 GPU.
Full recipe and experiments: GitHub repository [vinc3r3/captcha2-resolver](https://github.com/vinc3r3/captcha2-resolver).

## Limitations & intended use

Trained for one captcha style; it will not generalize to other captchas. Research and education
only. Do not use it to circumvent access controls.
