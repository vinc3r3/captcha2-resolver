# experiments/

Earlier approaches, kept **as they were run** for reference. They are not part of the package. They
are excluded from ruff and have their own loose dependencies (torch, paddleocr). Results and
conclusions are in [../docs/experiments.md](../docs/experiments.md).

| folder | what | outcome |
|---|---|---|
| `paddle_zero_shot/` | off-the-shelf PP-OCRv5 + a synthetic distortion robustness sweep | not usable without fine-tuning |
| `synthetic_crnn/` | 50k synthetic look-alike captchas → CNN+BiGRU+CTC from scratch | 99.9% on synthetic, 81.4% on real: domain gap |

Paths inside the scripts point at the original working folders (`cap_data/`, `captchas/`). Pass
`--images`/`--dir` to point them at `data/images/`.
