"""Measure how much each synthetic distortion hurts PaddleOCR.

For every distortion in gen_captchas.EFFECTS, renders --n captchas with only that distortion
at each strength level, runs the recognizer, and prints a table of character accuracy
(and exact-match rate in parentheses). The "all" row turns every distortion on together.

Usage:
    python robustness.py
    python robustness.py --n 100 --model PP-OCRv5_mobile_rec
"""

import argparse
import random

from paddleocr import TextRecognition

from gen_captchas import EFFECTS, make_captcha, random_text
from test_paddle import edit_distance, normalize, to_bgr

LEVELS = [0.25, 0.5, 0.75, 1.0]


def score(model: TextRecognition, n: int, seed: int, levels: dict[str, float]) -> tuple[float, float]:
    """Return (character accuracy, exact-match rate) on n fresh captchas."""
    rng = random.Random(seed)
    texts = [random_text(rng, 4, 7) for _ in range(n)]
    images = [to_bgr(make_captcha(t, rng, **levels)) for t in texts]
    preds = [normalize(r["rec_text"]) for r in model.predict(input=images, batch_size=8)]
    errors = [edit_distance(p, t) for p, t in zip(preds, texts)]
    chars = sum(max(0, len(t) - e) for t, e in zip(texts, errors)) / sum(map(len, texts))
    return chars, sum(e == 0 for e in errors) / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=40, help="captchas per cell")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="PP-OCRv5_server_rec")
    args = parser.parse_args()

    model = TextRecognition(model_name=args.model)
    clean, clean_exact = score(model, args.n, args.seed, {})
    print(f"\nNo distortion: {clean:.0%} ({clean_exact:.0%} exact), n={args.n} per cell\n")
    print(f"{'':<10}" + "".join(f"{lv:>14}" for lv in LEVELS))
    for name in [*EFFECTS, "all"]:
        cells = []
        for lv in LEVELS:
            levels = {e: lv for e in EFFECTS} if name == "all" else {name: lv}
            chars, exact = score(model, args.n, args.seed, levels)
            cells.append(f"{chars:>7.0%} ({exact:>3.0%})")
        print(f"{name:<10}" + "".join(cells), flush=True)


if __name__ == "__main__":
    main()
