"""Generate synthetic text captchas with labels, for testing OCR robustness.

Each distortion has a strength from 0 (off) to 1 (strong), so you can switch them on one at a time.

Usage:
    python gen_captchas.py --out synthetic --n 200                      # all distortions at 0.5
    python gen_captchas.py --out clean --n 50 --level 0                 # plain text, no distortions
    python gen_captchas.py --out lines_only --n 50 --level 0 --lines 1  # only strike-through lines

Writes the images plus labels.txt ("<filename> <answer>" per line), the format test_paddle.py reads.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
FONTS = ["Arial.ttf", "Arial Bold.ttf", "Verdana.ttf", "Verdana Bold.ttf", "Georgia Bold.ttf",
         "Tahoma Bold.ttf", "Trebuchet MS Bold.ttf", "Courier New Bold.ttf", "Comic Sans MS Bold.ttf"]
CHARSET = "abcdefghjkmnpqrstuvwxyz23456789"  # no 0/o, 1/l/i: those are ambiguous even for people

# What each distortion does at strength 1.
EFFECTS = {
    "rotate": "each character tilted up to ±30°",
    "warp": "wavy displacement of the whole text",
    "lines": "up to 3 curved lines across the text",
    "hatch": "characters drawn with scribbled strokes instead of solid fill",
    "gradient": "background shading from white to dark grey",
    "noise": "random dark speckles",
    "jpeg": "heavy JPEG compression",
}


def make_captcha(text: str, rng: random.Random, width: int = 240, height: int = 64, **levels: float) -> Image.Image:
    """Render `text` with the given distortion strengths (keys from EFFECTS, missing = 0)."""
    lv = {name: levels.get(name, 0.0) for name in EFFECTS}

    ink = np.zeros((height, width), np.float32)  # 1 = black, 0 = paper
    font_size = round(height * 0.7)
    x = rng.uniform(0.05, 0.12) * width
    advance = (width * 0.85 - x) / len(text)
    for ch in text:
        font = ImageFont.truetype(str(FONT_DIR / rng.choice(FONTS)), font_size)
        glyph = _render_char(ch, font, rng.uniform(-30, 30) * lv["rotate"])
        if rng.random() < lv["hatch"]:
            glyph = _hatch(glyph, rng)
        gh, gw = glyph.shape
        top = int(np.clip((height - gh) / 2 + rng.uniform(-0.08, 0.08) * height, 0, height - gh))
        left = int(np.clip(x + (advance - gw) / 2, 0, width - gw))
        region = ink[top:top + gh, left:left + gw]
        np.maximum(region, glyph[: region.shape[0], : region.shape[1]], out=region)
        x += advance

    if lv["warp"]:
        ink = _warp(ink, lv["warp"], rng)
    if lv["lines"]:
        ink = np.maximum(ink, _lines(width, height, round(lv["lines"] * 3), rng))

    paper = _background(width, height, lv["gradient"], rng)
    img = paper * (1 - ink)
    if lv["noise"]:
        speckles = np.random.default_rng(rng.randrange(2**32)).random(img.shape) < lv["noise"] * 0.08
        img[speckles] *= 0.2
    out = Image.fromarray(img.clip(0, 255).astype(np.uint8)).convert("RGB")

    if lv["jpeg"]:
        quality = round(95 - lv["jpeg"] * 85)
        ok, buf = cv2.imencode(".jpg", np.asarray(out), [cv2.IMWRITE_JPEG_QUALITY, quality])
        out = Image.fromarray(cv2.imdecode(buf, cv2.IMREAD_UNCHANGED))
    return out


def _render_char(ch: str, font: ImageFont.FreeTypeFont, angle: float) -> np.ndarray:
    left, top, right, bottom = font.getbbox(ch)
    pad = 4
    canvas = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), 0)
    ImageDraw.Draw(canvas).text((pad - left, pad - top), ch, font=font, fill=255)
    canvas = canvas.rotate(angle, resample=Image.BICUBIC, expand=True)
    return np.asarray(canvas, np.float32) / 255


def _hatch(glyph: np.ndarray, rng: random.Random) -> np.ndarray:
    """Replace a solid glyph with many thin random strokes clipped to a slightly grown glyph shape."""
    h, w = glyph.shape
    strokes = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(strokes)
    for _ in range(int(h * w / 4)):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        dx, dy = rng.uniform(-4, 4), rng.uniform(-8, 8)
        draw.line((x0, y0, x0 + dx, y0 + dy), fill=255, width=1)
    grown = cv2.dilate((glyph > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8))
    return np.asarray(strokes, np.float32) / 255 * grown


def _warp(ink: np.ndarray, level: float, rng: random.Random) -> np.ndarray:
    h, w = ink.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    amp, period = level * h * 0.12, rng.uniform(0.6, 1.2) * w
    phase = rng.uniform(0, 2 * np.pi)
    map_y = ys + amp * np.sin(2 * np.pi * xs / period + phase)
    map_x = xs + amp * 0.5 * np.sin(2 * np.pi * ys / (h * 1.5) + phase)
    return cv2.remap(ink, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)


def _lines(width: int, height: int, count: int, rng: random.Random) -> np.ndarray:
    """Quadratic Bezier curves running across most of the image."""
    layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layer)
    for _ in range(count):
        p0 = (rng.uniform(0, 0.2) * width, rng.uniform(0.2, 0.8) * height)
        p1 = (rng.uniform(0.3, 0.7) * width, rng.uniform(-0.3, 1.3) * height)
        p2 = (rng.uniform(0.8, 1.0) * width, rng.uniform(0.2, 0.8) * height)
        t = np.linspace(0, 1, 60)[:, None]
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
        draw.line([tuple(p) for p in pts], fill=255, width=rng.choice([2, 3, 4]))
    return np.asarray(layer, np.float32) / 255


def _background(width: int, height: int, level: float, rng: random.Random) -> np.ndarray:
    ramp = np.linspace(0, 1, width)[None, :] if rng.random() < 0.5 else np.linspace(1, 0, width)[None, :]
    return np.broadcast_to(255 - level * 140 * ramp, (height, width)).astype(np.float32)


def random_text(rng: random.Random, min_len: int, max_len: int) -> str:
    return "".join(rng.choice(CHARSET) for _ in range(rng.randint(min_len, max_len)))


def generate(out: Path, n: int, seed: int, min_len: int, max_len: int, **levels: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        text = random_text(rng, min_len, max_len)
        name = f"{i:05d}.png"
        make_captcha(text, rng, **levels).save(out / name)
        lines.append(f"{name} {text}")
    (out / "labels.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("synthetic"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=7)
    parser.add_argument("--level", type=float, default=0.5, help="default strength for every distortion")
    for name, help_text in EFFECTS.items():
        parser.add_argument(f"--{name}", type=float, help=f"0-1, at 1: {help_text}")
    args = parser.parse_args()

    levels = {name: args.level if getattr(args, name) is None else getattr(args, name) for name in EFFECTS}
    generate(args.out, args.n, args.seed, args.min_len, args.max_len, **levels)
    print(f"Wrote {args.n} captchas to {args.out}/ with " + ", ".join(f"{k}={v:g}" for k, v in levels.items()))


if __name__ == "__main__":
    main()
