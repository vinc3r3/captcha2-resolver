"""Synthetic captcha generator mimicking cap_data/ (300x50, 6 chars from CHARSET).

Usage: python3 synthetic/gen.py N [--out synthetic/images] [--seed 0]
Writes images plus labels.csv (filename,label) next to them.
"""
import argparse
import csv
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CHARSET = "2345678abcdefgmnpwxy"
LENGTH = 6
W, H = 300, 50
FONT_DIR = "/System/Library/Fonts/Supplemental"
FONTS = ["Tahoma Bold.ttf", "Verdana Bold.ttf", "Arial Bold.ttf"]


def polyline(draw, pts, width):
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        draw.line([(x0, y0), (x1, y1)], fill=0, width=width)
        r = width / 2
        draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=0)


def tapered_line(draw, rng, x0, x1):
    """Long line across the image, 2-3 segments, thickness changing along it."""
    n = rng.randint(2, 3)
    xs = sorted([x0] + [rng.uniform(x0 + 20, x1 - 20) for _ in range(n - 1)] + [x1])
    ys = [rng.uniform(8, 44) for _ in xs]
    w0, w1 = rng.choice([(2, 4), (4, 2), (2, 2), (2, 3), (3, 2), (5, 2)])
    steps = 40
    for i in range(steps):
        t0, t1 = i / steps, (i + 1) / steps
        p0 = interp(xs, ys, x0 + (x1 - x0) * t0)
        p1 = interp(xs, ys, x0 + (x1 - x0) * t1)
        draw.line([p0, p1], fill=0, width=max(1, round(w0 + (w1 - w0) * t0)))


def interp(xs, ys, x):
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / max(xs[i + 1] - xs[i], 1e-6)
            return (x, ys[i] + (ys[i + 1] - ys[i]) * t)
    return (x, ys[-1])


def side_mark(draw, rng, x0, x1):
    """Short stroke at the edge: '<', 'U'-arc, '^'-arc, or hook."""
    kind = rng.choice(["v", "arc_down", "arc_up", "hook", "dash"])
    w = rng.choice([3, 3, 4])
    if kind == "v":
        cy = rng.uniform(15, 35)
        tip = (x0, cy) if rng.random() < .5 else (x1, cy)
        other = x1 if tip[0] == x0 else x0
        polyline(draw, [(other, cy - rng.uniform(5, 12)), tip, (other, cy + rng.uniform(5, 12))], w)
    elif kind in ("arc_down", "arc_up"):
        cy = rng.uniform(15, 35)
        h = rng.uniform(6, 14) * (1 if kind == "arc_down" else -1)
        pts = [(x0 + (x1 - x0) * t, cy + h * math.sin(math.pi * t) + rng.uniform(-1, 1))
               for t in np.linspace(0, 1, 6)]
        polyline(draw, pts, w)
    elif kind == "hook":
        y = rng.uniform(5, 30)
        pts = [(x0, y), (x0 + rng.uniform(0, 8), y + rng.uniform(5, 10)),
               (x1 - rng.uniform(0, 6), y + rng.uniform(8, 14)), (x1, y + rng.uniform(4, 12))]
        polyline(draw, pts, w)
    else:
        y = rng.uniform(5, 45)
        polyline(draw, [(x0, y), (x1, y + rng.uniform(-6, 6))], w + 1)


def render_text(rng, text):
    path = os.path.join(FONT_DIR, rng.choice(FONTS))
    spacing = rng.uniform(-3, -1)
    target_w = rng.uniform(135, 155)
    size = 40
    for _ in range(4):  # fit total text width to target_w
        font = ImageFont.truetype(path, size)
        tw = sum(font.getbbox(c)[2] for c in text) + spacing * (len(text) - 1)
        size = max(28, min(46, round(size * target_w / tw)))
    font = ImageFont.truetype(path, size)
    layer = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(layer)
    x = rng.uniform(78, 92)
    y = rng.uniform(46, 49) - font.getbbox("b")[3]  # baseline near bottom
    boxes = []
    for ch in text:
        l, _, r, _ = font.getbbox(ch)
        d.text((x, y), ch, font=font, fill=0)
        boxes.append((x + l, x + r))
        x += (r - l) + spacing + l
    return np.asarray(layer).copy(), boxes


def row_jitter(arr, x0, x1, amp, rng):
    """Shift each row of columns [x0,x1) by a random offset -> hatched smear."""
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    region = arr[:, x0:x1].copy()
    freq, phase = rng.uniform(.8, 2.5), rng.uniform(0, 2 * math.pi)
    for yy in range(H):
        row = np.full(x1 - x0, 255)
        # two shifted copies: wavy offset plus per-row noise -> hatched look
        for _ in range(2 if amp > 1 else 1):
            s = round(amp * math.sin(yy * freq + phase)) + rng.randint(-1, 1)
            sh = np.roll(region[yy], s)
            if s > 0:
                sh[:s] = 255
            elif s < 0:
                sh[s:] = 255
            row = np.minimum(row, sh)
        arr[yy, x0:x1] = row
    return arr


def ripple(arr, x0, x1, rng):
    """High-frequency 2-D displacement over columns [x0,x1): pixels scatter in x and y,
    giving the hairy cross-hatched texture of the real smeared characters."""
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    yy, xx = np.mgrid[0:H, x0:x1].astype(float)
    ax, ay = rng.uniform(1, 2.5), rng.uniform(1.5, 3.5)
    # periods of ~2-5 px along tilted directions -> moire-like hatching
    kx1, ky1 = rng.uniform(.2, .5), rng.uniform(-.3, .3)
    kx2, ky2 = rng.uniform(-.3, .3), rng.uniform(.2, .5)
    p1, p2 = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
    dx = ax * np.sin(2 * math.pi * (kx1 * xx + ky1 * yy) + p1)
    dy = ay * np.sin(2 * math.pi * (kx2 * xx + ky2 * yy) + p2)
    sx = np.clip(np.rint(xx + dx), 0, W - 1).astype(int)
    sy = np.clip(np.rint(yy + dy), 0, H - 1).astype(int)
    src = arr.copy()
    arr[:, x0:x1] = src[sy, sx]
    return arr


def make(rng, text):
    bg = np.tile(np.linspace(rng.uniform(190, 205), rng.uniform(245, 252), W), (H, 1))
    txt, boxes = render_text(rng, text)

    # Global sawtooth edges (small per-row shift on all text) in some images.
    if rng.random() < .5:
        txt = row_jitter(txt, 0, W, 1, rng)
    # Hairy rippled smear over 1-3 consecutive characters.
    for _ in range(rng.choice([1, 1, 2])):
        k = rng.randint(1, 3)
        i = rng.randint(0, LENGTH - k)
        txt = ripple(txt, boxes[i][0] - 3, boxes[i + k - 1][1] + 3, rng)

    img = Image.fromarray(np.minimum(bg, txt).astype(np.uint8), "L")
    d = ImageDraw.Draw(img)
    tapered_line(d, rng, rng.uniform(25, 35), rng.uniform(265, 275))
    side_mark(d, rng, rng.uniform(28, 35), rng.uniform(65, 75))
    side_mark(d, rng, rng.uniform(232, 245), rng.uniform(265, 275))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(.3, .6)))
    ImageDraw.Draw(img).rectangle([0, 0, W - 1, H - 1], outline=0)
    return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "images"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["filename", "label"])
        for i in range(a.n):
            text = "".join(rng.choice(CHARSET) for _ in range(LENGTH))
            name = f"syn_{i:06d}.jpg"
            make(rng, text).save(os.path.join(a.out, name), quality=rng.randint(70, 90))
            wr.writerow([name, text])


if __name__ == "__main__":
    main()
