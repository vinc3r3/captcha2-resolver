"""Run PaddleOCR over the images in captchas/ and score it against captchas/labels.txt.

Usage:
    python test_paddle.py                 # recognition-only (best for single-line captchas)
    python test_paddle.py --mode full     # full detection + recognition pipeline
    python test_paddle.py --dir other_dir --model PP-OCRv5_mobile_rec
    python test_paddle.py --no-crop       # don't crop the noisy left/right margins
    python test_paddle.py --save-crops crops/   # also write the cropped images for inspection

labels.txt (optional, in --dir) has one "<filename> <answer>" pair per line.
"""

import argparse
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# Cropping tuned on captchas/: the noise strokes cover ~5-20% of a column's height,
# the letters ~30-60%, so columns above INK_THRESHOLD are taken as text.
DARK_LEVEL = 100  # grayscale value below which a pixel counts as ink
INK_THRESHOLD = 0.25  # fraction of dark pixels in a (smoothed) column to count as text
BORDER_FRAC = 0.04  # ignore a frame this thick (relative to height), e.g. captcha2.png
PAD_FRAC = 0.03  # padding kept on each side of the text (relative to width)

def find_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_labels(directory: Path) -> dict[str, str]:
    labels_file = directory / "labels.txt"
    if not labels_file.exists():
        return {}
    pairs = (line.split() for line in labels_file.read_text().splitlines() if line.strip())
    return {name: answer for name, answer in pairs}


def crop_to_text(img: Image.Image) -> Image.Image:
    """Cut off the left and right margins, which hold only noise strokes, keeping the text."""
    gray = np.asarray(img.convert("L"))
    h, w = gray.shape
    dark = gray < DARK_LEVEL
    border = max(2, round(h * BORDER_FRAC))
    dark[:border, :] = dark[-border:, :] = False
    dark[:, :border] = dark[:, -border:] = False

    window = max(3, w // 40)
    density = np.convolve(dark.sum(axis=0) / h, np.ones(window) / window, mode="same")
    text_cols = np.flatnonzero(density > INK_THRESHOLD)
    if text_cols.size == 0:
        return img

    pad = round(w * PAD_FRAC)
    left, right = max(0, text_cols[0] - pad), min(w, text_cols[-1] + pad)
    return img.crop((left, 0, right, h))


def preprocess(path: Path, crop: bool) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return crop_to_text(img) if crop else img


def to_bgr(img: Image.Image) -> np.ndarray:
    """PaddleOCR expects BGR arrays."""
    return np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])


def recognize(images: list[np.ndarray], mode: str, model_name: str) -> list[tuple[str, float]]:
    """Return (text, score) for each image."""
    if mode == "rec":
        # Captchas are a single line of text, so skip detection and feed the whole image to the recognizer.
        from paddleocr import TextRecognition

        model = TextRecognition(model_name=model_name)
        results = [next(iter(model.predict(input=image, batch_size=1))) for image in images]
        return [(r["rec_text"], r["rec_score"]) for r in results]

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    results = [next(iter(ocr.predict(image))) for image in images]
    return [("".join(r["rec_texts"]), min(r["rec_scores"], default=0.0)) for r in results]


def normalize(text: str) -> str:
    """Fold full-width characters (e.g. "２５ｐ" -> "25p"), drop spaces, and lowercase."""
    return "".join(unicodedata.normalize("NFKC", text).split()).lower()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=Path(__file__).parent / "captchas")
    parser.add_argument("--mode", choices=["rec", "full"], default="rec")
    parser.add_argument("--model", default="PP-OCRv5_server_rec", help="recognition model for --mode rec")
    parser.add_argument("--no-crop", action="store_true", help="don't crop the noisy left/right margins")
    parser.add_argument("--save-crops", type=Path, metavar="DIR", help="write the (cropped) images here")
    args = parser.parse_args()

    paths = find_images(args.dir)
    if not paths:
        raise SystemExit(f"No images found in {args.dir}")
    labels = load_labels(args.dir)
    crop = not args.no_crop
    print(f"Testing {len(paths)} images from {args.dir} (mode={args.mode}, crop={crop})\n")

    processed = [preprocess(p, crop) for p in paths]
    if args.save_crops:
        args.save_crops.mkdir(parents=True, exist_ok=True)
        for path, img in zip(paths, processed):
            img.save(args.save_crops / path.name)

    start = time.perf_counter()
    predictions = recognize([to_bgr(img) for img in processed], args.mode, args.model)
    elapsed = time.perf_counter() - start

    exact = chars_right = chars_total = 0
    for path, (text, score) in zip(paths, predictions):
        expected = labels.get(path.name)
        if expected is None:
            print(f"{path.name:<16} {text!r:<12} score={score:.3f}")
            continue
        errors = edit_distance(normalize(text), normalize(expected))
        exact += errors == 0
        chars_right += max(0, len(expected) - errors)
        chars_total += len(expected)
        mark = "OK " if errors == 0 else "BAD"
        print(f"{mark} {path.name:<16} {text!r:<12} expected {expected!r:<10} score={score:.3f}")

    print(f"\n{elapsed / len(paths) * 1000:.0f}ms per image")
    if chars_total:
        n = sum(p.name in labels for p in paths)
        print(f"Exact match: {exact}/{n}   Character accuracy: {chars_right / chars_total:.1%}")


if __name__ == "__main__":
    main()
