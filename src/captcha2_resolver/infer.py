"""Stage 6 — inference with ONNX Runtime only (no Paddle at runtime).

    from captcha2_resolver import CaptchaSolver
    solve = CaptchaSolver("models/onnx")           # or CaptchaSolver.from_hub("user/repo")
    text, confidence = solve("image.jpg")

Preprocessing must match training exactly (PaddleOCR RecResizeImg): BGR, keep aspect ratio,
height 48, width <= 320, scale to [-1, 1], right-pad with zeros. Decoding is greedy CTC.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

H, W = 48, 320


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    import cv2

    if img_bgr is None:
        raise ValueError("could not read image")
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    w = min(W, math.ceil(H * img_bgr.shape[1] / img_bgr.shape[0]))
    x = np.zeros((1, 3, H, W), dtype=np.float32)
    x[0, :, :, :w] = (cv2.resize(img_bgr, (w, H)).astype(np.float32).transpose(2, 0, 1) / 255 - 0.5) / 0.5
    return x


def ctc_decode(probs: np.ndarray, chars: list[str]) -> tuple[str, float]:
    """probs: [T, classes]. Drop blanks (class 0) and repeats; confidence = mean max prob of kept steps."""
    ids, scores = probs.argmax(-1), probs.max(-1)
    keep = (ids != 0) & np.r_[True, ids[1:] != ids[:-1]]
    text = "".join(chars[i - 1] for i in ids[keep])
    return text, float(scores[keep].mean()) if keep.any() else 0.0


class CaptchaSolver:
    def __init__(self, model_dir: str | Path = "models/onnx", providers: list[str] | None = None):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"),
                                         providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.chars = [line.rstrip("\n") for line in (model_dir / "dict.txt").open(encoding="utf-8")]

    @classmethod
    def from_hub(cls, repo_id: str, revision: str | None = None, token: str | None = None) -> CaptchaSolver:
        """Requires `pip install huggingface_hub` and approved access to the gated repo."""
        from huggingface_hub import snapshot_download

        return cls(snapshot_download(repo_id, revision=revision, token=token,
                                     allow_patterns=["model.onnx", "dict.txt"]))

    def probs(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: x})[0]

    def __call__(self, image: str | Path | np.ndarray) -> tuple[str, float]:
        """image: file path or BGR ndarray (cv2.imread). Returns (text, confidence)."""
        import cv2

        img = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
        return ctc_decode(self.probs(preprocess(img))[0], self.chars)
