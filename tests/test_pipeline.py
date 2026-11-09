import base64
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from captcha2_resolver.collect import extract_image
from captcha2_resolver.data import merge, split, validate
from captcha2_resolver.infer import H, W, ctc_decode, preprocess
from captcha2_resolver.label.consolidate import consolidate, normalize

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
B64 = base64.b64encode(PNG).decode()


# ---------------------------------------------------------------- collect
@pytest.mark.parametrize("html", [
    f'<div class="c"><div style="background:url(\'data:image/png;base64,{B64}\')"></div></div>',
    f'<div class="c" style="background-image: url(data:image/png;base64,{B64})"></div>',
    f'<span class="c"><img src="data:image/png;base64,{B64}"></span>',
])
def test_extract_inline(html):
    assert extract_image(html, ".c") == ("bytes", PNG)


def test_extract_linked_and_missing():
    assert extract_image('<div class="c"><img src="/cap.jpg"></div>', ".c", "https://x.test/p") == \
        ("url", "https://x.test/cap.jpg")
    assert extract_image("<html></html>", ".c") is None


# ---------------------------------------------------------------- label
def test_normalize():
    assert normalize(" AbC123 ") == ("abc123", "")
    assert normalize("аbс123") == ("", "cyrillic")  # Cyrillic а/с
    assert normalize("ab c12") == ("", "invalid")


def _csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_consolidate_latest_wins_and_legacy_paths(tmp_path):
    r1, r2 = tmp_path / "runs" / "20260101T000000Z_a", tmp_path / "runs" / "20260102T000000Z_b"
    r1.mkdir(parents=True)
    r2.mkdir(parents=True)
    (r1 / "results.json").write_text(json.dumps([
        {"image_path": "/old/abs/x.jpg", "status": "ready", "text": "AAAAAA"},
        {"image_path": "/old/abs/y.jpg", "status": "error"},
    ]))
    (r2 / "results.json").write_text(json.dumps([{"filename": "x.jpg", "status": "ready", "text": "bbbbbb"}]))
    out = tmp_path / "rc.csv"
    assert consolidate(tmp_path / "runs", out)["ready"] == 1
    assert list(csv.DictReader(out.open()))[0]["rucaptcha_label"] == "bbbbbb"


# ---------------------------------------------------------------- data
@pytest.fixture
def dataset(tmp_path):
    imgs = tmp_path / "images"
    imgs.mkdir()
    for n in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        (imgs / n).write_bytes(b"x")
    _csv(tmp_path / "rc.csv", ["filename", "rucaptcha_raw", "rucaptcha_label", "drop_reason"],
         [["a.jpg", "abcdef", "abcdef", ""], ["b.jpg", "abcde", "abcde", ""],
          ["c.jpg", "фыв", "", "cyrillic"]])
    _csv(tmp_path / "llm.csv", ["filename", "llm_label", "llm_confidence"], [["d.jpg", "222222", "low"]])
    _csv(tmp_path / "splits.csv", ["filename", "split"], [["a.jpg", "validation"]])
    return tmp_path


def _merge(d, corrections):
    _csv(d / "corr.csv", ["filename", "before", "after", "stage", "note"], corrections)
    return merge.merge(d / "images", d / "rc.csv", d / "llm.csv", d / "corr.csv", d / "splits.csv",
                       d / "labels.csv")


def test_merge_applies_corrections_and_keeps_frozen_split(dataset):
    stats = _merge(dataset, [["b.jpg", "abcde", "abcdeg", "length_fix", ""]])
    rows = {r["filename"]: r for r in csv.DictReader((dataset / "labels.csv").open())}
    assert rows["b.jpg"]["label"] == "abcdeg" and rows["b.jpg"]["corrected"] == "true"
    assert rows["a.jpg"]["split"] == "validation"           # frozen assignment kept
    assert rows["c.jpg"]["split"] == rows["d.jpg"]["split"] == "unlabeled"
    assert rows["d.jpg"]["llm_label"] == "222222"
    assert stats["labeled"] == 2 and stats["new_split_assignments"] == 1
    assert "b.jpg" in split.load(dataset / "splits.csv")    # new assignment persisted


def test_merge_rejects_stale_correction(dataset):
    with pytest.raises(ValueError, match="stale correction"):
        _merge(dataset, [["b.jpg", "zzzzz", "abcdeg", "length_fix", ""]])


def test_merge_drop(dataset):
    _merge(dataset, [["b.jpg", "abcde", "", "drop", "unreadable"]])
    assert [r["filename"] for r in merge.load_labels(dataset / "labels.csv")] == ["a.jpg"]


def test_validate():
    rows = [{"filename": "a", "label": "abcdef"}, {"filename": "b", "label": "abc"},
            {"filename": "c", "label": "zzz222"}]
    errors, warnings = validate.check(rows)
    assert [r["filename"] for r in errors["length != 6"]] == ["b"]
    assert {r["filename"] for r in warnings["symbols outside core alphabet"]} == {"c"}
    assert [r["filename"] for r in warnings["3+ same symbols in a row"]] == ["c"]


def test_hash_split_is_deterministic_and_roughly_10pct():
    names = [f"{i}.jpg" for i in range(5000)]
    assert [split.hash_split(n) for n in names] == [split.hash_split(n) for n in names]
    frac = sum(split.hash_split(n) == "validation" for n in names) / len(names)
    assert 0.08 < frac < 0.12


# ---------------------------------------------------------------- infer
def test_ctc_decode_collapses_repeats_and_blanks():
    chars = list("abc")
    ids = [1, 1, 0, 1, 2, 0, 0, 3, 3]  # a a _ a b _ _ c c -> "aabc"
    probs = np.eye(4)[ids] * 0.9
    assert ctc_decode(probs, chars) == ("aabc", pytest.approx(0.9))


def test_preprocess_shape_and_range():
    img = np.full((50, 300, 3), 255, dtype=np.uint8)
    x = preprocess(img)
    assert x.shape == (1, 3, H, W) and x.dtype == np.float32
    w = int(np.ceil(H * 300 / 50))
    assert np.allclose(x[0, :, :, :w], 1.0) and np.allclose(x[0, :, :, w:], 0.0)


MODEL = Path("models/onnx")


@pytest.mark.skipif(not (MODEL / "model.onnx").exists(), reason="no local ONNX model")
def test_model_on_a_known_validation_image():
    from captcha2_resolver.infer import CaptchaSolver

    rows = merge.load_labels(Path("data/labels.csv"), "validation")
    solve = CaptchaSolver(MODEL)
    hits = sum(solve(f"data/images/{r['filename']}")[0] == r["label"] for r in rows[:20])
    assert hits >= 18
