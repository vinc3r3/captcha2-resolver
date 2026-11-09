"""Fine-tune PP-OCRv5 mobile rec with the official PaddleOCR tools.

`prepare()` writes everything PaddleOCR's `tools/train.py` needs into a work dir:
  train_list.txt / val_list.txt   "<abs image path>\\t<label>" (from data/labels.csv + frozen splits)
  dict.txt                        alphabet = every symbol present in the training labels
  *_adapted.pdparams              pretrained weights with heads cut down to our alphabet (see below)
  rec_config.yml                  official PP-OCRv5_mobile_rec.yml + our overrides

Head adaptation — the trick that made training on ~900 images work:
  The pretrained CTC and NRTR heads output ~18k classes. Instead of re-initialising them for our
  ~22 symbols (which is what PaddleOCR does on a shape mismatch), keep the pretrained *rows* of
  exactly those symbols. The model can then already read the captchas at step 0:
  val accuracy 0% -> 8%, normalized char similarity 0.08 -> 0.63, before any training.

Then `train()` runs tools/train.py -> tools/eval.py -> tools/export_model.py.

Recipe history: CPU training (Mac, no Apple-GPU backend in Paddle) took ~30 s/iteration (~3 h
for 25 epochs), so the final model was trained on a Colab T4: 100 epochs, batch 64, OneCycle
max_lr 7e-4, RecAug augmentation, fixed 48x320 input. See docs/experiments.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

PRETRAINED_URL = ("https://paddle-model-ecology.bj.bcebos.com/paddlex/"
                  "official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams")
PADDLEOCR_GIT = "https://github.com/PaddlePaddle/PaddleOCR.git"


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    max_lr: float = 7e-4
    augment: bool = True
    use_gpu: bool = True
    num_workers: int = 2
    eval_every: int = 50           # iterations; best val acc is kept as best_accuracy
    image_shape: tuple = (3, 48, 320)


def ensure_paddleocr(paddleocr_dir: Path) -> Path:
    if not (paddleocr_dir / "tools" / "train.py").exists():
        subprocess.run(["git", "clone", "--depth", "1", PADDLEOCR_GIT, str(paddleocr_dir)], check=True)
    return paddleocr_dir


def write_lists(labels_csv: Path, images_dir: Path, work: Path) -> tuple[list[str], dict[str, int]]:
    from captcha2_resolver.data.merge import load_labels

    rows = load_labels(labels_csv)
    counts = {}
    for split, fname in (("train", "train_list.txt"), ("validation", "val_list.txt")):
        part = [r for r in rows if r["split"] == split]
        counts[split] = len(part)
        with (work / fname).open("w", encoding="utf-8") as f:
            for r in part:
                f.write(f"{(images_dir / r['filename']).resolve()}\t{r['label']}\n")
    # Alphabet from train + validation labels: a val label with a symbol missing from the dict
    # would be silently skipped by PaddleOCR's label encoder and inflate val accuracy.
    alphabet = sorted(set("".join(r["label"] for r in rows if r["split"] in ("train", "validation"))))
    (work / "dict.txt").write_text("\n".join(alphabet) + "\n", encoding="utf-8")
    return alphabet, counts


def adapt_pretrained(pretrained: Path, src_dict: Path, alphabet: list[str], out: Path) -> Path:
    """Keep the pretrained head rows of our symbols.

    CTC  classes: [blank] + dict (+ space)
    NRTR classes: [blank, <unk>, <s>, </s>] + dict (+ space) + [pad]
    """
    import numpy as np
    import paddle

    src_chars = [line.rstrip("\r\n") for line in src_dict.open(encoding="utf-8")]
    src_index = {c: i for i, c in enumerate(src_chars)}  # last occurrence wins, like the label encoder
    ids = np.array([src_index[c] for c in alphabet])
    ctc_rows = np.concatenate([[0], 1 + ids])
    nrtr_rows = np.concatenate([[0, 1, 2, 3], 4 + ids, [-1]])
    params = paddle.load(str(pretrained))
    remap = {
        "head.ctc_head.fc.weight": (ctc_rows, 1),
        "head.ctc_head.fc.bias": (ctc_rows, 0),
        "head.gtc_head.embedding.embedding.weight": (nrtr_rows, 0),
        "head.gtc_head.tgt_word_prj.weight": (nrtr_rows, 1),
    }
    for key, (rows, axis) in remap.items():
        params[key] = paddle.to_tensor(np.take(params[key].numpy(), rows, axis=axis))
    paddle.save(params, str(out))
    return out


def build_config(paddleocr_dir: Path, work: Path, pretrained: Path, export_dir: Path,
                 cfg: TrainConfig) -> Path:
    import yaml

    base = paddleocr_dir / "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml"
    conf = yaml.safe_load(base.read_text())
    out_dir = work / "output"
    conf["Global"].update({
        "use_gpu": cfg.use_gpu,
        "distributed": False,
        "epoch_num": cfg.epochs,
        "print_batch_step": 10,
        "save_model_dir": str(out_dir),
        "save_epoch_step": cfg.epochs,        # only latest + best_accuracy otherwise
        "eval_batch_step": [0, cfg.eval_every],
        "pretrained_model": str(pretrained),
        "character_dict_path": str(work / "dict.txt"),
        "use_space_char": False,
        "save_inference_dir": str(export_dir),
        "save_res_path": str(out_dir / "predicts.txt"),
    })
    conf["Optimizer"]["lr"] = {"name": "OneCycle", "max_lr": cfg.max_lr}
    decode = {"DecodeImage": {"img_mode": "BGR", "channel_first": False}}
    tail = [
        {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
        {"RecResizeImg": {"image_shape": list(cfg.image_shape)}},
        {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_gtc", "length", "valid_ratio"]}},
    ]
    train_tf = [decode, {"RecAug": {}}, *tail] if cfg.augment else [decode, *tail]
    for section, lst, tf, shuffle in (("Train", "train_list.txt", train_tf, True),
                                      ("Eval", "val_list.txt", [decode, *tail], False)):
        conf[section]["dataset"] = {"name": "SimpleDataSet", "data_dir": "/",
                                    "label_file_list": [str(work / lst)], "transforms": tf}
        conf[section]["loader"] = {"shuffle": shuffle, "drop_last": shuffle,
                                   "batch_size_per_card": cfg.batch_size, "num_workers": cfg.num_workers}
    conf["Train"].pop("sampler", None)  # MultiScaleSampler -> fixed-size batches
    path = work / "rec_config.yml"
    path.write_text(yaml.safe_dump(conf, sort_keys=False))
    (work / "train_config.yml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    return path


def prepare(labels_csv: Path, images_dir: Path, work: Path, paddleocr_dir: Path, export_dir: Path,
            cfg: TrainConfig) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    ensure_paddleocr(paddleocr_dir)
    alphabet, counts = write_lists(labels_csv, images_dir, work)
    print(f"train={counts['train']} val={counts['validation']} alphabet({len(alphabet)})={''.join(alphabet)}")
    pretrained = work / "PP-OCRv5_mobile_rec_pretrained.pdparams"
    if not pretrained.exists():
        print("downloading", PRETRAINED_URL)
        urllib.request.urlretrieve(PRETRAINED_URL, pretrained)
    adapted = adapt_pretrained(pretrained, paddleocr_dir / "ppocr/utils/dict/ppocrv5_dict.txt",
                               alphabet, work / "PP-OCRv5_mobile_rec_adapted.pdparams")
    config = build_config(paddleocr_dir, work, adapted, export_dir, cfg)
    print("config:", config)
    return config


def _run(paddleocr_dir: Path, *args: str) -> None:
    cmd = [sys.executable, *args]
    print(" ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=paddleocr_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    try:
        for line in proc.stdout:
            if "not matched with loaded params" not in line:  # only appears without head adaptation
                print(line, end="")
    except KeyboardInterrupt:
        proc.terminate()  # don't leave training running on the GPU
        proc.wait()
        raise
    if proc.wait() != 0:
        raise RuntimeError(f"{args[0]} failed with exit code {proc.returncode}")


def train(config: Path, paddleocr_dir: Path, export_dir: Path, resume: bool = False) -> Path:
    out_dir = config.parent / "output"
    extra = ["-o", f"Global.checkpoints={out_dir / 'latest'}"] if resume else []
    _run(paddleocr_dir, "tools/train.py", "-c", str(config), *extra)
    best = out_dir / "best_accuracy"
    _run(paddleocr_dir, "tools/eval.py", "-c", str(config), "-o", f"Global.checkpoints={best}")
    _run(paddleocr_dir, "tools/export_model.py", "-c", str(config), "-o",
         f"Global.pretrained_model={best}", f"Global.save_inference_dir={export_dir}")
    return export_dir
