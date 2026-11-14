"""Publish to / fetch from the Hugging Face Hub with gated (manual-approval) access.

Dataset layout on the Hub (Parquet with an explicit schema and the original JPEG bytes embedded, so
`load_dataset(repo)` and the dataset viewer just work):

    README.md                          dataset card (schema, gating prompt + fields)
    data/{train,validation,unlabeled}-00000-of-00001.parquet
        image, filename, label, rucaptcha_raw, rucaptcha_label, llm_label, llm_confidence,
        corrected, correction_stage
    provenance/corrections.csv         every label fix with reason
    provenance/splits.csv

Publishing order avoids any window where files are openly downloadable:
    create repo as PRIVATE -> upload -> enable gating (manual) -> make public.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path

from captcha2_resolver.data.merge import FIELDS
from captcha2_resolver.label.consolidate import normalize

HUB_SPLITS = ("train", "validation", "unlabeled")
CARDS = Path(__file__).resolve().parents[3] / "hub"


def _api(token: str | None):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def _publish(api, repo_id: str, repo_type: str, folder: Path, gated: str, message: str) -> str:
    if gated not in ("manual", "auto", "private"):
        raise ValueError("gated must be one of: manual, auto, private")
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
    # create_repo(exist_ok=True) leaves an existing repo's visibility alone: force it private
    # so nothing is openly downloadable between the upload and enabling gating.
    api.update_repo_settings(repo_id, repo_type=repo_type, private=True)
    api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(folder), commit_message=message)
    if gated in ("manual", "auto"):
        api.update_repo_settings(repo_id, repo_type=repo_type, gated=gated)
        api.update_repo_settings(repo_id, repo_type=repo_type, private=False)
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}"


def _render(template: Path, **values) -> str:
    text = template.read_text(encoding="utf-8")
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def features():
    from datasets import Features, Image, Value

    return Features({"image": Image(), **{k: Value("bool" if k == "corrected" else "string") for k in COLUMNS}})


COLUMNS = [k for k in FIELDS if k != "split"]  # filename first, then the label provenance


def _to_dataset(rows: list[dict], images_dir: Path):
    from datasets import Dataset

    data = {"image": [{"bytes": (images_dir / r["filename"]).read_bytes(), "path": r["filename"]} for r in rows]}
    for k in COLUMNS:
        data[k] = [r[k] == "true" if k == "corrected" else r[k] for r in rows]
    return Dataset.from_dict(data, features=features())


def push_dataset(repo_id: str, labels_csv: Path, images_dir: Path, corrections: Path, splits: Path,
                 gated: str = "manual", token: str | None = None, dry_run: bool = False) -> str:
    with labels_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage / "data").mkdir()
        counts = {}
        for split in HUB_SPLITS:
            part = [r for r in rows if r["split"] == split]
            counts[split] = len(part)
            ds = _to_dataset(part, images_dir)
            ds.to_parquet(str(stage / "data" / f"{split}-00000-of-00001.parquet"))
        (stage / "provenance").mkdir()
        shutil.copy2(corrections, stage / "provenance" / "corrections.csv")
        shutil.copy2(splits, stage / "provenance" / "splits.csv")
        n_corr = sum(1 for _ in corrections.open(encoding="utf-8")) - 1
        n_human = sum(1 for r in rows if r["rucaptcha_label"])
        n_fixed = sum(1 for r in rows if r["corrected"] == "true")
        (stage / "README.md").write_text(_render(
            CARDS / "dataset_card.md", n_train=counts["train"], n_val=counts["validation"],
            n_unlabeled=counts["unlabeled"], n_total=len(rows), n_corrections=n_corr,
            n_corrected=n_fixed, n_human=n_human, pct_corrected=f"{n_fixed / max(n_human, 1):.1%}"), encoding="utf-8")
        if dry_run:
            out = Path("work/hub_dataset_preview")
            shutil.rmtree(out, ignore_errors=True)
            shutil.copytree(stage, out)
            return f"dry run: staged in {out} ({counts})"
        return _publish(_api(token), repo_id, "dataset", stage, gated, "Upload captcha dataset")


def push_model(repo_id: str, onnx_dir: Path, metrics: Path | None = None, dataset_repo: str = "",
               gated: str = "manual", token: str | None = None, dry_run: bool = False) -> str:
    missing = [n for n in ("model.onnx", "dict.txt") if not (onnx_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"{onnx_dir} has no {', '.join(missing)}: run `c2r export onnx` first "
                                "or pass --model <onnx dir>")
    if not (metrics and metrics.exists()):
        raise FileNotFoundError(f"no metrics file {metrics}: run `c2r evaluate` first (the model card "
                                "shows these numbers)")
    m = json.loads(metrics.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ("model.onnx", "dict.txt", "inference.yml"):
            if (onnx_dir / name).exists():
                shutil.copy2(onnx_dir / name, stage / name)
        if m:
            (stage / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")
        (stage / "README.md").write_text(_render(
            CARDS / "model_card.md", model_repo=repo_id, dataset_repo=dataset_repo or "(dataset repo)",
            seq_acc=f"{m.get('sequence_accuracy', float('nan')):.1%}",
            char_acc=f"{m.get('char_accuracy', float('nan')):.1%}",
            n_eval=m.get("n", "?"), latency=f"{m.get('ms_per_image', float('nan')):.1f}"), encoding="utf-8")
        if dry_run:
            out = Path("work/hub_model_preview")
            shutil.rmtree(out, ignore_errors=True)
            shutil.copytree(stage, out)
            return f"dry run: staged in {out}"
        return _publish(_api(token), repo_id, "model", stage, gated, "Upload ONNX captcha recognizer")


def pull_dataset(repo_id: str, data_dir: Path, token: str | None = None) -> dict:
    """Download the gated dataset (after your access request is approved) into the local layout."""
    from datasets import Image, load_dataset

    ds = load_dataset(repo_id, token=token)
    images = data_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in HUB_SPLITS:
        if split not in ds:
            continue
        part = ds[split].cast_column("image", Image(decode=False))  # original JPEG bytes, no re-encode
        for r in part:
            (images / r["filename"]).write_bytes(r["image"]["bytes"])
            rows.append({**{k: r[k] for k in COLUMNS}, "split": split,
                         "corrected": "true" if r["corrected"] else "false"})
    rows.sort(key=lambda r: r["filename"])
    with (data_dir / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    # Rebuild the raw inputs so `c2r data merge` reproduces labels.csv from scratch.
    (data_dir / "labels").mkdir(exist_ok=True)
    with (data_dir / "labels" / "rucaptcha.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "rucaptcha_raw", "rucaptcha_label", "drop_reason"])
        for r in rows:
            if r["rucaptcha_raw"]:
                w.writerow([r["filename"], r["rucaptcha_raw"], r["rucaptcha_label"],
                            normalize(r["rucaptcha_raw"])[1]])
    with (data_dir / "labels" / "llm.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "llm_label", "llm_confidence"])
        for r in rows:
            if r["llm_label"]:
                w.writerow([r["filename"], r["llm_label"], r["llm_confidence"]])
    return {"images": len(rows)}
