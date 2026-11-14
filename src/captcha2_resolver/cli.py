"""`c2r` — one command per pipeline stage. Run `c2r <stage> --help` for options.

  c2r collect                       # 1. scrape captcha images (source from .env)
  c2r label rucaptcha --size 10     # 2. paid human labels (resumable runs)
  c2r label consolidate             #    runs -> data/labels/rucaptcha.csv
  c2r data merge                    # 3. sources + corrections + splits -> data/labels.csv
  c2r data validate                 #    sanity checks
  c2r train prepare / c2r train run # 4. PaddleOCR fine-tuning
  c2r export onnx                   # 5. Paddle -> ONNX
  c2r evaluate                      #    metrics on the frozen validation split
  c2r predict img.jpg ...           # 6. inference
  c2r hub push-dataset / push-model / pull-dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from captcha2_resolver import paths as P


def _env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    except ImportError:
        pass
    # An empty `HF_TOKEN=` line must not shadow the token saved by `hf auth login`.
    for name in ("HF_TOKEN", "RUCAPTCHA_API_KEY"):
        if os.environ.get(name) == "":
            del os.environ[name]

def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"error: set {name} in .env (see .env.example)")
    return value

# ---------------------------------------------------------------- handlers
def cmd_collect(a):
    from captcha2_resolver.collect import CollectConfig, run

    cfg = CollectConfig(url=_require_env("CAPTCHA_SOURCE_URL"), selector=_require_env("CAPTCHA_SELECTOR"),
                        out_dir=a.out, target=a.target, interval=a.interval)
    print(f"collected {run(cfg)} images")

def cmd_label_rucaptcha(a):

    from captcha2_resolver.label import rucaptcha as rc

    names = rc.pending(a.resume) if a.resume else rc.select_scope(a.images, a.runs, a.size)
    if a.dry_run or not names:
        print("\n".join(names) or "nothing to send")
        return
    # Check the key before create_run: a saved scope.json marks its images as sent for good.
    client = rc.RucaptchaClient(_require_env("RUCAPTCHA_API_KEY"))
    run_dir = a.resume or rc.create_run(names, a.runs)
    print("run:", run_dir)
    records = rc.solve_run(run_dir, a.images, client, timeout=a.timeout)
    ready = sum(r["status"] == "ready" for r in records)
    print(f"{ready}/{len(records)} solved; details in {run_dir / 'results.json'}")

def cmd_label_consolidate(a):
    from captcha2_resolver.label.consolidate import consolidate

    print(consolidate(a.runs, a.out))

def cmd_data_merge(a):
    from captcha2_resolver.data import merge, validate

    stats = merge.merge(a.images, a.rucaptcha, a.llm, a.corrections, a.splits, a.out)
    print(json.dumps(stats, indent=2))
    errors, warnings = validate.check(merge.load_labels(a.out), a.images)
    n_err = validate.report(errors, warnings, verbose=False)
    if n_err:
        sys.exit(f"{n_err} label errors: fix them in {a.corrections} (run `c2r data validate` for details)")

def cmd_data_validate(a):
    from captcha2_resolver.data import merge, validate

    rows = merge.load_labels(a.labels)
    print(validate.char_stats([r["label"] for r in rows]))
    sys.exit(1 if validate.report(*validate.check(rows, a.images)) else 0)

def cmd_train_prepare(a):
    from captcha2_resolver.train.paddle import TrainConfig, prepare

    cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, max_lr=a.max_lr, augment=not a.no_augment,
                      use_gpu=not a.cpu, num_workers=a.num_workers, eval_every=a.eval_every)
    prepare(a.labels, a.images, a.work, a.paddleocr, a.export, cfg)

def cmd_train_run(a):
    from captcha2_resolver.train.paddle import train

    print("exported to", train(a.work / "rec_config.yml", a.paddleocr, a.export, resume=a.resume))

def cmd_export_onnx(a):
    from captcha2_resolver.export.onnx import convert

    convert(a.src, a.out, opset=a.opset, check_image=a.check)

def cmd_evaluate(a):
    from captcha2_resolver.evaluate import evaluate

    m = evaluate(a.model, a.labels, a.images, a.split, out_dir=a.out)
    mistakes = m.pop("mistakes")
    print(json.dumps(m, indent=2))
    for x in mistakes:
        print(f"  {x['filename']}  label={x['label']}  pred={x['pred']}  ({x['confidence']})")

def cmd_predict(a):
    from captcha2_resolver.infer import CaptchaSolver

    solve = CaptchaSolver.from_hub(a.hub) if a.hub else CaptchaSolver(a.model)
    for img in a.images:
        text, conf = solve(img)
        print(f"{img}\t{text}\t{conf:.3f}")

def cmd_hub_push_dataset(a):
    from captcha2_resolver.data.hub import push_dataset

    repo = a.repo or _require_env("HF_DATASET_REPO")
    print(push_dataset(repo, a.labels, a.images, a.corrections, a.splits, gated=a.gated,
                       token=os.getenv("HF_TOKEN") or None, dry_run=a.dry_run))

def cmd_hub_push_model(a):
    from captcha2_resolver.data.hub import push_model

    repo = a.repo or _require_env("HF_MODEL_REPO")
    print(push_model(repo, a.model, a.metrics, dataset_repo=os.getenv("HF_DATASET_REPO", ""),
                     gated=a.gated, token=os.getenv("HF_TOKEN") or None, dry_run=a.dry_run))

def cmd_hub_pull_dataset(a):
    from captcha2_resolver.data.hub import pull_dataset

    print(pull_dataset(a.repo or _require_env("HF_DATASET_REPO"), a.data, token=os.getenv("HF_TOKEN") or None))

# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="c2r", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("collect", help="scrape captcha images")
    s.add_argument("--out", type=Path, default=P.IMAGES)
    s.add_argument("--target", type=int, default=2000, help="stop when the folder holds this many")
    s.add_argument("--interval", type=float, default=5.0, help="min seconds per request")
    s.set_defaults(fn=cmd_collect)

    lab = sub.add_parser("label", help="labeling").add_subparsers(dest="sub", required=True)
    s = lab.add_parser("rucaptcha", help="paid human labels (sends paid requests!)")
    s.add_argument("--images", type=Path, default=P.IMAGES)
    s.add_argument("--runs", type=Path, default=P.RUNS)
    s.add_argument("--size", type=int, default=10)
    s.add_argument("--resume", type=Path, help="resume a run folder")
    s.add_argument("--timeout", type=float, default=180)
    s.add_argument("--dry-run", action="store_true", help="list the scope, send nothing")
    s.set_defaults(fn=cmd_label_rucaptcha)
    s = lab.add_parser("consolidate", help="runs -> data/labels/rucaptcha.csv")
    s.add_argument("--runs", type=Path, default=P.RUNS)
    s.add_argument("--out", type=Path, default=P.RUCAPTCHA_LABELS)
    s.set_defaults(fn=cmd_label_consolidate)

    dat = sub.add_parser("data", help="dataset").add_subparsers(dest="sub", required=True)
    s = dat.add_parser("merge", help="build data/labels.csv")
    s.add_argument("--images", type=Path, default=P.IMAGES)
    s.add_argument("--rucaptcha", type=Path, default=P.RUCAPTCHA_LABELS)
    s.add_argument("--llm", type=Path, default=P.LLM_LABELS)
    s.add_argument("--corrections", type=Path, default=P.CORRECTIONS)
    s.add_argument("--splits", type=Path, default=P.SPLITS)
    s.add_argument("--out", type=Path, default=P.LABELS)
    s.set_defaults(fn=cmd_data_merge)
    s = dat.add_parser("validate", help="sanity checks on data/labels.csv")
    s.add_argument("--labels", type=Path, default=P.LABELS)
    s.add_argument("--images", type=Path, default=P.IMAGES)
    s.set_defaults(fn=cmd_data_validate)

    tr = sub.add_parser("train", help="PaddleOCR fine-tuning").add_subparsers(dest="sub", required=True)
    s = tr.add_parser("prepare", help="lists, dict, adapted weights, config")
    for flag, default in (("--labels", P.LABELS), ("--images", P.IMAGES), ("--work", P.WORK),
                          ("--paddleocr", Path("PaddleOCR")), ("--export", P.PADDLE_EXPORT)):
        s.add_argument(flag, type=Path, default=default)
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--max-lr", type=float, default=7e-4)
    s.add_argument("--num-workers", type=int, default=2)
    s.add_argument("--eval-every", type=int, default=50)
    s.add_argument("--no-augment", action="store_true")
    s.add_argument("--cpu", action="store_true")
    s.set_defaults(fn=cmd_train_prepare)
    s = tr.add_parser("run", help="train -> eval -> export")
    s.add_argument("--work", type=Path, default=P.WORK)
    s.add_argument("--paddleocr", type=Path, default=Path("PaddleOCR"))
    s.add_argument("--export", type=Path, default=P.PADDLE_EXPORT)
    s.add_argument("--resume", action="store_true")
    s.set_defaults(fn=cmd_train_run)

    ex = sub.add_parser("export", help="export").add_subparsers(dest="sub", required=True)
    s = ex.add_parser("onnx", help="Paddle export dir or .zip -> ONNX")
    s.add_argument("--src", type=Path, default=P.PADDLE_EXPORT)
    s.add_argument("--out", type=Path, default=P.ONNX_EXPORT)
    s.add_argument("--opset", type=int, default=14)
    s.add_argument("--check", type=Path, metavar="IMAGE", help="compare ONNX vs Paddle on one image")
    s.set_defaults(fn=cmd_export_onnx)

    s = sub.add_parser("evaluate", help="metrics on a split")
    s.add_argument("--model", type=Path, default=P.ONNX_EXPORT)
    s.add_argument("--labels", type=Path, default=P.LABELS)
    s.add_argument("--images", type=Path, default=P.IMAGES)
    s.add_argument("--split", default="validation")
    s.add_argument("--out", type=Path, default=P.ONNX_EXPORT, help="where metrics/mistakes are written")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("predict", help="read captchas")
    s.add_argument("images", nargs="+")
    s.add_argument("--model", type=Path, default=P.ONNX_EXPORT)
    s.add_argument("--hub", help="load model from a (gated) HF repo instead")
    s.set_defaults(fn=cmd_predict)

    hub = sub.add_parser("hub", help="Hugging Face Hub").add_subparsers(dest="sub", required=True)
    for name, fn in (("push-dataset", cmd_hub_push_dataset), ("push-model", cmd_hub_push_model)):
        s = hub.add_parser(name)
        s.add_argument("--repo", help="defaults to HF_DATASET_REPO / HF_MODEL_REPO")
        s.add_argument("--gated", choices=["manual", "auto", "private"], default="manual")
        s.add_argument("--dry-run", action="store_true", help="stage files locally, upload nothing")
        s.set_defaults(fn=fn)
    hub.choices["push-dataset"].add_argument("--labels", type=Path, default=P.LABELS)
    hub.choices["push-dataset"].add_argument("--images", type=Path, default=P.IMAGES)
    hub.choices["push-dataset"].add_argument("--corrections", type=Path, default=P.CORRECTIONS)
    hub.choices["push-dataset"].add_argument("--splits", type=Path, default=P.SPLITS)
    hub.choices["push-model"].add_argument("--model", type=Path, default=P.ONNX_EXPORT)
    hub.choices["push-model"].add_argument("--metrics", type=Path,
                                           default=P.ONNX_EXPORT / "metrics_validation.json")
    s = hub.add_parser("pull-dataset", help="download the gated dataset into data/")
    s.add_argument("--repo")
    s.add_argument("--data", type=Path, default=P.DATA)
    s.set_defaults(fn=cmd_hub_pull_dataset)
    return p

def main(argv: list[str] | None = None) -> None:
    _env()
    args = build_parser().parse_args(argv)
    args.fn(args)

if __name__ == "__main__":
    main()
