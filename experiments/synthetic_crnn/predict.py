"""Predict labels for cap_data/ with model.pt -> predictions_model.csv.

Usage: python3 predict.py [--eval labels.csv]
--greedy: allow any length (default forces 6 chars).
--eval: CSV with filename,label (e.g. data/labels.csv) to score against.
"""
import argparse
import csv
import os

import numpy as np
import torch

from train import CRNN, ROOT, decode, decode_fixed, load_gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(ROOT, "cap_data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "predictions_model.csv"))
    ap.add_argument("--eval")
    ap.add_argument("--model", default=os.path.join(ROOT, "model.pt"))
    ap.add_argument("--greedy", action="store_true", help="free-length greedy decoding")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CRNN().to(dev)
    model.load_state_dict(torch.load(a.model, map_location=dev))
    model.eval()

    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith((".jpg", ".png")))
    preds = []
    with torch.no_grad():
        for i in range(0, len(files), 256):
            x = np.stack([load_gray(os.path.join(a.images, f)) for f in files[i:i + 256]])
            x = torch.from_numpy(x).to(dev).float().div(255).unsqueeze(1)
            preds += (decode if a.greedy else decode_fixed)(model(x).cpu())
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "confidence"])
        for fn, (p, c) in zip(files, preds):
            w.writerow([fn, p, f"{c:.4f}"])
    confs = np.array([c for _, c in preds])
    print(f"wrote {len(files)} predictions to {a.out}")
    print(f"length!=6: {sum(len(p) != 6 for p, _ in preds)}; "
          f"conf>0.9: {(confs > .9).mean():.1%}; conf<0.5: {(confs < .5).mean():.1%}")

    if a.eval:
        truth = {r["filename"]: r["label"] for r in csv.DictReader(open(a.eval))}
        pairs = [(p, truth[fn]) for fn, (p, _) in zip(files, preds) if fn in truth]
        seq = np.mean([p == t for p, t in pairs])
        ch = np.mean([sum(x == y for x, y in zip(p, t)) / max(len(t), 1) for p, t in pairs])
        print(f"eval on {len(pairs)} images: seq_acc {seq:.3f}, char_acc {ch:.3f}")


if __name__ == "__main__":
    main()
