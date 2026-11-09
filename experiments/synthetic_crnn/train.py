"""Train a CNN + BiGRU + CTC captcha recognizer on synthetic/images.

Usage: python3 train.py [--epochs 20] [--batch 128]
Saves the best model (by synthetic val accuracy) to model.pt.
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CHARSET = "2345678abcdefgmnpwxy"
LENGTH = 6
H, W = 50, 300
ROOT = os.path.dirname(os.path.abspath(__file__))


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


class CRNN(nn.Module):
    def __init__(self, n_classes=len(CHARSET) + 1):
        super().__init__()

        def block(i, o, pool):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(pool))
        # 50x300 -> 25x150 -> 12x75 -> 6x75 -> 3x75
        self.cnn = nn.Sequential(block(1, 32, 2), block(32, 64, 2),
                                 block(64, 128, (2, 1)), block(128, 192, (2, 1)))
        self.rnn = nn.GRU(192 * 3, 192, num_layers=2, bidirectional=True,
                          batch_first=True, dropout=0.2)
        self.fc = nn.Linear(384, n_classes)

    def forward(self, x):  # x: (B,1,H,W) float in [0,1]
        f = self.cnn(x)                      # (B,C,3,T)
        f = f.permute(0, 3, 1, 2).flatten(2)  # (B,T,C*3)
        f, _ = self.rnn(f)
        return self.fc(f)                    # (B,T,classes); class 0 = blank


def decode(logits):
    """Greedy CTC decode. Returns list of (text, confidence)."""
    probs = logits.softmax(-1)
    best_p, best = probs.max(-1)
    out = []
    for seq, ps in zip(best.tolist(), best_p.tolist()):
        chars, conf, prev = [], 1.0, 0
        for c, p in zip(seq, ps):
            if c != 0 and c != prev:
                chars.append(CHARSET[c - 1])
                conf *= p
            prev = c
        out.append(("".join(chars), conf))
    return out


def decode_fixed(logits, length=LENGTH):
    """Best CTC path that collapses to exactly `length` chars (Viterbi over (count, last)).
    Returns list of (text, confidence) with confidence = path probability."""
    out = []
    K = logits.shape[-1]
    for lp in logits.log_softmax(-1).numpy():  # (T,K)
        T = lp.shape[0]
        dp = np.full((length + 1, K), -np.inf)
        dp[0, 0] = 0.0
        back = np.zeros((T, length + 1, K, 2), dtype=np.int16)  # prev (n, last)
        for t in range(T):
            new = np.full_like(dp, -np.inf)
            # blank: keep n, any last
            arg = dp.argmax(1)
            new[:, 0] = dp[np.arange(length + 1), arg] + lp[t, 0]
            back[t, :, 0, 0] = np.arange(length + 1)
            back[t, :, 0, 1] = arg
            # repeat of same char: keep n, last=c
            stay = dp[:, 1:] + lp[t, 1:]
            # new char c from any last != c, n-1 -> n
            order = np.argsort(-dp, axis=1)
            top1, top2 = order[:, 0], order[:, 1]
            for n in range(1, length + 1):
                prev = dp[n - 1]
                c = np.arange(1, K)
                src = np.where(top1[n - 1] == c, top2[n - 1], top1[n - 1])
                adv = prev[src] + lp[t, 1:]
                use_adv = adv > stay[n]
                new[n, 1:] = np.where(use_adv, adv, stay[n])
                back[t, n, 1:, 0] = np.where(use_adv, n - 1, n)
                back[t, n, 1:, 1] = np.where(use_adv, src, c)
            new[0, 1:] = -np.inf
            dp = new
        last = int(dp[length].argmax())
        score = dp[length, last]
        n, chars = length, []
        for t in range(T - 1, -1, -1):
            pn, pl = back[t, n, last]
            if last != 0 and (pn != n or pl != last):
                chars.append(CHARSET[last - 1])
            n, last = int(pn), int(pl)
        out.append(("".join(reversed(chars)), float(np.exp(score))))
    return out


def augment(x):
    """Batch augmentation on device: shift/scale, contrast, noise. x: (B,1,H,W)."""
    b = x.shape[0]
    theta = torch.zeros(b, 2, 3, device=x.device)
    s = 1 + (torch.rand(b, device=x.device) - .5) * .12
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s * (1 + (torch.rand(b, device=x.device) - .5) * .1)
    theta[:, 0, 1] = (torch.rand(b, device=x.device) - .5) * .06
    theta[:, 0, 2] = (torch.rand(b, device=x.device) - .5) * .12
    theta[:, 1, 2] = (torch.rand(b, device=x.device) - .5) * .12
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    # pad with white (MPS lacks "border" mode): invert, zero-pad, invert back
    x = 1 - F.grid_sample(1 - x, grid, padding_mode="zeros", align_corners=False)
    c = 1 + (torch.rand(b, 1, 1, 1, device=x.device) - .5) * .4
    x = (x - .5) * c + .5 + (torch.rand(b, 1, 1, 1, device=x.device) - .5) * .2
    x = x + torch.randn_like(x) * .03
    return x.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data", default=os.path.join(ROOT, "synthetic", "images"))
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)

    rows = list(csv.DictReader(open(os.path.join(a.data, "labels.csv"))))
    t0 = time.time()
    X = np.stack([load_gray(os.path.join(a.data, r["filename"])) for r in rows])
    Y = np.array([[CHARSET.index(c) + 1 for c in r["label"]] for r in rows], dtype=np.int64)
    print(f"loaded {len(rows)} images in {time.time() - t0:.0f}s, device={dev}", flush=True)
    n_val = 2000
    Xtr, Ytr, Xva, Yva = X[n_val:], Y[n_val:], X[:n_val], Y[:n_val]
    va_labels = [r["label"] for r in rows[:n_val]]

    model = CRNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = a.epochs * (len(Xtr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=.15)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        tot, t0 = 0.0, time.time()
        for i in range(len(Xtr) // a.batch):
            idx = perm[i * a.batch:(i + 1) * a.batch]
            x = torch.from_numpy(Xtr[idx]).to(dev).float().div(255).unsqueeze(1)
            y = torch.from_numpy(Ytr[idx]).to(dev)
            logits = model(augment(x))                       # (B,T,C)
            lp = logits.log_softmax(-1).permute(1, 0, 2).cpu()  # CTC on CPU (MPS lacks it)
            T = lp.shape[0]
            loss = ctc(lp, y.cpu(), torch.full((len(idx),), T, dtype=torch.long),
                       torch.full((len(idx),), LENGTH, dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, n_val, 500):
                x = torch.from_numpy(Xva[i:i + 500]).to(dev).float().div(255).unsqueeze(1)
                preds += [p for p, _ in decode(model(x).cpu())]
        acc = np.mean([p == t for p, t in zip(preds, va_labels)])
        cacc = np.mean([sum(a == b for a, b in zip(p, t)) / LENGTH for p, t in zip(preds, va_labels)])
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / (i + 1):.3f} val_seq_acc {acc:.4f} "
              f"val_char_acc {cacc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(ROOT, "model.pt"))
    print(f"best val_seq_acc {best:.4f}; saved model.pt")


if __name__ == "__main__":
    main()
