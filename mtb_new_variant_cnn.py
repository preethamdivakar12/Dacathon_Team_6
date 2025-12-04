#!/usr/bin/env python3
"""
Mycobacterium tuberculosis (MTB) strain classifier + new-variant detector
=======================================================================

This script implements a training and inference pipeline inspired by
"COVID-19 Genome Sequence Analysis for New Variant Prediction and Generation"
(Ullah et al., 2022), adapted for long (~4 Mbp) MTB genomes.

What it does
------------
1) Loads FASTA files (one per strain or per sample) and turns genomes into fixed-length windows.
2) Encodes nucleotides with one-hot over A,C,G,T,N (N -> all zeros).
3) Trains a 1D-CNN + Self-Attention classifier over windows.
4) Aggregates window predictions across a whole genome to get per-genome class probabilities.
5) Computes Shannon entropy of the probability vector for novelty detection.

Inputs
------
- A directory of FASTA files (e.g., 8 strain references or sample genomes):
    data/
      H37Rv.fasta
      CDC1551.fasta
      ...
- A labels CSV mapping file names to class names (strain IDs):
    labels.csv with columns: filename,class

Outputs
-------
- Trained checkpoint (*.pt)
- Metrics JSON and CSV
- Inference CSV with per-sample probabilities, predicted class, and entropy

Usage
-----
Train:
    python mtb_new_variant_cnn.py train \
        --fasta_dir data/train \
        --labels labels_train.csv \
        --val_fasta_dir data/val \
        --val_labels labels_val.csv \
        --out_dir runs/exp1 \
        --window_len 4096 --stride 2048 \
        --epochs 20 --batch_size 32 --lr 3e-4

Test / inference (includes novelty detection):
    python mtb_new_variant_cnn.py infer \
        --fasta_dir data/test \
        --labels labels_test.csv \
        --checkpoint runs/exp1/model.pt \
        --out runs/exp1/test_preds.csv \
        --entropy_threshold 1.3

Notes
-----
- For 4 Mbp genomes, windows of 2–8 kbp with 50% overlap work well.
- Entropy threshold: tune on a holdout set mixing known classes and any out-of-class genomes.
- This code uses PyTorch and runs on CPU or GPU.

"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------
# Utils
# ---------------

NUC2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX2NUC = {0: "A", 1: "C", 2: "G", 3: "T"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_fasta(fp: Path) -> str:
    """Read FASTA file and return concatenated uppercase sequence (no headers)."""
    seq_parts: List[str] = []
    with open(fp, "r") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                continue
            seq_parts.append(line.strip())
    return ("".join(seq_parts)).upper()


def one_hot_encode(seq: str, window_len: int) -> np.ndarray:
    """One-hot encode a sequence chunk of length window_len into shape (4, L).
    Unknown/ambiguous bases -> zeros column.
    """
    L = len(seq)
    if L != window_len:
        # Pad with N if shorter (zeros), trim if longer
        if L < window_len:
            seq = seq + ("N" * (window_len - L))
        else:
            seq = seq[:window_len]
        L = window_len
    arr = np.zeros((4, L), dtype=np.float32)
    for i, ch in enumerate(seq):
        j = NUC2IDX.get(ch, None)
        if j is not None:
            arr[j, i] = 1.0
    return arr

# add near the top of the file, beside other utils
def random_snp(seq: str, rate: float = 0.002) -> str:
    """Introduce random single-nucleotide substitutions at given rate."""
    import random
    bases = "ACGT"
    seq_list = list(seq)
    n = len(seq_list)
    # number of mutations ~ Binomial(n, rate)
    k = max(0, int(n * rate))
    idxs = random.sample(range(n), k=min(k, n))
    for i in idxs:
        old = seq_list[i]
        if old not in bases:
            continue
        choices = [b for b in bases if b != old]
        seq_list[i] = random.choice(choices)
    return "".join(seq_list)

def sliding_windows(seq: str, window_len: int, stride: int) -> List[str]:
    windows = []
    for start in range(0, max(1, len(seq) - window_len + 1), stride):
        windows.append(seq[start : start + window_len])
    if len(seq) < window_len:  # ensure at least 1
        windows.append(seq)
    return windows


# ---------------
# Dataset
# ---------------

@dataclass
class SampleSpec:
    fasta_path: Path
    label: int
    name: str


class GenomeWindowDataset(Dataset):
    def __init__(
        self,
        samples: List[SampleSpec],
        classes: List[str],
        window_len: int = 4096,
        stride: int = 2048,
        max_windows_per_genome: Optional[int] = None,
        augment_rc: bool = True,
    ):
        self.samples = samples
        self.classes = classes
        self.window_len = window_len
        self.stride = stride
        self.max_windows_per_genome = max_windows_per_genome
        self.augment_rc = augment_rc
        # Pre-index windows (lazy could be done to save RAM, but index helps balancing)
        self.index: List[Tuple[int, int]] = []  # (sample_idx, window_idx)
        self.genome_windows: List[List[str]] = []
        for si, s in enumerate(samples):
            seq = read_fasta(s.fasta_path)
            wins = sliding_windows(seq, window_len, stride)
            if max_windows_per_genome and len(wins) > max_windows_per_genome:
                # uniform subsample
                idxs = sorted(np.random.choice(len(wins), max_windows_per_genome, replace=False))
                wins = [wins[i] for i in idxs]
            self.genome_windows.append(wins)
            for wi in range(len(wins)):
                self.index.append((si, wi))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, wi = self.index[i]
        s = self.samples[si]
        seq = self.genome_windows[si][wi]
        if self.augment_rc and random.random() < 0.5:
            # reverse-complement augmentation
            rc_map = str.maketrans("ACGT", "TGCA")
            seq = seq.translate(rc_map)[::-1]

        if self.augment_rc and random.random() < 0.7:   # 70% of batches ##
            seq = random_snp(seq, rate=0.002)   ##
        
        x = one_hot_encode(seq, self.window_len)  # (4, L)
        y = s.label
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ---------------
# Model: 1D-CNN + Self-Attention (scaled dot-product)
# ---------------

class SelfAttention1D(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.k = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.v = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        # x: (B, C=d_model, L)
        B, C, L = x.shape
        q = self.q(x).view(B, self.n_heads, self.d_k, L)
        k = self.k(x).view(B, self.n_heads, self.d_k, L)
        v = self.v(x).view(B, self.n_heads, self.d_k, L)
        # attention scores: (B, heads, L, L)
        scores = torch.einsum("bhdl,bhdm->bhlm", q, k) / math.sqrt(self.d_k)
        attn = scores.softmax(dim=-1)
        ctx = torch.einsum("bhlm,bhdm->bhdl", attn, v).contiguous()
        ctx = ctx.view(B, C, L)
        out = self.proj(ctx)
        return out


class AttentionCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        # Input channels = 4 (A,C,G,T)
        self.conv1 = nn.Conv1d(4, 128, kernel_size=4, padding=0)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=4)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 16, kernel_size=4)
        self.bn4 = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.attn = SelfAttention1D(d_model=16, n_heads=4)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        # x: (B, 4, L)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.attn(x)  # (B, 16, L')
        x = self.dropout(x)
        x = x.mean(dim=-1)  # global average pooling over length
        logits = self.fc(x)
        return logits


# ---------------
# Training & Evaluation
# ---------------

def load_labels(labels_csv: Path) -> Dict[str, str]:
    mapping = {}
    with open(labels_csv, "r") as f:
        header = f.readline().strip().split(",")
        if header[:2] != ["filename", "class"]:
            raise ValueError("labels CSV must have header: filename,class")
        for line in f:
            if not line.strip():
                continue
            fname, cls = line.strip().split(",")[:2]
            mapping[fname] = cls
    return mapping


# def build_samples(fasta_dir: Path, labels_csv: Path, class_to_idx: Dict[str, int]) -> List[SampleSpec]:
    # labels = load_labels(labels_csv)
    # samples: List[SampleSpec] = []
    # for fname, cls in labels.items():
        # fp = fasta_dir / fname
        # if not fp.exists():
            # raise FileNotFoundError(fp)
        # samples.append(SampleSpec(fasta_path=fp, label=class_to_idx[cls], name=fname))
    # return samples  
def build_samples(
    fasta_dir: Path,
    labels_csv: Path,
    class_to_idx: Dict[str, int],
    ignore_unknown: bool = False
) -> List[SampleSpec]:
    labels = load_labels(labels_csv)
    samples: List[SampleSpec] = []
    for fname, cls in labels.items():
        fp = fasta_dir / fname
        if not fp.exists():
            raise FileNotFoundError(fp)
        if cls not in class_to_idx:
            if ignore_unknown:
                # Skip unknown validation samples (like synthetic_strain_1)
                continue
            else:
                raise KeyError(cls)
        samples.append(SampleSpec(fasta_path=fp, label=class_to_idx[cls], name=fname))
    return samples


def train(args):
    set_seed(args.seed)
    # Build class index
    train_labels = load_labels(Path(args.labels))
    classes = sorted(list(set(train_labels.values())))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Datasets
    # train_samples = build_samples(Path(args.fasta_dir), Path(args.labels), class_to_idx)
    # val_samples = build_samples(Path(args.val_fasta_dir), Path(args.val_labels), class_to_idx)
    train_samples = build_samples(Path(args.fasta_dir), Path(args.labels), class_to_idx, ignore_unknown=False)
    val_samples = build_samples(Path(args.val_fasta_dir), Path(args.val_labels), class_to_idx, ignore_unknown=True)


    train_ds = GenomeWindowDataset(
        train_samples,
        classes,
        window_len=args.window_len,
        stride=args.stride,
        max_windows_per_genome=args.max_windows,
        augment_rc=not args.no_rc_aug,
    )
    val_ds = GenomeWindowDataset(
        val_samples,
        classes,
        window_len=args.window_len,
        stride=args.stride,
        max_windows_per_genome=args.val_max_windows,
        augment_rc=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = AttentionCNN(n_classes=len(classes)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)##


    best_val_acc = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        sched.step()
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        # Validation
        model.eval()
        v_total, v_correct, v_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss_sum += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                v_correct += (preds == yb).sum().item()
                v_total += xb.size(0)
        val_acc = v_correct / max(1, v_total)
        val_loss = v_loss_sum / max(1, v_total)

        print(f"Epoch {epoch:03d} | train_acc={train_acc:.4f} loss={train_loss:.4f} | val_acc={val_acc:.4f} loss={val_loss:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "model_state": model.state_dict(),
                "classes": classes,
                "args": vars(args),
                "epoch": epoch,
                "val_acc": val_acc,
            }
            torch.save(ckpt, out_dir / "model.pt")

    # Save final
    ckpt = {
        "model_state": model.state_dict(),
        "classes": classes,
        "args": vars(args),
        "epoch": args.epochs,
        "val_acc": best_val_acc,
    }
    torch.save(ckpt, out_dir / "last.pt")


# ---------------
# Inference & Novelty Detection
# ---------------

def shannon_entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    classes: List[str] = ckpt["classes"]
    model = AttentionCNN(n_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])  # type: ignore
    model.to(device)
    model.eval()

    # labels may be absent at inference; if provided, use for accuracy calc
    # class_to_idx = {c: i for i, c in enumerate(classes)}
    # samples: List[SampleSpec] = []
    # if args.labels:
        # samples = build_samples(Path(args.fasta_dir), Path(args.labels), class_to_idx)
    # else:
        # label -1 for unknown
        # for fp in sorted(Path(args.fasta_dir).glob("*.fa*")):
            # samples.append(SampleSpec(fasta_path=fp, label=-1, name=fp.name))

    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples: List[SampleSpec] = []
    if args.labels:
        labels_map = load_labels(Path(args.labels))
        for fname, cls in labels_map.items():
            fp = Path(args.fasta_dir) / fname
            if not fp.exists():
                raise FileNotFoundError(fp)
            label_idx = class_to_idx.get(cls, -1)  # unknown labels (e.g., synthetic_strain_1) -> -1
            samples.append(SampleSpec(fasta_path=fp, label=label_idx, name=fname))
    else:
        for fp in sorted(Path(args.fasta_dir).glob("*.fa*")):
            samples.append(SampleSpec(fasta_path=fp, label=-1, name=fp.name))
    
    window_len = args.window_len
    stride = args.stride

    out_rows = [
        ["filename", "true_class", "pred_class", "entropy", "is_new"] + [f"p_{c}" for c in classes]
    ]

    correct, total = 0, 0

    with torch.no_grad():
        for s in samples:
            seq = read_fasta(s.fasta_path)
            wins = sliding_windows(seq, window_len, stride)
            # Batch windows
            probs_all: List[np.ndarray] = []
            for i in range(0, len(wins), args.batch_size):
                batch_seqs = wins[i : i + args.batch_size]
                xb = np.stack([one_hot_encode(w, window_len) for w in batch_seqs])
                xb_t = torch.from_numpy(xb).to(device)
                logits = model(xb_t)
                probs = logits.softmax(dim=1).cpu().numpy()
                probs_all.append(probs)
            probs_all = np.vstack(probs_all)
            # Aggregate across windows (mean)
            genome_probs = probs_all.mean(axis=0)
            ent = shannon_entropy(genome_probs)
            pred_idx = int(genome_probs.argmax())
            pred_class = classes[pred_idx]
            is_new = 1 if ent > args.entropy_threshold else 0

            true_class = "" if s.label == -1 else classes[s.label]
            if s.label != -1:
                total += 1
                # If flagged as new, count as incorrect for standard accuracy
                if not is_new and pred_idx == s.label:
                    correct += 1

            out_rows.append(
                [
                    s.name,
                    true_class,
                    "NEW" if is_new else pred_class,
                    f"{ent:.6f}",
                    is_new,
                    *[f"{p:.6f}" for p in genome_probs.tolist()],
                ]
            )

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in out_rows:
            f.write(",".join(map(str, row)) + "\n")

    if total > 0:
        acc = correct / max(1, total)
        print(f"Genome-level accuracy (ignoring NEW flags as correct): {acc:.4f} on {total} labeled samples")
    print(f"Wrote predictions to {out_path}")


# ---------------
# CLI
# ---------------

def build_argparser():
    p = argparse.ArgumentParser(description="MTB variant CNN + novelty detector")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train
    t = sub.add_parser("train", help="train the model")
    t.add_argument("--fasta_dir", required=True)
    t.add_argument("--labels", required=True)
    t.add_argument("--val_fasta_dir", required=True)
    t.add_argument("--val_labels", required=True)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--window_len", type=int, default=4096)
    t.add_argument("--stride", type=int, default=2048)
    t.add_argument("--max_windows", type=int, default=2000, help="max windows per genome for training")
    t.add_argument("--val_max_windows", type=int, default=4000)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--cpu", action="store_true")
    t.add_argument("--no_rc_aug", action="store_true", help="disable reverse-complement augmentation")

    # Infer
    i = sub.add_parser("infer", help="run inference & novelty detection")
    i.add_argument("--fasta_dir", required=True)
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--labels", default="", help="optional labels CSV for accuracy calc")
    i.add_argument("--window_len", type=int, default=4096)
    i.add_argument("--stride", type=int, default=2048)
    i.add_argument("--batch_size", type=int, default=64)
    i.add_argument("--entropy_threshold", type=float, default=1.3, help="Shannon entropy threshold for NEW")
    i.add_argument("--cpu", action="store_true")

    return p


# ---------------
# Extra Utilities: dataset split, labels creation, entropy calibration
# ---------------

def make_splits(args):
    """Create train/val/test splits and labels CSVs from a directory organized as:
    root/
      classA/*.fasta
      classB/*.fasta
      ...
    Splits are stratified per class with ratios.
    """
    root = Path(args.root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows, test_rows = [], [], []
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        files = sorted([p for p in cls_dir.glob("*.fa*")])
        if not files:
            continue
        rnd = random.Random(args.seed)
        rnd.shuffle(files)
        n = len(files)
        n_train = max(1, int(n * args.train_ratio))
        n_val = max(1, int(n * args.val_ratio))
        n_test = max(0, n - n_train - n_val)
        parts = [
            (files[:n_train], train_rows, 'train'),
            (files[n_train:n_train+n_val], val_rows, 'val'),
            (files[n_train+n_val:], test_rows, 'test'),
        ]
        for fs, acc, split in parts:
            split_dir = out / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for fp in fs:
                dst = split_dir / fp.name
                if fp.resolve() != dst.resolve():
                    # copy lazily to avoid huge duplication; symlink if supported
                    try:
                        if dst.exists():
                            dst.unlink()
                        os.symlink(fp.resolve(), dst)
                    except Exception:
                        import shutil
                        shutil.copy2(fp, dst)
                acc.append((fp.name, cls))
    for split, rows in [('train', train_rows), ('val', val_rows), ('test', test_rows)]:
        with open(out / f"labels_{split}.csv", 'w') as f:
            f.write("filename,class\n")


            for fn, cls in rows:
                f.write(f"{fn},{cls}")
    print(f"Wrote splits to {out}. Counts: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")


def calibrate_entropy(args):
    """Run inference on a validation set (known classes) to compute genome-level
    entropy distribution and suggest thresholds (e.g., 95th percentile)."""
    import csv
    import statistics as stats
    # Reuse infer on val set, capture entropies
    tmp_out = Path(args.out)
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    # Build call-like behavior
    class Simple: pass
    S = Simple()
    S.fasta_dir = args.fasta_dir
    S.checkpoint = args.checkpoint
    S.out = str(tmp_out)
    S.labels = args.labels
    S.window_len = args.window_len
    S.stride = args.stride
    S.batch_size = args.batch_size
    S.entropy_threshold = 9999.0  # disable new flag for collection
    S.cpu = args.cpu
    infer(S)
    entropies = []
    with open(tmp_out, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entropies.append(float(row['entropy']))
    if not entropies:
        print("No entropies computed.")
        return
    entropies.sort()
    def perc(p):
        k = max(0, min(len(entropies)-1, int(round(p/100.0*(len(entropies)-1)))))
        return entropies[k]
    suggest_90 = perc(90)
    suggest_95 = perc(95)
    suggest_99 = perc(99)
    mean_e = stats.mean(entropies)
    std_e = stats.pstdev(entropies)
    print(f"Entropy stats on validation set: n={len(entropies)} mean={mean_e:.4f} std={std_e:.4f}")
    print(f"Suggested thresholds (higher flags NEW): P90={suggest_90:.4f} P95={suggest_95:.4f} P99={suggest_99:.4f}")


def add_util_args(parser):
    sub = parser.add_subparsers(dest="cmd", required=True)
    # existing subparsers will be re-added here


def build_argparser():
    p = argparse.ArgumentParser(description="MTB variant CNN + novelty detector")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train
    t = sub.add_parser("train", help="train the model")
    t.add_argument("--fasta_dir", required=True)
    t.add_argument("--labels", required=True)
    t.add_argument("--val_fasta_dir", required=True)
    t.add_argument("--val_labels", required=True)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--window_len", type=int, default=4096)
    t.add_argument("--stride", type=int, default=2048)
    t.add_argument("--max_windows", type=int, default=2000, help="max windows per genome for training")
    t.add_argument("--val_max_windows", type=int, default=4000)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--cpu", action="store_true")
    t.add_argument("--no_rc_aug", action="store_true", help="disable reverse-complement augmentation")

    # Infer
    i = sub.add_parser("infer", help="run inference & novelty detection")
    i.add_argument("--fasta_dir", required=True)
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--labels", default="", help="optional labels CSV for accuracy calc")
    i.add_argument("--window_len", type=int, default=4096)
    i.add_argument("--stride", type=int, default=2048)
    i.add_argument("--batch_size", type=int, default=64)
    i.add_argument("--entropy_threshold", type=float, default=1.3, help="Shannon entropy threshold for NEW")
    i.add_argument("--cpu", action="store_true")

    # Make splits
    s = sub.add_parser("make_splits", help="create train/val/test splits and labels CSVs from class-folders")
    s.add_argument("--root", required=True, help="root dir with class subfolders containing FASTA files")
    s.add_argument("--out_dir", required=True, help="output dir where splits and labels_*.csv are written")
    s.add_argument("--train_ratio", type=float, default=0.7)
    s.add_argument("--val_ratio", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=42)

    # Calibrate entropy threshold on validation set
    c = sub.add_parser("calibrate", help="suggest entropy thresholds from validation entropies")
    c.add_argument("--fasta_dir", required=True)
    c.add_argument("--labels", required=True)
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--out", required=True, help="temp CSV path to collect entropies")
    c.add_argument("--window_len", type=int, default=4096)
    c.add_argument("--stride", type=int, default=2048)
    c.add_argument("--batch_size", type=int, default=64)
    c.add_argument("--cpu", action="store_true")

    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "infer":
        infer(args)
    elif args.cmd == "make_splits":
        make_splits(args)
    elif args.cmd == "calibrate":
        calibrate_entropy(args)
