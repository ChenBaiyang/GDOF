"""Run GDOF experiments and save ROC-AUC and average precision to CSV."""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from gdof import GDOF


ROOT = Path(__file__).resolve().parent
FIELDS = ("dataset", "alpha", "n_neg", "n_pos", "seed", "auc", "pr")


def load_dataset(path):
    with np.load(path) as data:
        X, y = data["X"], data["y"]
        nominals = data["nominals"] if "nominals" in data else None
    if y.shape != (len(X),) or not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("y must contain normal (0) and outlier (1) labels, one per sample")
    return X, y, nominals


def labeled_outliers(y, n_pos, seed):
    positives = np.flatnonzero(y == 1)
    if not 0 < n_pos < len(positives):
        raise ValueError("n_pos must leave at least one outlier for evaluation")
    np.random.RandomState(seed).shuffle(positives)
    pos_idx = positives[:n_pos]
    return pos_idx, np.setdiff1d(np.arange(len(y)), pos_idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--datasets", nargs="+", help="Dataset names without .npz; default: all")
    parser.add_argument("--n-pos", nargs="+", type=int, default=[10, 15, 20, 25, 30])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--n-neg", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--no-normalize", action="store_true", help="Require pre-scaled numerical data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "results.csv")
    args = parser.parse_args()
    datasets = args.datasets or [p.stem for p in sorted(args.data_dir.glob("*.npz"))]
    if not datasets:
        raise ValueError("No .npz datasets found")
    total = len(datasets) * len(args.n_pos) * len(args.seeds)
    completed = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for dataset in datasets:
            X, y, nominals = load_dataset(args.data_dir / (dataset + ".npz"))
            for n_pos in args.n_pos:
                for seed in args.seeds:
                    pos_idx, test_idx = labeled_outliers(y, n_pos, seed)
                    model = GDOF(n_neg=args.n_neg, alpha=args.alpha, normalize=not args.no_normalize,
                                 device=args.device, batch_size=args.batch_size).fit(X, pos_idx, nominals)
                    scores = model.predict_score()[test_idx]
                    auc = float(roc_auc_score(y[test_idx], scores))
                    ap = float(average_precision_score(y[test_idx], scores))
                    writer.writerow(dict(zip(FIELDS, (dataset, args.alpha, args.n_neg, n_pos, seed, auc, ap))))
                    stream.flush()
                    completed += 1
                    print("[{}/{}] {} n_pos={} seed={} n_neg={} (requested {}) AUC={:.6f} AP={:.6f}".format(
                        completed, total, dataset, n_pos, seed, model.n_neg_, args.n_neg, auc, ap), flush=True)
    print("Results:", args.output)


if __name__ == "__main__":
    main()
