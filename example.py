"""Transductive and independent test-set examples for GDOF."""

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from gdof import GDOF


def known_outliers(y, n_pos=5):
    candidates = np.flatnonzero(y == 1)
    if len(candidates) <= n_pos:
        raise ValueError("Need more outliers than the known-outlier budget")
    return np.random.RandomState(0).permutation(candidates)[:n_pos]


def report(protocol, y, scores):
    print("{}: ROC-AUC={:.6f}, AP={:.6f}".format(
        protocol, roc_auc_score(y, scores), average_precision_score(y, scores)))


def main():
    # Replace this block with your X, binary y and optional nominals mask.
    with np.load(Path(__file__).resolve().parent / "data" / "breastw.npz") as data:
        X, y = data["X"], data["y"]
        nominals = data["nominals"] if "nominals" in data else None

    # Transductive: all feature vectors participate; only pos supplies labels.
    pos = known_outliers(y)
    model = GDOF().fit(X, pos, nominals)
    evaluation = np.setdiff1d(np.arange(len(X)), pos)
    report("Transductive", y[evaluation], model.predict_score()[evaluation])

    # Inductive extension: hold out test features before fitting preprocessing.
    order = np.random.RandomState(2).permutation(len(X))
    train, test = np.split(order, [int(0.75 * len(X))])
    model = GDOF().fit(X[train], known_outliers(y[train]), nominals)
    report("Inductive extension", y[test], model.predict_score(X[test]))


if __name__ == "__main__":
    main()
