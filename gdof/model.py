"""Granule Density-based Outlier Factor with blockwise CPU/CUDA computation."""

import numpy as np
import torch

from .preprocessing import Preprocessor


def _learn_radius(positive, negative):
    """Maximize the difference of negative and positive cumulative similarity."""
    candidates = np.arange(0.999, 0, -0.001).round(4)
    averages = []
    for matrix in (positive, negative):
        values = -np.sort(-matrix, axis=None)
        values[-1] = 0
        cumulative = np.cumsum(values)
        indices = np.searchsorted(-values, -candidates, side="right")
        # Keep normalization consistent across NumPy 1.x and 2.x.
        dtype = np.result_type(values.dtype, np.min_scalar_type(len(values)))
        averages.append(cumulative[indices - 1].astype(dtype) / np.array(len(values), dtype=dtype))
    return float(round(1 - candidates[np.argmax(averages[1] - averages[0])], 3))


class GDOF:
    """Label-informed outlier detection; larger scores indicate outliers.

    n_neg: requested pseudo-negatives; alpha: negative density weight.
    normalize: learn numerical min-max scales from fitting data.
    random_state: pseudo-negative sampling seed.
    device: 'auto' (CUDA first), 'cpu', or a Torch CUDA device string.
    batch_size: relation rows per block, without changing neighborhoods.
    """

    def __init__(self, n_neg=200, alpha=0.5, normalize=True, random_state=1,
                 device="auto", batch_size=1024):
        self.n_neg = n_neg
        self.alpha = alpha
        self.normalize = normalize
        self.random_state = random_state
        self.device = device
        self.batch_size = batch_size
        self._fitted = False

    def _blocks(self, n):
        for start in range(0, n, self.batch_size):
            yield slice(start, min(start + self.batch_size, n))

    def _relation(self, values, feature, radius=None):
        reference = self._reference[:, feature]
        if self.nominals_[feature]:
            relation = (values[:, None] == reference[None, :]).to(torch.float32)
        else:
            relation = (1 - torch.abs(values[:, None] - reference[None, :])).clamp(min=0)
            if radius is not None:
                relation[relation < 1 - radius] = 0
        return relation

    @torch.no_grad()
    def fit(self, X, pos_idx, nominals=None):
        """Fit X using known-outlier indices; return self.

        nominals is a bool list/array over original columns. If omitted,
        columns with 2–5 distinct integer values are categorical.
        """
        self._fitted = False
        for name in ("n_neg", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        device = self.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device_ = torch.device(device)
        self.preprocessor_ = Preprocessor(self.normalize)
        prepared = self.preprocessor_.fit_transform(X, nominals)
        n, m = prepared.shape
        positive = np.asarray(pos_idx)
        if positive.ndim != 1 or positive.dtype.kind not in "iu" or len(positive) == 0:
            raise ValueError("pos_idx must be a nonempty one-dimensional array of integer indices")
        if (positive < 0).any() or (positive >= n).any() or len(np.unique(positive)) != len(positive):
            raise ValueError("pos_idx must contain unique indices within X")
        self.pos_idx_ = positive.astype(np.int64, copy=True)
        unlabeled = np.setdiff1d(np.arange(n), self.pos_idx_)
        self.n_neg_ = min(self.n_neg, int(len(unlabeled) * 0.8))
        if self.n_neg_ == 0:
            raise ValueError("At least two unlabeled samples are required")
        self.n_features_in_ = self.preprocessor_.n_features_in_
        self.feature_mask_ = self.preprocessor_.feature_mask_.copy()
        self.nominals_in_ = self.preprocessor_.nominals_in_.copy()
        self.nominals_ = self.preprocessor_.nominals_.copy()
        self._reference = torch.as_tensor(prepared, dtype=torch.float32, device=self.device_)

        # Symmetry permits row sums; float64 accumulation is independent of row blocks.
        similarities = torch.zeros(n, dtype=torch.float64, device=self.device_)
        for feature in range(m):
            for block in self._blocks(n):
                relation = self._relation(self._reference[block, feature], feature)
                similarities[block] += relation.sum(dim=1, dtype=torch.float64)
        mean_similarity = (similarities / (m * n)).to(torch.float32)
        pool = torch.as_tensor(unlabeled, dtype=torch.long, device=self.device_)
        probabilities = torch.softmax(mean_similarity[pool].to(torch.float64), dim=0).cpu().numpy()
        rng = np.random.RandomState(self.random_state)
        self.neg_idx_ = rng.choice(unlabeled, self.n_neg_, replace=False, p=probabilities)
        pos = torch.as_tensor(self.pos_idx_, dtype=torch.long, device=self.device_)
        neg = torch.as_tensor(self.neg_idx_, dtype=torch.long, device=self.device_)
        self.lambdas_ = np.zeros(m, dtype=np.float64)
        self.row_sums_ = torch.empty((m, n), dtype=torch.float32, device=self.device_)
        self.densities_ = torch.empty_like(self.row_sums_)
        for feature in range(m):
            if not self.nominals_[feature]:
                selected = []
                for indices in (pos, neg):
                    rows = [self._relation(self._reference[indices[block], feature], feature).cpu().numpy()
                            for block in self._blocks(len(indices))]
                    selected.append(np.concatenate(rows, axis=0))
                self.lambdas_[feature] = _learn_radius(*selected)
            radius = self.lambdas_[feature]
            for block in self._blocks(n):
                relation = self._relation(self._reference[block, feature], feature, radius)
                self.row_sums_[feature, block] = relation.sum(dim=1)
            for block in self._blocks(n):
                self.densities_[feature, block] = self._density(self._reference[block, feature], feature)
        density = self.densities_.T
        self.weights_ = self.alpha * density[neg].mean(dim=0) - (1 - self.alpha) * density[pos].mean(dim=0)
        self.decision_scores_ = self._scores(self.densities_).cpu().numpy().copy()
        self._fitted = True
        return self

    def _density(self, values, feature):
        relation = self._relation(values, feature, self.lambdas_[feature])
        neighbors = (relation > 0).to(torch.float32)
        row_sum = relation.sum(dim=1)
        denominator = torch.matmul(neighbors, self.row_sums_[feature])
        density = row_sum.square() * neighbors.mean(dim=1)
        # New queries with no reference neighbors have zero density.
        return torch.where(denominator > 0, density / denominator.clamp(min=torch.finfo(torch.float32).tiny),
                           torch.zeros_like(density))

    def _scores(self, densities):
        return 1 - (densities.T * self.weights_).mean(dim=1)

    def fit_score(self, X, pos_idx, nominals=None):
        """Fit and return scores for all rows of X."""
        return self.fit(X, pos_idx, nominals).predict_score()

    @torch.no_grad()
    def predict_score(self, X=None):
        """Return fitted scores, or score queries against the frozen reference."""
        if not self._fitted:
            raise RuntimeError("Call fit before predict_score")
        if X is None:
            return self.decision_scores_.copy()
        prepared = self.preprocessor_.transform(X)
        result = np.empty(len(prepared), dtype=np.float32)
        for block in self._blocks(len(prepared)):
            query = torch.as_tensor(prepared[block], dtype=torch.float32, device=self.device_)
            densities = torch.stack([self._density(query[:, feature], feature)
                                     for feature in range(len(self.nominals_))])
            result[block] = self._scores(densities).cpu().numpy()
        return result
