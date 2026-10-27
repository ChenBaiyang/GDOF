"""Training-only preprocessing; categorical columns are never scaled."""

import numpy as np


def _array(X):
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be a two-dimensional NumPy array with at least one feature")
    if X.dtype.kind not in "biuf" or not np.isfinite(X).all():
        raise ValueError("X must contain finite real numbers; encode categories and handle missing values first")
    return X


def _infer_nominals(X):
    """Identify columns with 2–5 distinct, exactly integer-valued entries."""
    mask = np.zeros(X.shape[1], dtype=bool)
    for feature, column in enumerate(X.T):
        integer_valued = column.dtype.kind in "biu" or np.equal(column, np.floor(column)).all()
        if integer_valued:
            mask[feature] = 2 <= np.unique(column).size <= 5
    return mask


class Preprocessor:
    """Keep training columns/scales/category vocabularies fixed for inference."""

    def __init__(self, normalize):
        self.normalize = normalize

    def fit_transform(self, X, nominals):
        X = _array(X)
        if len(X) == 0:
            raise ValueError("Fitting requires at least one sample")
        self.n_features_in_ = X.shape[1]
        mask = _infer_nominals(X) if nominals is None else np.asarray(nominals)
        if mask.shape != (X.shape[1],) or mask.dtype.kind not in "biuf" or not np.isin(mask, [0, 1]).all():
            raise ValueError("nominals must be a boolean/0-1 mask with one entry per original feature")
        mask = mask.astype(bool)
        self.nominals_in_ = mask.copy()
        deviations = np.std(X.astype(np.float64), axis=0)
        # Exact categorical identity also handles integer codes beyond float32 precision.
        for column in np.flatnonzero(mask):
            values = X[:, column]
            if not np.equal(values, np.floor(values)).all():
                raise ValueError("Categorical feature {} must use integer codes".format(column))
            deviations[column] = float(np.unique(values).size > 1)
        self.feature_mask_ = deviations > 1e-6
        if not self.feature_mask_.any():
            raise ValueError("All features are constant (standard deviation <= 1e-6)")
        self.nominals_ = mask[self.feature_mask_]
        self.categories_ = {}
        self.minima_ = np.zeros(int(self.feature_mask_.sum()), dtype=np.float64)
        self.ranges_ = np.ones_like(self.minima_)
        for target, source in enumerate(np.flatnonzero(self.feature_mask_)):
            column = X[:, source]
            if mask[source]:
                self.categories_[target] = np.unique(column)
            else:
                column = column.astype(np.float64)
                if not self.normalize and (column.min() < 0 or column.max() > 1):
                    raise ValueError("Numerical feature {} is outside [0, 1]; use normalize=True".format(source))
                if self.normalize:
                    self.minima_[target] = column.min()
                    self.ranges_[target] = column.max() - column.min()
        return self.transform(X)

    def transform(self, X):
        X = _array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Expected {} original features, received {}".format(self.n_features_in_, X.shape[1]))
        result = np.empty((len(X), len(self.nominals_)), dtype=np.float32)
        for target, source in enumerate(np.flatnonzero(self.feature_mask_)):
            column = X[:, source]
            if self.nominals_[target]:
                if not np.equal(column, np.floor(column)).all():
                    raise ValueError("Categorical feature {} must use integer codes".format(source))
                categories = self.categories_[target]
                indices = np.searchsorted(categories, column)
                known = (indices < len(categories)) & (categories[np.minimum(indices, len(categories) - 1)] == column)
                result[:, target] = np.where(known, indices, -1)
            else:
                # Never clip queries: excursions beyond the training range are informative.
                result[:, target] = (column.astype(np.float64) - self.minima_[target]) / self.ranges_[target]
        if not np.isfinite(result).all():
            raise ValueError("Transformed numerical values exceed float32 range")
        return result
