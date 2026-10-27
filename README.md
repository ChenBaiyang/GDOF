# GDOF

Code and datasets for **Label-Informed Outlier Detection Based on Granule Density**, *IEEE Transactions on Fuzzy Systems*, 33(4):1391–1401, 2025.
[Paper](https://doi.org/10.1109/TFUZZ.2024.3514853) · [Accepted manuscript](https://arxiv.org/abs/2512.18774)

The current software version, **1.1**, is a **maintenance update** of the published implementation.

## Installation

Python 3.8+, NumPy and PyTorch are required; the scripts also use scikit-learn. From the repository root:

```shell
python -m pip install ".[experiments]"
```

For GPU acceleration, install a CUDA-enabled build from the [PyTorch installation page](https://pytorch.org/get-started/locally/). GDOF selects CUDA when available and CPU otherwise. Use `GDOF(device="cpu")` to force CPU or `GDOF(device="cuda:1")` to select a GPU.

## Usage

```python
import numpy as np
from gdof import GDOF

X = np.array([[10.1, 0], [11.2, 1], [12.3, 0], [13.4, 1], [100.5, 2]])
pos_idx = [4]  # indices of known outliers; all other objects are unlabeled

model = GDOF().fit(X, pos_idx)

# Transductive: return the saved scores for all rows of X used in fit(); one score per input row. larger means more anomalous.
scores = model.predict_score()

# Inductive: score new rows against the fixed training reference X.
test_scores = model.predict_score(np.array([[14., 0], [150., 3]]))
```

Inputs must be finite, two-dimensional real NumPy arrays. Encode string categories and handle missing values before fitting. Numerical features are min-max scaled using fitting data; numerical columns with standard deviation at most `1e-6` and single-valued categorical columns are removed. Predictions reuse the fitted transformations without clipping values outside the training range. Scores are not calibrated probabilities.

When `nominals` is omitted, a feature is categorical if it has **2–5 distinct integer values**, otherwise numerical. This rule is applied before scaling and frozen for prediction. `1.0` is integer-valued; `1.000001` is not. To specify types explicitly:

```python
model.fit(X, pos_idx, nominals=[False, True])
model.fit(X, pos_idx, nominals=[False] * X.shape[1])  # all numerical
```

Explicit masks take priority and must have one entry per original feature. Categories use integer codes and equality comparisons; unknown categories match no fitted category. The heuristic can misclassify low-cardinality counts or attributes with more than five categories, so provide known types explicitly. Inspect `model.nominals_in_` for the original-column mask.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_neg` | `200` | Pseudo-negative count, capped at `floor(0.8 * n_unlabeled)` |
| `alpha` | `0.5` | Attribute weights: `alpha * mean_negative_density - (1-alpha) * mean_known_outlier_density` |
| `normalize` | `True` | Scale numerical features; `False` requires retained numerical fitting columns in `[0,1]` |
| `random_state` | `1` | Pseudo-negative sampling seed |
| `batch_size` | `1024` | Relation rows per block; lower it to reduce memory use |
| `device` | `"auto"` | CUDA when available, otherwise CPU |

`pos_idx` must contain unique known-outlier indices, with at least one known outlier and two unlabeled objects. Pseudo-negatives are sampled from unlabeled objects and may include outliers; `model.n_neg_` gives the effective count. Batch size controls memory without changing neighborhoods; pairwise computation remains quadratic in fitting sample count.

## Examples and reproducing results in the paper

```shell
# Example usage of the algorithm
python example.py

# Reproducing one single run
python reproducing_paper_results.py --datasets breastw --n-pos 5 --seeds 0

# Reproducing GDOF results in Table II and Table III of the paper
python reproducing_paper_results.py --n-pos 5
```

`example.py` loads `data/breastw.npz` and prints ROC-AUC/AP for both protocols; replace its data-loading block with your own data:

- **Transductive (paper protocol):** all feature vectors participate in fitting, with only `pos_idx` supplying labels. Evaluate `predict_score()` after excluding known outliers.
- **Independent test set (inductive extension):** split data before fitting, select known outliers only within the training set, then call `predict_score(X_test)`. Preprocessing, training reference, granule radii, training densities and attribute weights stay fixed; test queries do not interact.

Dataset `.npz` files contain `X`, binary `y` (0 normal, 1 outlier), and an optional `nominals` mask. Supplied masks override inference. Ground-truth labels are used only to select known outliers and calculate metrics.

Without filters, `reproducing_paper_results.py` runs all `.npz` files in `data/` (20 supplied datasets) × known-outlier counts `{10,15,20,25,30}` × seeds `0..9`, using `n_neg=200` and `alpha=0.5`. The seeds control known-outlier selection; pseudo-negative sampling uses the model's default `random_state=1`. Use `--data-dir`, `--datasets`, `--n-pos`, `--seeds`, `--n-neg`, `--alpha`, `--device`, `--batch-size` and `--no-normalize` to configure a run; `--help` lists all options.

The default output is `outputs/results.csv`.

## Citation

```bibtex
@Article{chen2025GDOF,
  author  = {Chen, Baiyang and Yuan, Zhong and Peng, Dezhong and Chen, Hongmei and Song, Xiaomin and Zheng, Huiming},
  title   = {Label-Informed Outlier Detection Based on Granule Density},
  journal = {IEEE Transactions on Fuzzy Systems},
  year    = {2025},
  volume  = {33},
  number  = {4},
  pages   = {1391--1401},
  doi     = {10.1109/TFUZZ.2024.3514853}
}
```

Contact: farstars@qq.com. [Apache-2.0 license](LICENSE).
