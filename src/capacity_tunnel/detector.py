"""Detector: L2-regularized logistic regression, ROC/AUC, cross-validation.

The classifier is implemented directly in numpy (no scikit-learn).  Features
are standardized on the training fold only.  Thresholds are swept on
out-of-fold scores; the reported operating point is FPR at TPR = 0.95.

The module also builds the detection dataset: tunnel flows come from the real
channel's emission schedule (:func:`capacity_tunnel.channel.emit_flow`), benign
flows from :mod:`capacity_tunnel.benign`.  Every flow is seeded.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import benign, channel, features

TUNNEL_CLASSES: Tuple[str, ...] = ("b32", "timing", "hardened")
BENIGN_CLASSES: Tuple[str, ...] = ("doh_browse", "bulk", "interactive")
ALL_CLASSES: Tuple[str, ...] = TUNNEL_CLASSES + BENIGN_CLASSES

FEATURE_SETS: Dict[str, Tuple[str, ...]] = {
    "entropy_only": ("size_entropy", "timing_entropy"),
    "size_only": (
        "size_entropy",
        "size_mi",
        "size_alphabet",
        "unique_ratio",
        "log_mean_size",
        "size_class_share",
        "size_acf_peak",
    ),
    "timing_only": (
        "timing_entropy",
        "timing_snr",
        "timing_separation",
        "timing_capacity",
        "iat_cv",
        "periodicity",
    ),
    "capacity_only": (
        "size_entropy_rate",
        "size_uniformity",
        "timing_separation",
        "timing_capacity",
        "log_capacity_bps",
        "log_capacity_ratio",
    ),
    "no_direction": tuple(n for n in features.FEATURE_NAMES if n != "up_fraction"),
    "full": features.FEATURE_NAMES,
}


@dataclass
class LogReg:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray
    names: Tuple[str, ...] = features.FEATURE_NAMES

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mean) / self.std
        return Z @ self.weights + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.decision_function(X))

    def to_dict(self) -> Dict[str, object]:
        return {
            "weights": [float(v) for v in self.weights],
            "bias": float(self.bias),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "names": list(self.names),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "LogReg":
        return cls(
            weights=np.array(payload["weights"], dtype=float),
            bias=float(payload["bias"]),
            mean=np.array(payload["mean"], dtype=float),
            std=np.array(payload["std"], dtype=float),
            names=tuple(payload.get("names", features.FEATURE_NAMES)),
        )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def fit_logreg(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-2,
    iters: int = 1500,
    lr: float = 0.5,
    names: Tuple[str, ...] = features.FEATURE_NAMES,
) -> LogReg:
    """Full-batch gradient descent on the mean L2-regularized log loss."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Z = (X - mean) / std

    w = np.zeros(d, dtype=float)
    b = 0.0
    prev_loss = math.inf
    for _ in range(iters):
        p = _sigmoid(Z @ w + b)
        resid = p - y
        grad = Z.T @ resid / n + l2 * w
        grad_b = float(resid.mean())
        w -= lr * grad
        b -= lr * grad_b
        if (_ % 50) == 0:
            eps = 1e-12
            loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + 0.5 * l2 * w @ w)
            if abs(prev_loss - loss) < 1e-12:
                break
            prev_loss = loss
    return LogReg(weights=w, bias=b, mean=mean, std=std, names=names)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def roc_auc(y: Sequence[int], scores: Sequence[float]) -> float:
    """Mann-Whitney AUC with tie handling.  Returns 0.5 for degenerate input."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    ranks = _rankdata(s)
    auc = (ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def roc_curve(y: Sequence[int], scores: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    tpr = np.concatenate([[0.0], tps / pos])
    fpr = np.concatenate([[0.0], fps / neg])
    thresholds = np.concatenate([[math.inf], s_sorted])
    return fpr, tpr, thresholds


def fpr_at_tpr(y: Sequence[int], scores: Sequence[float], target: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    mask = tpr >= target
    if not np.any(mask):
        return 1.0
    return float(fpr[mask].min())


def tpr_at_fpr(y: Sequence[int], scores: Sequence[float], target: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    mask = fpr <= target
    if not np.any(mask):
        return 0.0
    return float(tpr[mask].max())


def threshold_at_tpr(y: Sequence[int], scores: Sequence[float], target: float = 0.95) -> float:
    fpr, tpr, thr = roc_curve(y, scores)
    mask = tpr >= target
    if not np.any(mask):
        return float(np.max(scores))
    idx = int(np.argmax(mask))
    return float(thr[idx])


def stratified_folds(y: Sequence[int], k: int = 5, seed: int = 0) -> List[np.ndarray]:
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.nonzero(y == cls)[0]
        idx = idx[rng.permutation(idx.size)]
        for pos, i in enumerate(idx):
            folds[pos % k].append(int(i))
    return [np.array(sorted(f), dtype=int) for f in folds]


def stratified_folds_by_class(labels: Sequence[str], k: int = 5, seed: int = 0) -> List[np.ndarray]:
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(k)]
    for cls in sorted(set(labels.tolist())):
        idx = np.nonzero(labels == cls)[0]
        idx = idx[rng.permutation(idx.size)]
        for pos, i in enumerate(idx):
            folds[pos % k].append(int(i))
    return [np.array(sorted(f), dtype=int) for f in folds]


@dataclass
class CVResult:
    oof_scores: np.ndarray
    fold_aucs: List[float]
    fold_fpr: List[float]
    auc: float
    fpr_at_95tpr: float
    threshold: float


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
    k: int = 5,
    seed: int = 0,
    l2: float = 1e-2,
    iters: int = 1500,
) -> CVResult:
    """Stratified k-fold CV (stratified by class label, not just the binary label)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    folds = stratified_folds_by_class(labels, k=k, seed=seed)
    oof = np.zeros(y.size, dtype=float)
    fold_aucs: List[float] = []
    fold_fpr: List[float] = []
    for i, test_idx in enumerate(folds):
        train_mask = np.ones(y.size, dtype=bool)
        train_mask[test_idx] = False
        model = fit_logreg(X[train_mask], y[train_mask], l2=l2, iters=iters)
        scores = model.decision_function(X[test_idx])
        oof[test_idx] = scores
        if len(np.unique(y[test_idx])) == 2:
            fold_aucs.append(roc_auc(y[test_idx], scores))
            fold_fpr.append(fpr_at_tpr(y[test_idx], scores, 0.95))
    return CVResult(
        oof_scores=oof,
        fold_aucs=fold_aucs,
        fold_fpr=fold_fpr,
        auc=roc_auc(y, oof),
        fpr_at_95tpr=fpr_at_tpr(y, oof, 0.95),
        threshold=threshold_at_tpr(y, oof, 0.95),
    )


def stratified_holdout(
    labels: Sequence[str], seed: int = 0, fraction: float = 0.30
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices into train/test, stratified by class label."""
    labels_arr = np.asarray(labels)
    rng = np.random.default_rng(seed)
    test_idx: List[int] = []
    for cls in sorted(set(labels_arr.tolist())):
        idx = np.nonzero(labels_arr == cls)[0]
        idx = idx[rng.permutation(idx.size)]
        k = max(1, int(round(fraction * idx.size)))
        test_idx.extend(int(i) for i in idx[:k])
    mask = np.zeros(labels_arr.size, dtype=bool)
    mask[np.array(sorted(test_idx), dtype=int)] = True
    return np.nonzero(~mask)[0], np.nonzero(mask)[0]


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    class_labels: List[str]
    flows_per_class: int
    window: int
    seed: int

    @property
    def is_tunnel(self) -> np.ndarray:
        return self.y.astype(bool)


def _benign_flow(cls: str, flow_seed: int, window: int, mix_prob: float = 0.25) -> List[tuple]:
    """Generate a benign flow, sometimes as a two-phase mixed-activity flow.

    A quarter of benign flows concatenate a phase of one activity with a phase
    of another (for example a browser flow that starts a bulk download, or an
    interactive session that transfers a file).  These mixed flows are the hard
    negatives for single-activity detectors.
    """
    rng = np.random.default_rng(flow_seed ^ 0x5F5F5F)
    rate = benign.scale_for_seed(flow_seed)
    if rng.random() >= mix_prob:
        return benign.generate(cls, seed=flow_seed, n=window, rate_scale=rate)
    others = [c for c in BENIGN_CLASSES if c != cls]
    other = others[int(rng.integers(0, len(others)))]
    first = int(rng.integers(window // 3, (2 * window) // 3))
    head = benign.generate(cls, seed=flow_seed, n=first, rate_scale=rate)
    tail = benign.generate(other, seed=flow_seed + 1, n=window - first, rate_scale=rate)
    offset = head[-1][0] + float(rng.exponential(0.05))
    return head + [(t + offset, d, s) for (t, d, s) in tail]


def build_dataset(per_class: int = 80, window: int = 512, seed: int = 2024) -> Dataset:
    """Generate the detection dataset with seeded flows for all six classes."""
    X: List[np.ndarray] = []
    y: List[int] = []
    labels: List[str] = []

    for ci, mode in enumerate(TUNNEL_CLASSES):
        for i in range(per_class):
            flow_seed = seed + 1013 * (ci + 1) + 17 * i
            rng = np.random.default_rng(flow_seed)
            rate = float(rng.uniform(0.80, 1.25))
            jitter = float(rng.uniform(0.75, 1.25))
            flow = channel.emit_flow(
                mode,
                seed=flow_seed,
                n_packets=window,
                rate_scale=rate,
                jitter_scale=jitter,
            )
            # A quarter of the tunnels run inbound (server-to-client tasking),
            # so direction alone is not a shortcut for the detector.
            if rng.random() < 0.25:
                flow = [(t, 1, s) for (t, d, s) in flow]
            X.append(features.extract(flow, window=window))
            y.append(1)
            labels.append(mode)

    for ci, cls in enumerate(BENIGN_CLASSES):
        for i in range(per_class):
            flow_seed = seed + 7717 * (ci + 1) + 31 * i
            flow = _benign_flow(cls, flow_seed, window)
            X.append(features.extract(flow, window=window))
            y.append(0)
            labels.append(cls)

    return Dataset(
        X=np.vstack(X),
        y=np.array(y, dtype=int),
        class_labels=labels,
        flows_per_class=per_class,
        window=window,
        seed=seed,
    )


def subset_indices(names: Sequence[str]) -> List[int]:
    index = {name: i for i, name in enumerate(features.FEATURE_NAMES)}
    return [index[n] for n in names]


def evasion_sweep(
    model: LogReg,
    benign_X: np.ndarray,
    benign_y: np.ndarray,
    varying: str = "padding",
    scales: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    per_scale: int = 40,
    window: int = 512,
    seed: int = 9091,
    entropy_model: Optional[LogReg] = None,
    entropy_indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, float]]:
    """Re-generate hardened flows with a stronger evasion knob and re-score them.

    ``varying`` selects the knob: ``"padding"`` scales the uniform size-padding
    envelope, ``"timing_jitter"`` scales the Gaussian jitter on the codebook
    gaps.  The detector is frozen (trained on the base split); the benign side
    is the held-out benign matrix passed by the caller.
    """
    if varying not in ("padding", "timing_jitter"):
        raise ValueError("varying must be 'padding' or 'timing_jitter'")

    benign_scores = model.decision_function(benign_X)
    benign_entropy = None
    if entropy_model is not None:
        benign_slice = benign_X[:, entropy_indices] if entropy_indices is not None else benign_X
        benign_entropy = entropy_model.decision_function(benign_slice)
    capacity_idx = features.FEATURE_NAMES.index("log_capacity_bps")
    separation_idx = features.FEATURE_NAMES.index("timing_separation")

    rows: List[Dict[str, float]] = []
    for si, scale in enumerate(scales):
        pad_scale = float(scale) if varying == "padding" else 1.0
        jitter_scale = float(scale) if varying == "timing_jitter" else 1.0
        X: List[np.ndarray] = []
        for i in range(per_scale):
            flow_seed = seed + 401 * si + 7 * i
            rng = np.random.default_rng(flow_seed)
            rate = float(rng.uniform(0.85, 1.15))
            flow = channel.emit_flow(
                "hardened",
                seed=flow_seed,
                n_packets=window,
                pad_scale=pad_scale,
                jitter_scale=jitter_scale,
                rate_scale=rate,
            )
            X.append(features.extract(flow, window=window))
        X_arr = np.vstack(X)
        y = np.concatenate([np.ones(X_arr.shape[0], dtype=int), benign_y])
        scores = np.concatenate([model.decision_function(X_arr), benign_scores])
        row: Dict[str, float] = {
            "scale": float(scale),
            "pad_scale": pad_scale,
            "jitter_scale": jitter_scale,
            "auc": roc_auc(y, scores),
            "fpr_at_95tpr": fpr_at_tpr(y, scores, 0.95),
            "n_tunnel": int(X_arr.shape[0]),
            "n_benign": int(benign_y.size),
            "median_capacity_bps": float(np.median(np.power(10.0, X_arr[:, capacity_idx]) - 1.0)),
            "median_timing_separation": float(np.median(X_arr[:, separation_idx])),
        }
        if entropy_model is not None and benign_entropy is not None:
            ent_slice = X_arr[:, entropy_indices] if entropy_indices is not None else X_arr
            ent_scores = np.concatenate([entropy_model.decision_function(ent_slice), benign_entropy])
            row["auc_entropy_only"] = roc_auc(y, ent_scores)
        rows.append(row)
    return rows


def save_model(model: LogReg, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(model.to_dict(), fh, indent=2, sort_keys=True)


def load_model(path: str) -> LogReg:
    with open(path, "r", encoding="utf-8") as fh:
        return LogReg.from_dict(json.load(fh))
