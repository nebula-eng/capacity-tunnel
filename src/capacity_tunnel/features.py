"""Per-flow feature extraction over a packet window.

Every feature is computed from packet metadata only: timestamp, direction, and
size.  No payload, no port, no address, and no label information is used.  The
feature vector is the union of classical shape statistics and the capacity
estimates from :mod:`capacity_tunnel.capacity`.

``FEATURE_NAMES`` is the canonical order.  :func:`extract` returns a float64
vector in that order; :func:`extract_named` returns a dict for inspection.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

from .capacity import SIZE_BIN_BYTES, estimate_flow, shannon_bits
from .channel import Flow

WINDOW = 512
FEATURE_NAMES: Tuple[str, ...] = (
    "size_entropy",
    "size_entropy_rate",
    "size_mi",
    "size_alphabet",
    "size_uniformity",
    "timing_entropy",
    "timing_snr",
    "timing_separation",
    "timing_capacity",
    "iat_cv",
    "fano_factor",
    "up_fraction",
    "size_acf_peak",
    "periodicity",
    "unique_ratio",
    "log_mean_size",
    "log_pkt_rate",
    "log_capacity_bps",
    "log_capacity_ratio",
    "size_class_share",
)

IAT_EDGES = np.geomspace(0.0004, 4.0, 17)  # 16 log-spaced bins, 0.4 ms .. 4 s


def _acf_peak(x: np.ndarray, max_lag: int) -> float:
    if x.size <= max_lag + 1:
        return 0.0
    y = x - x.mean()
    denom = float(np.dot(y, y))
    if denom <= 0.0:
        return 0.0
    peak = 0.0
    for lag in range(1, max_lag + 1):
        value = float(np.dot(y[:-lag], y[lag:]) / denom)
        peak = max(peak, abs(value))
    return peak


def _periodicity(times: np.ndarray, iats: np.ndarray) -> float:
    """Normalized autocorrelation peak of the packet-arrival event train.

    The bin width scales with the flow's own median inter-arrival time so the
    score is comparable across rates.  A paced tunnel produces a sharp peak at
    its symbol period; Poisson traffic does not.
    """
    if times.size < 16 or iats.size == 0:
        return 0.0
    median = float(np.median(iats))
    bin_width = max(0.0005, median / 4.0)
    n_bins = 256
    span = times[-1] - times[0]
    if span <= 0:
        return 0.0
    idx = np.floor((times - times[0]) / bin_width).astype(np.int64)
    idx = idx[idx < n_bins]
    if idx.size < 16:
        return 0.0
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    y = counts - counts.mean()
    denom = float(np.dot(y, y))
    if denom <= 0.0:
        return 0.0
    max_lag = min(64, n_bins // 2)
    peak = 0.0
    for lag in range(2, max_lag + 1):
        value = float(np.dot(y[:-lag], y[lag:]) / denom)
        peak = max(peak, abs(value))
    return peak


def extract_named(flow: Flow, window: int = WINDOW) -> Dict[str, float]:
    """Return the feature vector as a name -> value mapping."""
    recs = sorted(flow, key=lambda r: r[0])[:window]
    n = len(recs)
    if n < 8:
        return {name: 0.0 for name in FEATURE_NAMES}

    times = np.array([r[0] for r in recs], dtype=float)
    dirs = np.array([r[1] for r in recs], dtype=np.int64)
    sizes = np.array([r[2] for r in recs], dtype=float)
    iats = np.clip(np.diff(times), 1e-6, None)

    est = estimate_flow(recs, window=window)

    bins = np.minimum(sizes.astype(np.int64) // SIZE_BIN_BYTES, 256)
    counts = np.bincount(bins)
    size_entropy = shannon_bits(bins)

    if iats.size:
        iat_idx = np.clip(np.searchsorted(IAT_EDGES, iats, side="right") - 1, 0, len(IAT_EDGES) - 2)
        timing_entropy = shannon_bits(iat_idx)
    else:
        timing_entropy = 0.0

    if iats.size >= 4:
        bin_s = 0.010
        span = times[-1] - times[0]
        n_bins = max(4, int(math.floor(span / bin_s)) + 1)
        idx = np.floor((times - times[0]) / bin_s).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        per_bin = np.bincount(idx, minlength=n_bins).astype(float)
        mean_bin = float(per_bin.mean())
        fano = float(per_bin.var() / mean_bin) if mean_bin > 0 else 0.0
        iat_cv = float(np.std(iats) / max(np.mean(iats), 1e-12))
    else:
        fano = 0.0
        iat_cv = 0.0

    acf_peak = _acf_peak(sizes, 8)
    periodicity = _periodicity(times, iats)

    unique_ratio = float(np.unique(sizes).size / float(n))
    mean_size = float(sizes.mean())
    pkt_rate = float((n - 1) / (times[-1] - times[0])) if times[-1] > times[0] else 0.0
    up_fraction = float(np.mean(dirs == 0))
    share = float(counts.max() / float(n)) if counts.size else 0.0

    values = {
        "size_entropy": float(size_entropy),
        "size_entropy_rate": float(est.h_rate),
        "size_mi": float(est.mi_joint),
        "size_alphabet": float(est.alphabet_size),
        "size_uniformity": float(est.size_uniformity),
        "timing_entropy": float(timing_entropy),
        "timing_snr": float(est.timing_snr),
        "timing_separation": float(est.timing_separation),
        "timing_capacity": float(est.c_timing),
        "iat_cv": float(iat_cv),
        "fano_factor": float(fano),
        "up_fraction": float(up_fraction),
        "size_acf_peak": float(acf_peak),
        "periodicity": float(periodicity),
        "unique_ratio": float(unique_ratio),
        "log_mean_size": float(math.log10(max(mean_size, 1.0))),
        "log_pkt_rate": float(math.log10(max(pkt_rate, 1e-3))),
        "log_capacity_bps": float(math.log10(1.0 + max(est.capacity_bps, 0.0))),
        "log_capacity_ratio": float(math.log10(max(est.capacity_ratio, 0.0) + 1e-7)),
        "size_class_share": float(share),
    }
    return values


def extract(flow: Flow, window: int = WINDOW) -> np.ndarray:
    """Return the feature vector in ``FEATURE_NAMES`` order."""
    named = extract_named(flow, window=window)
    return np.array([named[name] for name in FEATURE_NAMES], dtype=float)


def to_matrix(flows: Sequence[Flow], window: int = WINDOW) -> np.ndarray:
    return np.vstack([extract(flow, window=window) for flow in flows])
