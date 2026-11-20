"""Covert-channel capacity estimator.

The estimator treats packet metadata as the output of a putative encoder.  If a
tunnel encodes symbols in size and/or inter-arrival time, then the per-flow
symbol rate is bounded by the empirical entropy rate of the emitted symbols
multiplied by the packet rate.  Benign traffic can have high marginal entropy
(continuous lognormal sizes) but its symbols do not form a small repeatable
alphabet, so the *entropy rate over an effective alphabet* stays low.

Definitions (all entropies in bits):

* Effective size alphabet: 16-byte bins whose empirical share is at least
  ``MIN_CLASS_SHARE``; at most ``MAX_CLASSES`` bins, ranked by mass.  Every
  other packet maps to a residual symbol.  A covert encoder must use a small,
  stable alphabet to be decodable at the receiver, so rare one-off sizes do not
  count as capacity.
* Joint symbol ``X_i = (size_class_i, timing_class_i)``, where ``timing_class``
  is the 1-D 2-means partition of the inter-arrival gaps.
* Entropy rate: ``H_rate = H(X) - I(X_i; X_{i+1})``, the Markov-1 entropy rate.
  ``I`` is bias-corrected by ``(K-1)^2 / (2 N ln 2)``.
* Timing capacity: ``SNR = var_between / var_within`` over the 2-means
  partition, ``C_t = min(1, 0.5 * log2(1 + SNR))``.  The clamp is the binary
  input constraint: the partition admits at most one bit per packet.
* Reported capacity: ``capacity_bps = pkt_rate * min(H_rate, 1 + C_t)``.

This is an estimate, not a proof: it is an upper bound on what an *observer*
could attribute to covert symbols, and it inherits every assumption of the
quantization.  Failure modes are listed in docs/DESIGN.md.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .channel import Flow

SIZE_BIN_BYTES = 16
MIN_CLASS_SHARE = 0.05
MAX_CLASSES = 12
MIN_WINDOW = 8
MIN_TIMING_SEPARATION = 0.20
LN2 = math.log(2.0)


def shannon_bits(labels: Sequence[int]) -> float:
    arr = np.asarray(labels)
    if arr.size == 0:
        return 0.0
    counts = np.bincount(arr.astype(np.int64))
    p = counts[counts > 0] / float(arr.size)
    return float(-(p * np.log2(p)).sum())


def _mm_correction(labels: Sequence[int]) -> float:
    arr = np.asarray(labels)
    if arr.size == 0:
        return 0.0
    k = int(np.unique(arr).size)
    return (k - 1) / (2.0 * arr.size * LN2)


def mutual_information(a: Sequence[int], b: Sequence[int], permutations: int = 16, seed: int = 0) -> float:
    """Mutual information calibrated against a permutation null.

    The plugin estimator is biased upward for sparse contingency tables.  We
    subtract the mean MI of ``permutations`` seeded shuffles of ``b``, which
    estimates that bias without assuming a table size.  The result is clipped
    at zero.
    """
    x = np.asarray(a, dtype=np.int64)
    y = np.asarray(b, dtype=np.int64)
    if x.size < 2 or x.size != y.size:
        return 0.0
    kx = int(x.max()) + 1
    plugin = shannon_bits(x) + shannon_bits(y) - shannon_bits(x * kx + y)
    if permutations <= 0:
        return float(max(0.0, plugin))
    rng = np.random.default_rng(seed)
    null: List[float] = []
    for _ in range(permutations):
        shuffled = rng.permutation(y)
        null.append(shannon_bits(x) + shannon_bits(shuffled) - shannon_bits(x * kx + shuffled))
    return float(max(0.0, plugin - float(np.mean(null))))


def size_alphabet(sizes: np.ndarray) -> Tuple[np.ndarray, List[int], int]:
    """Map sizes to a small recurring-bin alphabet plus a residual symbol."""
    if sizes.size == 0:
        return np.zeros(0, dtype=int), [], 0
    bins = np.minimum(sizes.astype(np.int64) // SIZE_BIN_BYTES, 256)
    counts = np.bincount(bins)
    order = np.argsort(-counts, kind="stable")
    threshold = MIN_CLASS_SHARE * sizes.size
    keep = [int(b) for b in order if counts[b] >= threshold][:MAX_CLASSES]
    keep.sort()
    labels = np.full(sizes.size, len(keep), dtype=int)
    for idx, b in enumerate(keep):
        labels[bins == b] = idx
    return labels, keep, len(keep)


def two_means_1d(x: np.ndarray, iters: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic 1-D 2-means.

    Initial centres are the 25th and 75th percentiles, which is deterministic
    and avoids degenerate splits on heavy-tailed gaps.  Empty clusters retain
    their previous centre.
    """
    if x.size == 0:
        return np.zeros(0, dtype=int), np.zeros(2)
    lo, hi = np.percentile(x, [25.0, 75.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(x))
        hi = float(np.max(x))
    if lo == hi:
        return np.zeros(x.size, dtype=int), np.array([lo, hi], dtype=float)
    centers = np.array([lo, hi], dtype=float)
    labels = np.zeros(x.size, dtype=int)
    for _ in range(iters):
        dist = np.abs(x[:, None] - centers[None, :])
        labels = np.argmin(dist, axis=1)
        new = centers.copy()
        for j in (0, 1):
            mask = labels == j
            if np.any(mask):
                new[j] = float(x[mask].mean())
        if np.allclose(new, centers, rtol=0.0, atol=1e-12):
            centers = new
            break
        centers = new
    return labels, centers


@dataclass
class CapacityEstimate:
    packets: int
    duration_s: float
    pkt_rate: float
    mean_size: float
    h_size: float
    alphabet_size: int
    size_uniformity: float
    h_joint: float
    mi_joint: float
    h_rate: float
    timing_cv: float
    timing_snr: float
    timing_separation: float
    c_timing: float
    capacity_bps: float
    wire_bitrate_bps: float
    capacity_ratio: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def estimate_flow(flow: Flow, window: int = 512) -> CapacityEstimate:
    """Estimate capacity for the first ``window`` packets of ``flow``."""
    recs = sorted(flow, key=lambda r: r[0])[:window]
    n = len(recs)
    if n < MIN_WINDOW:
        return CapacityEstimate(
            packets=n,
            duration_s=0.0,
            pkt_rate=0.0,
            mean_size=0.0,
            h_size=0.0,
            alphabet_size=0,
            size_uniformity=0.0,
            h_joint=0.0,
            mi_joint=0.0,
            h_rate=0.0,
            timing_cv=0.0,
            timing_snr=0.0,
            timing_separation=0.0,
            c_timing=0.0,
            capacity_bps=0.0,
            wire_bitrate_bps=0.0,
            capacity_ratio=0.0,
        )

    times = np.array([r[0] for r in recs], dtype=float)
    sizes = np.array([r[2] for r in recs], dtype=float)
    duration = float(times[-1] - times[0])
    iats = np.diff(times)
    iats = iats[np.isfinite(iats)]
    iats = np.clip(iats, 1e-6, None)

    if duration > 0:
        pkt_rate = (n - 1) / duration
        wire_bitrate = float(8.0 * sizes.sum() / duration)
    else:
        pkt_rate = 0.0
        wire_bitrate = 0.0

    size_labels, centers, k_size = size_alphabet(sizes)
    h_size = max(0.0, shannon_bits(size_labels) - _mm_correction(size_labels))
    uniformity = 0.0
    if k_size > 1 and h_size > 0:
        uniformity = min(1.0, h_size / math.log2(float(k_size)))

    separation = 0.0
    codebook_ok = False
    if iats.size >= 2 and float(np.mean(iats)) > 0:
        timing_labels, centers_t = two_means_1d(np.log1p(iats / 1e-3))
        cv = float(np.std(iats) / max(np.mean(iats), 1e-12))
        means = []
        variances = []
        weights = []
        for j in (0, 1):
            mask = timing_labels == j
            weights.append(float(mask.sum()))
            if np.any(mask):
                means.append(float(iats[mask].mean()))
                variances.append(float(iats[mask].var()))
            else:
                means.append(float(iats.mean()))
                variances.append(0.0)
        total = float(iats.size)
        overall = float(iats.mean())
        between = sum(w * (m - overall) ** 2 for w, m in zip(weights, means)) / total
        within = sum(w * v for w, v in zip(weights, variances)) / total
        snr = float(between / max(within, overall * overall * 1e-6 + 1e-12))
        snr = float(np.clip(snr, 0.0, 50.0))
        separation = abs(means[0] - means[1]) / max(max(means[0], means[1]), 1e-9)
        # A timing codebook must separate its classes by at least 20% of the
        # mean gap; otherwise the 2-means split is modelling jitter, not
        # symbols.  The reference codebooks separate 5 ms and 25 ms (0.80).
        codebook_ok = separation >= MIN_TIMING_SEPARATION
        c_timing = float(np.clip(0.5 * math.log2(1.0 + snr), 0.0, 1.0)) if codebook_ok else 0.0
        timing_per_packet = np.zeros(n, dtype=int)
        if codebook_ok:
            # Timing label for packet i is the class of the gap that follows it.
            timing_per_packet[: iats.size] = timing_labels[: n - 1]
            if iats.size:
                timing_per_packet[n - 1] = timing_labels[-1]
    else:
        cv = 0.0
        snr = 0.0
        c_timing = 0.0
        timing_per_packet = np.zeros(n, dtype=int)

    joint = size_labels * 2 + timing_per_packet
    h_joint = max(0.0, shannon_bits(joint) - _mm_correction(joint))
    mi_joint = mutual_information(joint[:-1], joint[1:])
    h_rate = float(max(0.0, h_joint - mi_joint))

    capacity_bps = float(pkt_rate * min(h_rate, 1.0 + c_timing))
    ratio = capacity_bps / wire_bitrate if wire_bitrate > 0 else 0.0

    return CapacityEstimate(
        packets=n,
        duration_s=duration,
        pkt_rate=float(pkt_rate),
        mean_size=float(sizes.mean()),
        h_size=float(h_size),
        alphabet_size=int(k_size),
        size_uniformity=float(uniformity),
        h_joint=float(h_joint),
        mi_joint=float(mi_joint),
        h_rate=h_rate,
        timing_cv=float(cv),
        timing_snr=float(snr),
        timing_separation=float(separation),
        c_timing=float(c_timing),
        capacity_bps=capacity_bps,
        wire_bitrate_bps=float(wire_bitrate),
        capacity_ratio=float(ratio),
    )
