"""Benign traffic generators.

Each generator returns a flow: a list of ``(timestamp_seconds, direction,
size_bytes)`` tuples sorted by time.  Direction 0 is client-to-server, 1 is
server-to-client.  All randomness is drawn from an explicitly seeded PCG64
generator, so a flow is reproducible from ``(class, seed)``.

The models are deliberately simple but distributionally realistic: lognormal
message sizes, Poisson page arrivals, and bursty on/off structure for web
browsing; near-MTU saturating transfers for bulk; heavy-tailed think times for
interactive sessions.  They are not traces and must not be treated as such.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .channel import Flow

MTU = 1500
MIN_SIZE = 40

BENIGN_CLASSES = ("doh_browse", "bulk", "interactive")


def _size(rng: np.random.Generator, mu: float, sigma: float, lo: int, hi: int) -> int:
    value = float(rng.lognormal(mu, sigma))
    return int(np.clip(value, lo, hi))


def doh_browse(seed: int, n: int = 512, rate_scale: float = 1.0) -> Flow:
    """Encrypted DNS browsing: request/response pairs in bursty page loads.

    Between pages the client is idle for an exponential think time (mean 340 ms
    scaled).  Inside a page, the client issues one query and the resolver
    returns 3-10 responses separated by a few milliseconds, with occasional
    uplink ACKs.  Sizes are lognormal; responses are much larger than queries.
    """
    rng = np.random.default_rng(seed)
    records: List[tuple] = []
    t = 0.0
    think = 0.34 / rate_scale
    while len(records) < n:
        t += float(rng.exponential(think))
        records.append((t, 0, _size(rng, np.log(190.0), 0.22, 70, 600)))
        responses = int(rng.integers(3, 11))
        for _ in range(responses):
            t += float(rng.exponential(0.006)) + 0.0008
            records.append((t, 1, _size(rng, np.log(820.0), 0.50, 180, MTU)))
            if rng.random() < 0.12:
                t += float(rng.exponential(0.004)) + 0.0005
                records.append((t, 0, _size(rng, np.log(120.0), 0.20, 60, 300)))
    return records[:n]


def bulk(seed: int, n: int = 512, rate_scale: float = 1.0) -> Flow:
    """Saturating download: near-MTU server packets paced by the receive window.

    The downlink is a tight periodic train (2.2 ms nominal spacing, heavy-tailed
    jitter); the uplink carries small ACK-like segments every ~35 ms.  Both
    directions are near-constant in size, which makes this class look regular
    in packet-size space but with a large size magnitude.
    """
    rng = np.random.default_rng(seed)
    records: List[tuple] = []
    t = 0.0
    down_gap = 0.0022 / rate_scale
    up_gap = 0.035 / rate_scale
    next_up = 0.0
    # A small share of bulk flows are idle keepalive trains: a stalled transfer
    # whose only remaining traffic is a periodic ACK.  This is a hard negative
    # for detectors that key on constant size and regular pacing.
    if rng.random() < 0.12:
        while len(records) < n:
            t += float(up_gap * rng.lognormal(0.0, 0.08))
            records.append((t, 0, int(np.clip(rng.normal(62.0, 3.0), MIN_SIZE, 90))))
        return records[:n]
    while len(records) < n:
        t += float(down_gap * rng.lognormal(0.0, 0.25))
        down = int(np.clip(rng.normal(1430.0, 18.0), 1200, MTU))
        records.append((t, 1, down))
        while next_up <= t:
            records.append((next_up, 0, int(np.clip(rng.normal(62.0, 6.0), MIN_SIZE, 120))))
            next_up += float(up_gap * rng.lognormal(0.0, 0.30))
    records.sort(key=lambda r: r[0])
    return records[:n]


def interactive(seed: int, n: int = 512, rate_scale: float = 1.0) -> Flow:
    """Interactive session: small requests, small responses, long think times.

    Think time is exponential with a 1.15 s mean, so packet rates are one to
    three packets per second.  Sizes are heavy tailed but bounded well below
    the MTU, and both directions carry comparable volume.
    """
    rng = np.random.default_rng(seed)
    records: List[tuple] = []
    t = 0.0
    think = 1.15 / rate_scale
    while len(records) < n:
        t += float(rng.exponential(think))
        records.append((t, 0, _size(rng, np.log(150.0), 0.35, 60, 700)))
        replies = int(rng.integers(1, 4))
        for _ in range(replies):
            t += float(rng.exponential(0.055)) + 0.010
            records.append((t, 1, _size(rng, np.log(320.0), 0.55, 60, 1200)))
        if rng.random() < 0.20:
            t += float(rng.exponential(0.02)) + 0.004
            records.append((t, 0, _size(rng, np.log(95.0), 0.25, 50, 300)))
    return records[:n]


GENERATORS: Dict[str, Callable[..., Flow]] = {
    "doh_browse": doh_browse,
    "bulk": bulk,
    "interactive": interactive,
}


def generate(cls: str, seed: int, n: int = 512, rate_scale: float = 1.0) -> Flow:
    if cls not in GENERATORS:
        raise ValueError("unknown benign class: %r" % (cls,))
    return GENERATORS[cls](seed=seed, n=n, rate_scale=rate_scale)


def scale_for_seed(seed: int, low: float = 0.80, high: float = 1.25) -> float:
    """Deterministic per-flow rate scaling shared by all benign classes."""
    return float(low + (high - low) * ((seed * 2654435761) % 10007) / 10007.0)
