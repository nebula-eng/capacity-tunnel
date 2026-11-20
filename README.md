# capacity-tunnel

Covert-tunnel detection from covert-channel capacity.

[![CI](https://github.com/nebula-eng/capacity-tunnel/actions/workflows/ci.yml/badge.svg)](https://github.com/nebula-eng/capacity-tunnel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776ab.svg)](pyproject.toml)

Covert tunnels that pad packets or jitter their timing defeat entropy-based
detectors: the marginal distribution of packet sizes looks benign even though
the flow is carrying a payload. This project tests a different signal. A tunnel
that must *carry* data has to shape packet sizes and inter-arrival gaps into a
decodable symbol alphabet. That shaping leaves a structural signature: a small,
stable, near-uniform alphabet with low conditional entropy and, for timing
channels, two well-separated gap clusters. The detector estimates the achievable
covert-channel capacity of each flow from packet metadata alone (size,
direction, timestamp) and classifies on that estimate rather than on entropy.

The repository contains a real covert channel over UDP loopback, three benign
traffic generators, a per-flow capacity estimator, a numpy logistic-regression
detector, and a reproducible evaluation pipeline with figures.

## Relation to entropy detectors

A size-entropy detector asks "how random are the packet sizes?". Padding raises
that number, and benign traffic already has high size entropy, so the statistic
is easy to move and hard to threshold. The capacity estimator instead asks "how
many bits per packet could an encoder actually transmit through this flow?".
That requires four things entropy ignores:

1. A *recurring* alphabet. Sizes are mapped to 16-byte bins that hold at least
   5% of the window; one-off sizes collapse into a residual symbol and cannot
   contribute capacity. DoH browsing has 5.995779 bits of raw size entropy in this
   dataset but a median effective alphabet of one recurring bin, so its entropy
   rate collapses to 0.929632 bit/packet.
2. A Markov-1 *entropy rate* `H(X) - I(X_i; X_{i+1})`, with the mutual
   information calibrated against a permutation null.
3. A *timing* capacity from a deterministic 1-D 2-means partition of the
   inter-arrival gaps, gated on a 20% relative separation so that jitter is not
   mistaken for a codebook.
4. Capacity in bits per second, scaled by the observed packet rate, plus a
   capacity-to-wire-bitrate ratio.

On the evaluation dataset the entropy-only baseline reaches AUC 0.902378
(FPR@95TPR 0.229167), while the 20-feature capacity detector reaches AUC
1.000000 (FPR@95TPR 0.000000). The point is not that entropy is useless; it is
that capacity is the quantity the adversary has to spend.

## Install

Not on PyPI. Install from a clone:

```bash
git clone https://github.com/nebula-eng/capacity-tunnel.git
cd capacity-tunnel
python -m pip install -e ".[plots]"
```

`numpy` is a runtime dependency: the estimator and the detector are vectorised.
`matplotlib` is behind the `plots` extra and is needed only for `make eval` and
`make figures`. Python 3.11 or newer.

## Quickstart

```
make demo     # real UDP loopback channel, all three modes, prints throughput and BER
make test     # 30 unittest cases, including real channel round-trips
make eval     # dataset -> 5-fold CV -> baselines -> channel -> evasion sweeps -> figures
```

Without make:

```
PYTHONPATH=src python -m capacity_tunnel.cli demo --payload 64
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m capacity_tunnel.cli eval --out results
PYTHONPATH=src python scripts/make_figures.py
```

`make eval` writes `results/results.json`, `results/model.json`,
`results/dataset.npz`, `results/flow_scores.npz`, and three figures at
1600x1000 (dpi 200). A full run takes about 21 s of compute (22 s wall) on the
reference host; 12 s of that is the three live channels, 8 s of it the timing mode
alone at 5/25 ms per bit.

## Results

Measured with seed 2024, 80 flows per class, 512-packet windows, 5-fold
stratified CV. Values below are copied from `results/results.json`.

| Mode | AUC vs benign | FPR@95TPR | Throughput (bit/s) | BER | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| b32 | 1.000000 | 0.000000 | 1248.072450 | 0.0 | match |
| timing | 1.000000 | 0.000000 | 62.148717 | 0.0 | match |
| hardened | 1.000000 | 0.000000 | 171.630709 | 0.0 | match |
| all tunnels | 1.000000 | 0.000000 | - | - | - |

Channel throughput is protocol-normalized: payload bits divided by the summed
scheduled inter-packet gaps of the executed plan. Wall-clock time is longer by
scheduler latency and is printed by `make demo`; it is not used in any reported
metric. `make demo` uses a different seed from the evaluation, so its throughput
differs by up to about 2% because the per-gap jitter is reseeded.

Feature-set ablations (same folds, same model class):

| Feature set | Features | AUC | FPR@95TPR |
| --- | --- | --- | --- |
| entropy only (`size_entropy`, `timing_entropy`) | 2 | 0.902378 | 0.229167 |
| size only | 7 | 0.991667 | 0.037500 |
| timing only | 6 | 0.990278 | 0.037500 |
| capacity only | 6 | 1.000000 | 0.000000 |
| no direction | 19 | 1.000000 | 0.000000 |
| full | 20 | 1.000000 | 0.000000 |

Evasion sweeps re-generate hardened flows against a frozen detector:

| Knob | 1x | 2x | 4x | 8x | 16x |
| --- | --- | --- | --- | --- | --- |
| size padding, AUC | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| size padding, median capacity (bit/s) | 135.752520 | 65.402450 | 69.155964 | 67.829148 | 65.345690 |
| timing jitter, AUC | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.997222 |
| timing jitter, median capacity (bit/s) | 130.785773 | 130.820032 | 129.028090 | 96.897550 | 71.503964 |

The capacity estimate falls by roughly a factor of two under either evasion,
while detection survives because size and timing carry independent structural
evidence. See `docs/EVALUATION.md` for the full tables and the entropy-only
column.

## Layout

```
src/capacity_tunnel/
  __init__.py   package surface
  channel.py    real UDP loopback channel (b32, timing, hardened)
  benign.py     DoH browsing, bulk download, interactive generators
  features.py   20 per-flow features over a 512-packet window
  capacity.py   capacity estimator (alphabet, entropy rate, timing capacity)
  detector.py   logistic regression, ROC/AUC, CV, dataset builder, sweeps
  cli.py        demo | gen | train | eval | detect
tests/          30 unittest cases
scripts/         make_figures.py, check_repro.py
docs/            DESIGN.md, THREAT_MODEL.md, EVALUATION.md
results/         results.json, model.json, dataset.npz, flow_scores.npz, figures
```
## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md): estimator derivation, feature definitions,
  parameter table, trade-offs.
- [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md): what the detector can and cannot
  see, evasion strategies, assumptions, ethics.
- [`docs/EVALUATION.md`](docs/EVALUATION.md): dataset construction, baselines, full
  tables, threats to validity.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): the design rules a patch has to keep.
- [`SECURITY.md`](SECURITY.md): what counts as a vulnerability here, and how to report.
- [`CHANGELOG.md`](CHANGELOG.md): release notes.
## Limitations

* The detection dataset is synthetic. Tunnel flows come from the same emission
  schedule the real channel executes, but the benign generators are models, not
  traces. AUC 1.000000 reflects a controlled separation and will not transfer to
  field traffic unchanged.
* The b32 mode is a payload channel: its symbols live in DNS-like label
  characters, which a metadata-only detector cannot read. Its metadata capacity
  is 0 by construction, and it is detected by transport regularity (constant
  40-byte datagrams, a 4 ms pace, strong periodicity), not by capacity. A DPI
  sensor that parses DNS labels would catch it trivially.
* The estimator assumes a small, stable symbol alphabet. If a tunnel uses
  per-packet codebooks, or if padding spreads sizes so thinly that no bin holds
  5%, the alphabet estimate under-counts capacity. The padding sweep shows this:
  median capacity falls from 135.752520 to 65.345690 bit/s even though the
  tunnel is still there.
* Only metadata is used. A tunnel that hides entirely inside payload bytes and
  keeps a benign size/timing profile is out of scope.
* All traffic is loopback. No real network is contacted, and no evasion against
  a production sensor is claimed.

## License

MIT, see [`LICENSE`](LICENSE).
