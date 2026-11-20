# Evaluation

## 1. Methodology

The evaluation answers three questions.

1. Can a metadata-only classifier separate the three reference tunnel modes from
   three benign traffic classes, and does it hold up out of fold?
2. How much of that separation is attributable to the capacity estimate rather
   than to raw entropy or a single shape statistic?
3. What happens to detection and to the capacity estimate when the tunnel pads,
   jitters its timing, or reverses direction?

Every number in this document is copied from `results/results.json`, which is
written by `make eval`. No value is entered by hand.

## 2. Dataset

| Property | Value |
| --- | --- |
| Flows | 480 (240 tunnel, 240 benign) |
| Flows per class | 80 |
| Classes | b32, timing, hardened, doh_browse, bulk, interactive |
| Window | 512 packets |
| Features | 20 |
| Seed | 2024 |

Tunnel flows come from `channel.emit_flow`, which executes the same
`encode()` schedule as the live UDP channel. Benign flows come from the three
generators in `benign.py`. Construction details that matter for interpretation:

* A quarter of tunnel flows are direction-inverted (downlink).
* A quarter of benign flows mix two activities in sequence.
* 12% of bulk flows are constant-size 62-byte keepalive trains at roughly 28
  packets/s, the closest benign analogue to a slow tunnel.
* Per-flow rate and timing jitter are randomized within the ranges given in
  `DESIGN.md`.

The dataset is synthetic. It is designed to test the estimator's behaviour under
controlled structure, not to estimate field accuracy.

## 3. Metrics

* AUC: Mann-Whitney area under the ROC curve over out-of-fold scores.
* FPR@95TPR: false positive rate at the smallest threshold that reaches 95%
  true positive rate.
* TPR@5FPR: true positive rate at 5% false positive rate.
* BER: bit error rate of the live channel, with missing bits counted as
  errors.
* Throughput: payload bits divided by the summed scheduled inter-packet gaps
  of the executed plan (protocol-normalized, not wall clock).

## 4. Protocol

Five-fold cross-validation, stratified by the six class labels so every fold
contains all modes. Standardization statistics are computed on the training fold
only. The classifier is L2-regularized logistic regression (lambda = 0.01, 1500
iterations, learning rate 0.5). The reported threshold is the out-of-fold
threshold that yields TPR = 0.95.

The evasion sweeps use a separate 70/30 stratified split. The model is trained on
70% and frozen; the benign reference is the 30% hold-out (72 flows). Each sweep
point adds 40 freshly seeded hardened flows.

## 5. Main result

| Metric | Value |
| --- | --- |
| AUC | 1.000000 |
| FPR@95TPR | 0.000000 |
| TPR@5FPR | 1.000000 |
| Threshold at TPR 0.95 | 2.647622 |
| Fold AUCs | 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| Fold FPR@95TPR | 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 |

Per-mode tasks (each mode against all 240 benign flows, using the same
out-of-fold scores):

| Mode | n tunnel | n benign | AUC | FPR@95TPR | TPR@5FPR |
| --- | --- | --- | --- | --- | --- |
| b32 | 80 | 240 | 1.000000 | 0.000000 | 1.000000 |
| timing | 80 | 240 | 1.000000 | 0.000000 | 1.000000 |
| hardened | 80 | 240 | 1.000000 | 0.000000 | 1.000000 |

## 6. Ablations

Each row is a separate 5-fold run with the same folds, model class, and
hyperparameters, restricted to the listed features.

| Feature set | n features | AUC | FPR@95TPR |
| --- | --- | --- | --- |
| entropy only | 2 | 0.902378 | 0.229167 |
| size only | 7 | 0.991667 | 0.037500 |
| timing only | 6 | 0.990278 | 0.037500 |
| capacity only | 6 | 1.000000 | 0.000000 |
| no direction | 19 | 1.000000 | 0.000000 |
| full | 20 | 1.000000 | 0.000000 |

The entropy-only baseline uses `size_entropy` and `timing_entropy`, the two
statistics a distribution-shape detector would use. It loses 0.097622 AUC and
0.229167 FPR relative to the full model. The capacity-only set, which excludes
every raw shape statistic except the alphabet uniformity, reaches 1.000000. The
difference between those two rows is the project's main empirical claim.

Direction is not load-bearing: removing `up_fraction` leaves the AUC at
1.000000. This matters because a third of the tunnel flows are downlink.

Per-class medians over the 80 flows of each class, from
`descriptive.per_class_medians`:

| Class | Size entropy (bit) | Entropy rate (bit/pkt) | Alphabet | Capacity (bit/s) | Capacity ratio (log10) |
| --- | --- | --- | --- | --- | --- |
| b32 | 0.000000 | 0.000000 | 1.0 | 0.000000 | -7.000000 |
| timing | 0.000000 | 0.997407 | 1.0 | 67.461660 | -3.382083 |
| hardened | 4.306747 | 4.087424 | 11.0 | 132.291314 | -3.511854 |
| doh_browse | 5.995779 | 0.929632 | 1.0 | 19.155562 | -3.791743 |
| bulk | 2.523607 | 2.678595 | 4.5 | 592.018391 | -3.835395 |
| interactive | 5.108657 | 2.174614 | 5.0 | 3.731769 | -3.210473 |

The DoH row is the clearest illustration of the estimator's premise: 5.995779
bits of raw size entropy, but a single recurring 16-byte bin and an entropy rate
of 0.929632 bit/packet. Bulk has a higher estimated capacity than two tunnel
modes, which is why the classifier does not rely on capacity alone.

Single-feature AUC over the same 480 flows (`descriptive.single_feature_auc`).
A value below 0.5 means tunnels have the smaller value; the model absorbs the
sign.

| Feature | AUC | Feature | AUC |
| --- | --- | --- | --- |
| timing_snr | 0.987274 | size_entropy_rate | 0.427483 |
| periodicity | 0.846823 | log_mean_size | 0.331128 |
| log_pkt_rate | 0.812344 | size_uniformity | 0.325677 |
| size_class_share | 0.761476 | iat_cv | 0.157517 |
| up_fraction | 0.731849 | timing_separation | 0.154792 |
| timing_capacity | 0.672917 | fano_factor | 0.147274 |
| log_capacity_bps | 0.531372 | timing_entropy | 0.108351 |
| log_capacity_ratio | 0.446979 | size_acf_peak | 0.083611 |
| size_alphabet | 0.437465 | size_entropy | 0.080625 |
| unique_ratio | 0.080339 | size_mi | 0.048924 |

No single feature is sufficient on its own except `timing_snr` at 0.987274, and
that feature is itself a capacity-model parameter. The combined model reaches
1.000000 by summing several weak-to-moderate structural signals.

## 7. Live channel measurement

Payload is 64 random bytes per mode, sent over 127.0.0.1.

| Mode | Packets sent | Data received | Scheduled duration (s) | Throughput (bit/s) | BER | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| b32 | 103 | 103 | 0.410233 | 1248.072450 | 0.000000 | match |
| timing | 513 | 512 | 8.238304 | 62.148717 | 0.000000 | match |
| hardened | 199 | 198 | 2.983149 | 171.630709 | 0.000000 | match |

The received count is one less than the sent count for timing and hardened
because the terminating flush datagram is not a data symbol. All three payloads
round-trip with an identical SHA-256 digest and zero bit errors.

The three throughputs illustrate the structural trade-off: b32 gets 5 bits per
40-byte datagram at a 4 ms pace, timing gets 1 bit per 15 ms average, and
hardened gets 2.585 bits per packet at 15 ms average. b32 is the fastest because
its information is in the payload, not in the shape.

## 8. Evasion sweeps

### 8.1 Size padding

Padding is uniform in `U(0, 96 x s)` bytes per datagram, applied to the hardened
mode. The detector is frozen at its training-time parameters. Benign reference:
72 held-out flows. Tunnel: 40 flows per point.

| s | AUC | AUC entropy-only | FPR@95TPR | Median capacity (bit/s) | Median timing separation |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.000000 | 0.833333 | 0.000000 | 135.752520 | 0.800131 |
| 1.5 | 1.000000 | 0.833333 | 0.000000 | 96.959326 | 0.799958 |
| 2 | 1.000000 | 0.833333 | 0.000000 | 65.402450 | 0.799453 |
| 4 | 1.000000 | 0.833333 | 0.000000 | 69.155964 | 0.799906 |
| 8 | 1.000000 | 0.833333 | 0.000000 | 67.829148 | 0.799132 |
| 16 | 1.000000 | 0.833333 | 0.000000 | 65.345690 | 0.799489 |

Padding does not move the detection AUC, because the timing codebook and packet
rate are untouched. It does move the capacity estimate: the 16-byte alphabet is
diluted by the widening padding envelope, and the median estimate drops by about
52% between 1x and 2x. The entropy-only baseline does not change at all
(0.833333), which is consistent with padding being a defence against entropy
statistics rather than against structural ones.

The non-monotonicity between 1.5x (96.959326) and 2x (65.402450) is consistent
with the fixed 5% share threshold: as the padding envelope widens, size bins drop
out of the effective alphabet one at a time. The estimate is therefore not
monotone in padding strength. It is reported as measured, not smoothed.

### 8.2 Timing jitter

Gaussian jitter with sigma `1.2 x s` ms is added to every codebook gap.

| s | AUC | AUC entropy-only | FPR@95TPR | Median capacity (bit/s) | Median timing separation |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.000000 | 0.833333 | 0.000000 | 130.785773 | 0.800233 |
| 2 | 1.000000 | 0.833333 | 0.000000 | 130.820032 | 0.791915 |
| 4 | 1.000000 | 0.833333 | 0.000000 | 129.028090 | 0.790126 |
| 8 | 1.000000 | 0.833333 | 0.000000 | 96.897550 | 0.822199 |
| 16 | 0.997222 | 0.832639 | 0.013889 | 71.503964 | 0.866285 |

At 16x jitter the timing separation is still 0.866285 because the 5 ms and 25 ms
clusters remain far apart relative to their means, but the within-cluster
variance inflates and the estimated capacity falls by about 45%. Detection AUC
falls only 0.002778. The capacity estimate is therefore the more sensitive
instrument for this evasion, which is the intended behaviour of the design.

## 9. Reproduction and determinism

```
make eval
```

Two consecutive runs produce byte-identical `results/results.json` (verified by
`diff`). All random processes are seeded: per-flow seeds in `build_dataset`, a
fixed fold permutation, a fixed permutation-null seed, and fixed jitter seeds in
the sweeps. Channel throughput and BER are deterministic because throughput is
computed from the executed schedule, not from a wall clock. Runtime is under
21 s of compute (22 s wall) on the reference host, of which 12 s is the three live
channels and 8 s the timing mode alone.

`make test` runs 30 unittest cases, including channel round-trips for all three
modes with BER exactly 0, hand-constructed feature cases with known entropy, and
a synthetic separable dataset for the detector. The complete run takes about
2 s and prints `OK`.

The figures are produced at exactly 1600 x 1000 pixels (8 x 5 inches at 200 dpi):
`results/fig_roc.png`, `results/fig_capacity_dist.png`, and
`results/fig_robustness.png`.

## 10. Threats to validity

Perfect separation. The headline AUC is exactly 1.000000. This does not mean
the detector is perfect; it means the synthetic classes are linearly separable by
the available structural features. The largest single-feature AUC in this dataset
is 0.987274 for `timing_snr` (measured on the same 480 flows), so a single
statistic already separates most of the data. The interesting result is the
ablation gap between entropy-only (0.902378) and capacity-only (1.000000), not
the last 0.002.

Synthetic benign traffic. The three generators are parametric models, not
captured traces. They do not contain browser multiplexing, TCP retransmission
patterns, or diurnal load variation. A field deployment would face harder
negatives, and the AUC would be lower. The keepalive sub-population and the
mixed-phase flows are an attempt to include some hard negatives, not a substitute
for real traffic.

Single-host channel. All measurements are loopback on one Windows host. The
5/25 ms codebook is easy to decode there because jitter is under 1 ms. A wide-area
path with queueing delay would raise BER and force a slower symbol clock, which
would change both throughput and the timing capacity estimate.

Estimator bias. The 5% share threshold and 16-byte grid are fixed. They
penalize heavy padding (visible in section 8.1) and would under-count a codebook
whose classes are closer than 16 bytes. The timing gate rejects any codebook
with less than 20% relative separation even if a well-clocked receiver could
decode it. These are conservative choices, but they are choices.

Window size. All results use 512-packet windows. A shorter window would
increase variance in the alphabet and entropy-rate estimates; a longer window
would blur non-stationary traffic. Neither is measured here beyond informal
checks.

Class balance. The dataset is exactly balanced (80 per class). Real traffic
is heavily imbalanced, so the reported FPR at a fixed TPR would correspond to a
different absolute false-alarm count.

No calibration. Scores are decision values, not probabilities. The 0.5
threshold used by `detect` is arbitrary; the reported operating point is the
out-of-fold threshold at TPR 0.95.

No adversarial retraining. The evasion sweeps freeze the detector. An
operator who retrains on padded or jittered samples would likely recover AUC,
which is why the capacity estimate, not the classifier score, is the durable
signal.

## 11. What would change the conclusion

* Benign traces with regular pacing and stable small alphabets (paced media,
  gaming, industrial polling) would raise the false positive rate and force a
  capacity-ratio threshold rather than a raw capacity threshold.
* A generative adversary that matches benign size and gap distributions while
  still decoding would collapse the structural gap. Measuring that gap is the
  natural next experiment.
* TCP tunnels would add ACK and congestion-control structure that this channel
  does not model.

None of these is tested here. The reported numbers should be read as an upper
bound on performance for this channel family, not as a field estimate.
