# Design

## 1. Scope

The detector observes packet metadata for one flow over a window of at most 512
packets: timestamp, direction, and size. It does not parse payloads, ports, or
addresses. The premise is a NetFlow/IPFIX-style collector on encrypted traffic,
where the only available signal is shape.

The system has two halves. Offline, a reference channel generates tunnel flows
and three benign generators produce negative flows. Online, a 20-dimensional
feature vector is computed per flow and scored by a logistic-regression model.
The capacity estimator is the part that distinguishes this detector from an
entropy heuristic.

## 2. Architecture

```
                    reference emitter (real UDP, 127.0.0.1)
                    +--------------------------------------+
                    | channel.encode() -> packet schedule  |
                    |   b32 | timing | hardened             |
                    +------------------+-------------------+
                                       |
                    +------------------+-------------------+
                    | run_channel(): send/recv threads,    |
                    | SHA-256 verify, BER, throughput      |
                    +------------------+-------------------+
                                       |
   benign generators                   |  emit_flow()
   doh_browse | bulk | interactive     |  (same schedule,
        |                              |   seeded jitter)
        +--------------+---------------+
                       |
                 flow = [(t, dir, size), ...]
                       |
              features.extract(flow, window=512)
                       |
          +------------+-------------+
          |                          |
   capacity.estimate_flow     shape statistics
   (alphabet, H_rate, C_t)    (CV, Fano, ACF, ...)
          |                          |
          +------------+-------------+
                       |
              20-float vector
                       |
        standardize (train-fold statistics)
                       |
        logistic regression (L2, full-batch GD)
                       |
              score -> threshold -> verdict
```

Evaluation wraps this with stratified 5-fold CV, six feature-set ablations, a
real channel measurement per mode, and two evasion sweeps.

## 3. Reference channel

The channel is a unidirectional UDP stream from a client to a server on an
ephemeral loopback port. Each datagram starts with an 8-byte header:

```
 0        2   3    4      6        8
 +--------+---+----+------+--------+-------------------+
 | "CT"   |mod|flag| seq  | plen   | body / padding     |
 +--------+---+----+------+--------+-------------------+
```

`flag` bit 0 marks the last data datagram (b32), bit 1 marks the flush datagram
that closes a timing symbol. `seq` allows the receiver to sort reordered
datagrams before decoding; a missing sequence number is not reconstructed, so
loss shows up as bit errors rather than silent corruption.

| Mode | Symbol | Size | Gap | Payload location |
| --- | --- | --- | --- | --- |
| b32 | 5 bits, base32 character | 40 B constant | 4 ms | one character per datagram |
| timing | 1 bit | 300 B constant | 5 ms = 0, 25 ms = 1 | none (timing only) |
| hardened | base-6 digit, 2.585 bits | 360/760/1160 B + U(0, 96) | 5/25 ms + N(0, 1.2 ms) | none (shape only) |

The flush datagram exists because the last symbol of a timing or hardened flow
is the gap *after* the last data datagram; the receiver needs one more arrival
to measure it. Because a late timestamp on that final arrival would corrupt a
whole symbol, the flush repeats the final timing class in its own size (300 vs
304 bytes for timing, 360 vs 760 bytes for hardened). The decoder prefers the
redundant copy and falls back to the gap if the flush is missing. The receiver
stops on the flush flag or after 350 ms of silence, so payload length does not
have to be known in advance, and the flush is excluded from detection flows
because it is a protocol terminator rather than a data symbol.

Bit error rate is computed over the full transmitted bit string, with missing
bits counted as errors. Throughput is payload bits divided by the summed
scheduled gaps of the executed plan. This is a property of the channel, not of
the host scheduler; wall-clock throughput is measured separately in `demo` and
is not used in any reported metric.

## 4. Dataset construction

480 flows, 80 per class, window 512, seed 2024. Each flow has an independent
seed derived from the class index and the flow index.

Tunnel flows are produced by `channel.emit_flow`, which calls the same
`encode()` used by the real transport. The dataset therefore shares the
emission model with the working channel; only socket timing is absent (a
0.25 ms capture jitter is added). Per-flow rate is drawn from U(0.80, 1.25) and
timing-jitter multiplier from U(0.75, 1.25). A quarter of tunnel flows have their
direction inverted to model inbound tasking, so `up_fraction` is not a label.

Benign flows are produced by the three generators in `benign.py`. Two design
choices make them non-trivial negatives:

* 25% of benign flows are two-phase mixtures (for example DoH browsing followed
  by a bulk download), which breaks single-activity assumptions.
* 12% of bulk flows are idle keepalive trains: a constant 62-byte uplink packet
  every 35 ms. This is structurally close to a slow constant-rate tunnel and is
  the hardest negative in the set.

## 5. Features

All features are computed from the first 512 packets of a flow, in timestamp
order. Names are the canonical order in `features.FEATURE_NAMES`.

| # | Name | Definition |
| --- | --- | --- |
| 1 | size_entropy | Shannon entropy of sizes quantized to 16-byte bins |
| 2 | size_entropy_rate | Joint Markov-1 entropy rate H(X) - I(X_i; X_i+1) |
| 3 | size_mi | Null-calibrated mutual information of consecutive joint symbols |
| 4 | size_alphabet | Number of recurring 16-byte bins (>= 5% share) |
| 5 | size_uniformity | H(size) / log2(alphabet), 0 when alphabet <= 1 |
| 6 | timing_entropy | Entropy of IATs in 16 log-spaced bins, 0.4 ms to 4 s |
| 7 | timing_snr | Between/within variance ratio of the 2-means IAT partition |
| 8 | timing_separation | abs(m1 - m2) / max(m1, m2) for the two IAT clusters |
| 9 | timing_capacity | min(1, 0.5 log2(1 + SNR)) when separation >= 0.20, else 0 |
| 10 | iat_cv | std(IAT) / mean(IAT) |
| 11 | fano_factor | Var/mean of packet counts in 10 ms bins |
| 12 | up_fraction | Fraction of packets with direction 0 (uplink) |
| 13 | size_acf_peak | max abs autocorrelation of size sequence, lags 1..8 |
| 14 | periodicity | max abs ACF of the 1 ms-scale event train, lags 2..64 |
| 15 | unique_ratio | Distinct sizes / packets |
| 16 | log_mean_size | log10(mean packet size) |
| 17 | log_pkt_rate | log10(packets per second) |
| 18 | log_capacity_bps | log10(1 + capacity_bps) |
| 19 | log_capacity_ratio | log10(capacity / wire bitrate + 1e-7) |
| 20 | size_class_share | Share of the most common 16-byte bin |

The event train used by `periodicity` is binned at `max(0.5 ms, median IAT / 4)`
with 256 bins, so the score is comparable across flow rates.

## 6. Capacity estimator

### 6.1 Effective alphabet

Sizes are quantized to 16-byte bins. Bins holding at least 5% of the window are
kept, ranked by mass, capped at 12; every other packet is mapped to one residual
symbol. The rationale is decodability: a receiver needs a small, repeatable
alphabet. A long tail of one-off sizes cannot be a codebook, so it must not be
counted as capacity. The 5% threshold and 16-byte width are conventions
justified by the codebooks used here (the hardened classes are 400 bytes apart),
not universal constants.

### 6.2 Size entropy rate

For joint symbols `X_i = (size_class_i, timing_class_i)`:

```
H(X)    = -sum_x p(x) log2 p(x)                        (Miller-Madow corrected)
I(Xi;Xj)= H(Xi) + H(Xj) - H(Xi,Xj) - E[I under 16 seeded permutations of Xj]
H_rate  = max(0, H(X) - I)
```

The permutation null estimates and removes the upward bias of the plug-in
mutual-information estimator on sparse contingency tables. The classical bias
term for a K x K table is (K-1)^2 / (2 N ln 2), which is 0.17 bits for K = 12
and N = 512 and 0.88 bits for K = 26; on short windows that bias is the same
order as the signal.

### 6.3 Timing capacity

Inter-arrival times are log-transformed (`log1p(iat / 1 ms)`) and split by a
deterministic 1-D 2-means with initial centres at the 25th and 75th percentiles.
Working in log space keeps the partition stable across rates. With cluster means
`m1, m2`, cluster variances `v1, v2`, and weights `w1, w2`:

```
between = sum_j wj (mj - mean)^2 / n
within  = sum_j wj vj / n
SNR     = between / within
sep     = |m1 - m2| / max(m1, m2)
C_t     = min(1, 0.5 log2(1 + SNR))   if sep >= 0.20, else 0
```

The separation gate rejects the spurious split of a unimodal jitter
distribution: 0.3 ms of jitter around a 4 ms pace has `sep` near 0.10 and is
correctly worth zero timing capacity, while the reference 5/25 ms codebook has
`sep = 0.80`. The clamp at 1 bit follows from the binary partition: a two-cluster
quantization cannot carry more than one bit per packet regardless of SNR.

### 6.4 Combined estimate

```
pkt_rate     = (n - 1) / (t_last - t_first)
capacity_bps = pkt_rate * min(H_rate, 1 + C_t)
capacity_ratio = capacity_bps / (8 * sum(sizes) / duration)
```

The `min` is a conservative cap: the per-packet symbol budget is bounded both by
the observed alphabet's entropy rate and by one bit plus the timing capacity.
`capacity_ratio` normalizes by the wire bitrate and is reported as a
diagnostic. It is not sufficient on its own: bulk download has a median capacity
of 592.018391 bit/s, higher than the timing (67.461660) and hardened (132.291314)
tunnels,
because its packets are regular and its sizes repeat. The detector therefore
also uses alphabet uniformity, timing separation, and the packet rate.

### 6.5 Assumptions and failure modes

The estimate is an upper bound on what an observer can attribute to covert
symbols, not a proof that a tunnel exists.

* The alphabet is assumed stationary over the window. A per-packet codebook, or
  a codebook that rotates mid-flow, is under-counted.
* The 16-byte grid and 5% share threshold are fixed. Padding that spreads sizes
  across many bins lowers the measured capacity even when the tunnel still
  decodes. In the padding sweep the median estimate falls from 135.752520 to
  65.345690 bit/s at 16x padding.
* The timing gate requires 20% relative separation. A codebook using 5 ms and
  6 ms gaps would be rejected even though it is decodable with a good clock.
* Two clusters are assumed. A 4-ary timing code would be truncated to 1 bit per
  packet by the clamp.
* Benign traffic with genuinely regular pacing and a small stable size alphabet
  (for example a paced media stream) will score high. The keepalive negatives in
  this dataset are rejected only because their gaps are unimodal, so `C_t = 0`
  and their capacity is zero.
* Only one flow is scored. Aggregating over many flows, or over a longer window,
  changes both variance and the alphabet estimate.

## 7. Detector

Features are standardized with training-fold mean and standard deviation.
Logistic regression is fit by full-batch gradient descent on mean log loss with
L2 penalty 0.01, learning rate 0.5, at most 1500 iterations, early stop when the
loss changes by less than 1e-12 over 50 iterations. The model is linear; this is
a deliberate choice for interpretability and for a clean ablation, not a claim
that the boundary is linear.

Cross-validation is 5-fold, stratified by the six class labels (not just the
binary label), so every fold sees all modes. Out-of-fold scores feed AUC and the
FPR@95TPR operating point. The evasion sweeps use a separate 70/30 stratified
split: the model is frozen after training on 70%, and the 30% benign hold-out is
the reference distribution.

## 8. Complexity

Let n be the window size (<= 512), d = 20 features, k = 5 folds, I = 1500
iterations.

| Stage | Cost |
| --- | --- |
| Feature extraction | O(n + 16n) for ACF and permutation MI, plus O(n log n) sorting |
| 2-means | O(25 n) |
| Logistic fit | O(I n d) |
| 5-fold CV | O(k I n d) |
| Full eval | 480 flows, about 20 s total; 12 s is the real timing channel |

The dominant evaluation cost is the real channel, not the model. The timing mode
sends one bit per 15 ms on average, so a 64-byte payload takes about 8 s.

## 9. Parameters

| Parameter | Value | Where |
| --- | --- | --- |
| Window | 512 packets | features, dataset |
| Size bin | 16 bytes | capacity, features |
| Min class share | 0.05 | capacity |
| Max size classes | 12 | capacity |
| Min timing separation | 0.20 | capacity |
| Timing clamp | 1 bit/packet | capacity |
| MI permutations | 16 | capacity |
| L2 | 0.01 | detector |
| Learning rate / iters | 0.5 / 1500 | detector |
| CV folds | 5 | detector |
| Dataset seed | 2024 | cli |
| Flows per class | 80 | cli |
| Channel payload | 64 B | cli eval |
| Padding sweep | 1, 1.5, 2, 4, 8, 16 | detector |
| Jitter sweep | 1, 2, 4, 8, 16 | detector |

## 10. Trade-offs

* Alphabet granularity. Coarse bins make the estimate stable but blind to
  fine-grained codebooks; fine bins inflate entropy on continuous traffic. 16
  bytes matched the 24-byte class gaps of the reference channel.
* Recurring-share filter. The 5% threshold suppresses benign tails, but it
  also penalizes a tunnel that pads heavily. This is visible in the padding
  sweep and is a deliberate conservative bias.
* Permutation null. Sixteen permutations remove most of the sparse-table
  bias at negligible cost; a larger null would be smoother but does not change
  any reported decision.
* Linear detector. A gradient-boosted or kernel model would likely fit the
  same data equally well, but the linear model makes the feature ablations
  meaningful and the artifact small.
* Per-flow view. A production sensor would aggregate over hosts and time.
  This project deliberately evaluates the single-flow case, which is the harder
  and more honest unit for a capacity estimate.
