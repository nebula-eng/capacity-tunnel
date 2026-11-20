# Threat model

## 1. What is being modelled

A covert tunnel carries a payload between two cooperating endpoints while
trying to look like ordinary traffic to a passive observer. The observer is a
flow-metadata sensor: it sees, for each flow, a sequence of `(timestamp,
direction, size)` records and nothing else. This is the position of a NetFlow or
IPFIX exporter on encrypted traffic, or of a switch ACL with sampling.

The question is whether the *shape* required to encode symbols is visible even
when the marginal distributions are padded toward benign.

## 2. Adversary capabilities

The tunnel operator controls:

* the mapping from payload bits to packet sizes and inter-arrival gaps, subject
  to a decoder that must recover the payload with an acceptable error rate;
* padding: extra bytes appended to each datagram, up to the path MTU;
* timing jitter added to each gap, bounded by the decoder's tolerance;
* direction: exfiltration (uplink) or inbound tasking (downlink), and mixtures;
* rate: the base pacing of the symbol clock;
* payload length and session duration.

The operator does not control the observer's access to metadata, and cannot make
a symbol alphabet both large and reliably decodable without paying for it in
packets or time. That trade-off is the detector's leverage.

## 3. Defender view

The detector sees one flow at a time, up to 512 packets, and computes 20
features. It has no payload access, no DNS parsing, no endpoint reputation, and
no ground-truth labels at inference time. It was trained on the three reference
tunnel modes and three benign generators described in `DESIGN.md`.

The reported operating point is FPR at TPR = 0.95. A production deployment would
also need a score calibration step and a triage queue; neither is modelled here.

## 4. Evasion strategies considered

| Strategy | Implementation | Measured effect |
| --- | --- | --- |
| Size padding | uniform U(0, 96 x s) bytes added per datagram, s in 1..16 | AUC stays 1.000000; median capacity estimate falls from 135.752520 to 65.345690 bit/s |
| Timing jitter | Gaussian sigma 1.2 x s ms on each gap, s in 1..16 | AUC 0.997222 at s = 16; median capacity estimate falls from 130.785773 to 71.503964 bit/s |
| Direction inversion | 25% of tunnel flows flipped to downlink | Included in training; `no_direction` ablation still reaches AUC 1.000000 |
| Rate shaping | per-flow rate multiplier U(0.80, 1.25) | Included in training |
| Alphabet dilution | padding spreads sizes across bins | Capacity estimate falls, but the flow still scores as tunnel because timing structure survives |
| Payload-only channel | b32: symbols in base32 label characters, 40-byte datagrams | Metadata capacity is 0 by construction; detected by constant size, 250 packets/s and periodicity |

Two observations follow.

First, single-axis evasion does not defeat the detector in this dataset. Padding
moves the capacity estimate but leaves the timing codebook and the packet rate
intact; jitter blurs the codebook but leaves size classes and rate intact. The
detector combines independent structural axes, so the adversary has to suppress
all of them at once.

Second, the measured degradation under jitter is small in AUC terms
(1.000000 to 0.997222) but large in capacity terms (about 45% loss). That is the
expected shape of the trade-off: a tunnel that hides its timing structure has
less timing capacity left to carry payload. A detector that keyed only on
capacity would see the same flow become less suspicious as it becomes less
useful.

## 5. What the detector cannot see

* Payload-encoded channels. b32 hides symbols in DNS-like labels. A sensor
  that parses those labels would catch it immediately; a metadata-only sensor
  sees only regular 40-byte datagrams. This project detects b32 by transport
  regularity, not by reading the payload.
* Tunnels that mimic benign shape. A generative adversary that reproduces
  lognormal sizes and Poisson arrivals while still decoding a payload would
  defeat a shape detector. The dataset contains no such adversary.
* Multi-flow tunnels. Splitting a payload across many short flows, each
  individually benign-looking, is out of scope. The detector scores one flow at
  a time.
* TCP tunnels. Congestion control, ACKs and retransmissions add structure
  that this channel does not model. A real TCP tunnel may look different in both
  directions.
* Encrypted or compressed payload effects on size. The generators model
  size distributions directly; they do not model compression ratios.

## 6. Assumptions

1. The observer can measure per-packet timestamps with sub-millisecond accuracy.
   Coarse sampling (for example 1:1000 NetFlow) weakens the timing features.
2. The symbol alphabet is stationary over the 512-packet window.
3. The tunnel does not deliberately inject decoy packets to break the alphabet
   estimate.
4. Benign traffic in deployment resembles the three generators at the level of
   shape statistics. This is the weakest assumption; the generators are models,
   not traces.
5. The classifier is trained on traffic from the same channel family it will
   score. Cross-family generalization is not measured.

## 7. Ethics and legal

All experiments run on 127.0.0.1 with ephemeral ports. The channel binds to the
loopback interface only; no external host is contacted, no name resolution is
performed, and no real data is transmitted. The "payload" is random bytes.

The code is a measurement and detection artifact for defensive research. It is
not a deployment-ready exfiltration tool, and the channel is deliberately simple
so that its statistics can be audited. Running it against networks you do not
own or administer would be unlawful in most jurisdictions and is outside the
scope of this repository.

The evaluation numbers describe synthetic traffic on one host. They are not a
claim about detection performance on production networks, and they should not be
quoted as such.
