"""Real covert channel over UDP loopback, used as the reference emitter.

The channel sends a payload from a client to a server on 127.0.0.1 through an
ephemeral UDP port.  The server reassembles the payload and the caller compares
SHA-256 digests.  Three modes exist, each with a different covert structure:

``b32``      5-bit symbols carried in DNS-label-like characters.  Datagram size
             is constant (40 bytes), so packet metadata carries no symbol.
``timing``   1 bit per packet, encoded as a 5 ms (0) or 25 ms (1) inter-packet
             gap.  Datagram size is constant (300 bytes); the terminator
             repeats the final bit in its size so a late timestamp cannot
             corrupt the last symbol.
``hardened`` 6-ary symbols (3 size classes x 2 timing classes), 2.58 bits per
             packet, with uniform size padding (up to 96 bytes) and Gaussian
             timing jitter attached to defeat coarse entropy tests.

The sender emits packets according to a deterministic schedule produced by
:func:`encode`.  The same schedule function feeds :func:`emit_flow`, which is
used to build the detection dataset.  This keeps the measured channel and the
dataset on one code path; the only difference is that the real transport adds
socket-level timing noise.

Everything binds to 127.0.0.1.  No external network traffic is generated.
"""

from __future__ import annotations

import hashlib
import math
import socket
import struct
import threading
import time
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

MODES: Tuple[str, ...] = ("b32", "timing", "hardened")

B32_ALPHABET = "abcdefghijklmnopqrstuvwxyz234567"

_MODE_ID = {"b32": 1, "timing": 2, "hardened": 3}
_MAGIC = b"CT"
_HDR = struct.Struct("!2sBBHH")  # magic, mode, flags, seq, payload length
HEADER_LEN = _HDR.size

FLAG_LAST = 1
FLAG_FLUSH = 2

B32_PACKET_SIZE = 40
TIMING_PACKET_SIZE = 300
HARDENED_SIZES: Tuple[int, int, int] = (360, 760, 1160)

B32_GAP_MS = 4.0
TIMING_GAP_SHORT_MS = 5.0
TIMING_GAP_LONG_MS = 25.0
TIMING_DECODE_MS = 15.0

PAD_MAX_BYTES = 96.0
JITTER_MS = 1.2
B32_JITTER_MS = 0.3
CAPTURE_JITTER_MS = 0.25

# Size-class boundaries for the hardened receiver.  Midpoints between the
# nominal class centres keep the 96-byte padding envelope separated.
HARDENED_THRESHOLDS = (
    0.5 * (HARDENED_SIZES[0] + HARDENED_SIZES[1]),
    0.5 * (HARDENED_SIZES[1] + HARDENED_SIZES[2]),
)

Flow = List[Tuple[float, int, int]]
"""A flow is a list of (timestamp_seconds, direction, size_bytes).

direction 0 means client-to-server (uplink), 1 means server-to-client.
"""


class PacketSpec(NamedTuple):
    offset_s: float
    size: int
    data: bytes
    gap_s: float


class ChannelResult(NamedTuple):
    mode: str
    payload_bytes: int
    packets_sent: int
    packets_received: int
    bits_sent: int
    bit_errors: int
    ber: float
    sha256_match: bool
    nominal_duration_s: float
    wall_duration_s: float
    throughput_bps: float
    decoded_sha256: str


def bytes_to_bits(data: bytes) -> List[int]:
    out: List[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            out.append((byte >> shift) & 1)
    return out


def bits_to_bytes(bits: Sequence[int], nbytes: int) -> bytes:
    padded = list(bits) + [0] * max(0, nbytes * 8 - len(bits))
    out = bytearray()
    for i in range(nbytes):
        value = 0
        for j in range(8):
            value = (value << 1) | (padded[i * 8 + j] & 1)
        out.append(value)
    return bytes(out)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _packet(mode: str, flags: int, seq: int, plen: int, body: bytes, size: int) -> bytes:
    raw = _HDR.pack(_MAGIC, _MODE_ID[mode], flags, seq & 0xFFFF, plen & 0xFFFF) + body
    if len(raw) > size:
        raise ValueError("packet body exceeds declared size")
    return raw.ljust(size, b"\x00")


def _to_base6(data: bytes) -> List[int]:
    value = int.from_bytes(data, "big")
    digits: List[int] = []
    while value:
        digits.append(value % 6)
        value //= 6
    digits.reverse()
    return digits or [0]


def _from_base6(digits: Sequence[int], nbytes: int) -> bytes:
    value = 0
    for digit in digits:
        value = value * 6 + digit
    return (value % (1 << (8 * nbytes))).to_bytes(nbytes, "big")


def encode(
    mode: str,
    payload: bytes,
    seed: int = 0,
    rate_scale: float = 1.0,
    pad_scale: float = 1.0,
    jitter_scale: float = 1.0,
) -> Tuple[List[PacketSpec], Dict[str, int]]:
    """Build the emission schedule for ``payload``.

    The schedule is deterministic given ``seed`` and the tuning parameters.  It
    is the single source of truth for both the real transport and the synthetic
    flow generator used to build the detection dataset.
    """
    if mode not in MODES:
        raise ValueError("unknown mode: %r" % (mode,))
    if rate_scale <= 0:
        raise ValueError("rate_scale must be positive")

    rng = np.random.default_rng(seed)
    plen = len(payload)
    bits = bytes_to_bits(payload)
    specs: List[PacketSpec] = []
    offset = 0.0

    def gap_with(base_ms: float, jitter_ms: float, floor_frac: float = 0.55) -> float:
        base = base_ms / 1000.0 / rate_scale
        jitter = rng.normal(0.0, jitter_ms / 1000.0) * jitter_scale
        return max(base * floor_frac, base + jitter)

    if mode == "b32":
        nsym = (len(bits) + 4) // 5
        for i in range(nsym):
            chunk = bits[i * 5 : i * 5 + 5]
            chunk = chunk + [0] * (5 - len(chunk))
            symbol = 0
            for bit in chunk:
                symbol = (symbol << 1) | bit
            flags = FLAG_LAST if i == nsym - 1 else 0
            data = _packet(mode, flags, i, plen, bytes([symbol]), B32_PACKET_SIZE)
            gap = gap_with(B32_GAP_MS, B32_JITTER_MS)
            specs.append(PacketSpec(offset, B32_PACKET_SIZE, data, gap))
            offset += gap
        return specs, {"nbits": len(bits), "nbytes": plen, "nsym": nsym}

    if mode == "timing":
        for i, bit in enumerate(bits):
            base = TIMING_GAP_LONG_MS if bit else TIMING_GAP_SHORT_MS
            gap = gap_with(base, 0.6)
            data = _packet(mode, 0, i, plen, b"\x00", TIMING_PACKET_SIZE)
            specs.append(PacketSpec(offset, TIMING_PACKET_SIZE, data, gap))
            offset += gap
        # The terminator repeats the final bit in its size (300 or 304 bytes).
        last_bit = bits[-1] if bits else 0
        flush_size = TIMING_PACKET_SIZE + (4 if last_bit else 0)
        flush = _packet(mode, FLAG_FLUSH, len(bits), plen, b"\x00", flush_size)
        specs.append(PacketSpec(offset, flush_size, flush, 0.0))
        return specs, {"nbits": len(bits), "nbytes": plen, "nsym": len(bits)}

    digits = _to_base6(payload)
    for i, digit in enumerate(digits):
        size_idx, timing_bit = divmod(digit, 2)
        size = HARDENED_SIZES[size_idx] + int(round(rng.uniform(0.0, PAD_MAX_BYTES * pad_scale)))
        base = TIMING_GAP_LONG_MS if timing_bit else TIMING_GAP_SHORT_MS
        gap = gap_with(base, JITTER_MS)
        flags = 0
        data = _packet(mode, flags, i, plen, b"\x00", size)
        specs.append(PacketSpec(offset, size, data, gap))
        offset += gap
    # The terminator repeats the final timing class in its size.
    flush_size = HARDENED_SIZES[1] if digits[-1] % 2 else HARDENED_SIZES[0]
    flush = _packet(mode, FLAG_FLUSH, len(digits), plen, b"\x00", flush_size)
    specs.append(PacketSpec(offset, flush_size, flush, 0.0))
    return specs, {"nbits": len(bits), "nbytes": plen, "nsym": len(digits)}


class _Receiver(threading.Thread):
    """UDP listener that timestamps datagrams and stops on a terminal flag.

    The listener exits after ``idle_s`` seconds without traffic (the longest
    encoded gap is 25 ms) or at the hard cap, so payload length does not have to
    be known in advance and a long transfer is never truncated by a fixed
    deadline.
    """

    def __init__(self, idle_s: float = 0.35, hard_s: float = 30.0) -> None:
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.05)
        self.port = int(self.sock.getsockname()[1])
        self.idle_s = idle_s
        self.hard_s = hard_s
        self.records: List[Tuple[int, float, int, int, bytes]] = []
        self.error: Optional[BaseException] = None
        self._stop = threading.Event()

    def run(self) -> None:  # pragma: no cover - exercised through the socket path
        hard_deadline = time.perf_counter() + self.hard_s
        last_packet = time.perf_counter()
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                if now - last_packet > self.idle_s or now > hard_deadline:
                    break
                try:
                    raw, _ = self.sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                last_packet = time.perf_counter()
                if len(raw) < HEADER_LEN:
                    continue
                magic, mode_id, flags, seq, plen = _HDR.unpack(raw[:HEADER_LEN])
                if magic != _MAGIC or mode_id not in _MODE_ID.values():
                    continue
                self.records.append((seq, last_packet, len(raw), flags, raw))
                if flags & (FLAG_LAST | FLAG_FLUSH):
                    break
        except BaseException as exc:  # pragma: no cover - defensive
            self.error = exc
        finally:
            try:
                self.sock.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()


def _size_class(size: int) -> int:
    if size < HARDENED_THRESHOLDS[0]:
        return 0
    if size < HARDENED_THRESHOLDS[1]:
        return 1
    return 2


def _decode(
    mode: str, records: Sequence[Tuple[int, float, int, int, bytes]], meta: Dict[str, int]
) -> Tuple[List[int], int]:
    """Turn received datagrams into a bit stream.

    Datagrams are sorted by sequence number so that a reordered arrival does not
    shift symbols.  Missing sequence numbers are not reconstructed; their bits
    are reported as errors through the caller's BER computation.
    """
    nbits = int(meta["nbits"])
    if not records:
        return [], 0

    ordered = sorted(records, key=lambda r: (r[0], r[1]))
    data_records = [r for r in ordered if not (r[3] & FLAG_FLUSH)]

    if mode == "b32":
        bits: List[int] = []
        for _, _, _, _, raw in data_records:
            symbol = raw[HEADER_LEN] & 0x1F
            for shift in range(4, -1, -1):
                bits.append((symbol >> shift) & 1)
        return bits[:nbits], len(data_records)

    # Timing classes come from the gap to the next datagram in sequence order,
    # including the flush datagram that closes the last symbol.
    gaps: List[float] = []
    for i in range(len(ordered) - 1):
        gaps.append(ordered[i + 1][1] - ordered[i][1])

    flush_records = [r for r in ordered if r[3] & FLAG_FLUSH]
    flush_size = flush_records[0][2] if flush_records else -1

    if mode == "timing":
        bits = []
        for i in range(len(data_records)):
            gap_ms = gaps[i] * 1000.0 if i < len(gaps) else TIMING_GAP_SHORT_MS
            bit = 1 if gap_ms > TIMING_DECODE_MS else 0
            if i == len(data_records) - 1 and flush_size >= 0:
                # Redundant final bit carried by the terminator size.
                bit = 1 if flush_size > TIMING_PACKET_SIZE else 0
            bits.append(bit)
        return bits[:nbits], len(data_records)

    digits: List[int] = []
    for i in range(len(data_records)):
        _, _, size, _, _ = data_records[i]
        gap_ms = gaps[i] * 1000.0 if i < len(gaps) else TIMING_GAP_SHORT_MS
        timing_bit = 1 if gap_ms > TIMING_DECODE_MS else 0
        if i == len(data_records) - 1 and flush_size >= 0:
            timing_bit = 1 if _size_class(flush_size) == 1 else 0
        digits.append(_size_class(size) * 2 + timing_bit)
    return bytes_to_bits(_from_base6(digits, int(meta["nbytes"]))), len(data_records)


def run_channel(
    mode: str,
    payload: Optional[bytes] = None,
    seed: int = 0,
    timeout_s: float = 30.0,
    rate_scale: float = 1.0,
    pad_scale: float = 1.0,
) -> ChannelResult:
    """Send ``payload`` through a real loopback UDP channel and verify it."""
    if payload is None:
        payload = np.random.default_rng(seed).integers(0, 256, 32, dtype=np.uint8).tobytes()
    specs, meta = encode(mode, payload, seed=seed, rate_scale=rate_scale, pad_scale=pad_scale)

    receiver = _Receiver(idle_s=0.35, hard_s=timeout_s)
    receiver.start()

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.connect(("127.0.0.1", receiver.port))
    wall_start = time.perf_counter()
    try:
        # Gaps are emitted relative to the previous datagram.  Absolute
        # deadlines would let one late wakeup compress the following gap, which
        # corrupts timing symbols; relative gaps keep each symbol independent.
        for idx, spec in enumerate(specs):
            if idx > 0:
                delay = specs[idx - 1].gap_s
                if delay > 0:
                    time.sleep(delay)
            tx.send(spec.data)
    finally:
        tx.close()
    wall_duration = time.perf_counter() - wall_start

    receiver.join(timeout_s + 1.0)
    receiver.stop()
    if receiver.error is not None:
        raise RuntimeError("receiver failed: %r" % (receiver.error,))

    records = list(receiver.records)
    decoded_bits, n_rx = _decode(mode, records, meta)
    decoded = bits_to_bytes(decoded_bits, len(payload))

    src_bits = bytes_to_bits(payload)
    cmp_len = max(len(src_bits), len(decoded_bits))
    errors = 0
    for i in range(cmp_len):
        a = src_bits[i] if i < len(src_bits) else 0
        b = decoded_bits[i] if i < len(decoded_bits) else 0
        if a != b:
            errors += 1

    nominal = specs[-1].offset_s if specs else 0.0
    throughput = (len(payload) * 8.0 / nominal) if nominal > 0 else 0.0
    return ChannelResult(
        mode=mode,
        payload_bytes=len(payload),
        packets_sent=len(specs),
        packets_received=n_rx,
        bits_sent=len(src_bits),
        bit_errors=errors,
        ber=errors / cmp_len if cmp_len else 0.0,
        sha256_match=sha256_hex(decoded) == sha256_hex(payload),
        nominal_duration_s=nominal,
        wall_duration_s=wall_duration,
        throughput_bps=throughput,
        decoded_sha256=sha256_hex(decoded),
    )


def payload_for_mode(mode: str, n_packets: int, seed: int = 0) -> bytes:
    """Pick a payload length that yields at least ``n_packets`` datagrams."""
    if mode == "b32":
        nbits = int(math.ceil(n_packets * 5 / 8.0)) * 8
        nbytes = max(1, nbits // 8)
    elif mode == "timing":
        nbytes = max(1, int(math.ceil((n_packets - 1) / 8.0)))
    else:
        bits_per_packet = math.log2(6.0)
        nbytes = max(1, int(math.ceil(n_packets * bits_per_packet / 8.0)) + 1)
    rng = np.random.default_rng(seed + 9176)
    return rng.integers(0, 256, nbytes, dtype=np.uint8).tobytes()


def emit_flow(
    mode: str,
    seed: int = 0,
    n_packets: int = 512,
    pad_scale: float = 1.0,
    jitter_scale: float = 1.0,
    rate_scale: float = 1.0,
    capture_jitter: bool = True,
) -> Flow:
    """Emit a tunnel flow as (timestamp, direction, size) records.

    Uses the same :func:`encode` schedule as :func:`run_channel`; timestamps are
    the planned offsets plus optional capture-point jitter.  All records are
    uplink (direction 0), matching a client pushing data through the tunnel.
    """
    payload = payload_for_mode(mode, n_packets, seed)
    specs, meta = encode(
        mode,
        payload,
        seed=seed,
        rate_scale=rate_scale,
        pad_scale=pad_scale,
        jitter_scale=jitter_scale,
    )
    # The flush datagram is a protocol terminator, not a data symbol; keep only
    # data packets so the flow reflects what an observer would attribute to the
    # payload.
    specs = specs[: min(n_packets, int(meta["nsym"]))]
    rng = np.random.default_rng(seed + 4242)
    flow: Flow = []
    last_t = 0.0
    for spec in specs:
        t = spec.offset_s
        if capture_jitter:
            t = max(last_t, t + abs(rng.normal(0.0, CAPTURE_JITTER_MS / 1000.0)))
        flow.append((t, 0, int(spec.size)))
        last_t = t
    return flow


def channel_sweep(payload_bytes: int = 64, seed: int = 7) -> Dict[str, ChannelResult]:
    """Run the real channel once per mode with a fixed payload size."""
    out: Dict[str, ChannelResult] = {}
    for idx, mode in enumerate(MODES):
        payload = np.random.default_rng(seed + idx).integers(0, 256, payload_bytes, dtype=np.uint8).tobytes()
        out[mode] = run_channel(mode, payload=payload, seed=seed + idx)
    return out
