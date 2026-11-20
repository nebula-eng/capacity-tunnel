"""Unit tests for the real loopback covert channel."""

from __future__ import annotations

import unittest

from capacity_tunnel import channel


class TestChannelRoundTrip(unittest.TestCase):
    def test_b32_roundtrip(self) -> None:
        result = channel.run_channel("b32", payload=bytes(range(8)), seed=11)
        self.assertEqual(result.ber, 0.0)
        self.assertTrue(result.sha256_match)
        self.assertEqual(result.packets_received, result.packets_sent)
        self.assertGreater(result.throughput_bps, 0.0)

    def test_timing_roundtrip(self) -> None:
        result = channel.run_channel("timing", payload=bytes(range(8)), seed=12)
        self.assertEqual(result.ber, 0.0)
        self.assertTrue(result.sha256_match)
        self.assertEqual(result.packets_received, result.packets_sent - 1)  # flush is not data

    def test_hardened_roundtrip(self) -> None:
        result = channel.run_channel("hardened", payload=bytes(range(8)), seed=13)
        self.assertEqual(result.ber, 0.0)
        self.assertTrue(result.sha256_match)
        self.assertGreater(result.packets_received, 0)


class TestEmissionPlan(unittest.TestCase):
    def test_plan_is_deterministic(self) -> None:
        payload = bytes(range(24))
        first, meta_a = channel.encode("hardened", payload, seed=5)
        second, meta_b = channel.encode("hardened", payload, seed=5)
        self.assertEqual([p.size for p in first], [p.size for p in second])
        self.assertEqual([p.offset_s for p in first], [p.offset_s for p in second])
        self.assertEqual(meta_a, meta_b)

    def test_b32_sizes_are_constant(self) -> None:
        specs, _ = channel.encode("b32", bytes(range(16)), seed=1)
        self.assertEqual({s.size for s in specs}, {channel.B32_PACKET_SIZE})
        self.assertEqual(len(specs), 26)  # 128 bits / 5 bits per symbol, rounded up

    def test_timing_gaps_are_bimodal(self) -> None:
        specs, _ = channel.encode("timing", bytes(range(16)), seed=1)
        gaps = [s.gap_s * 1000.0 for s in specs[:-1]]
        short = [g for g in gaps if g < 15.0]
        long_ = [g for g in gaps if g >= 15.0]
        self.assertGreater(len(short), 0)
        self.assertGreater(len(long_), 0)
        self.assertLess(max(short), 10.0)
        self.assertGreater(min(long_), 20.0)

    def test_hardened_uses_six_symbols(self) -> None:
        payload = bytes([0xFF]) + bytes(range(63))
        specs, meta = channel.encode("hardened", payload, seed=2)
        classes = {channel._size_class(s.size) for s in specs[:-1]}
        self.assertEqual(classes, {0, 1, 2})
        self.assertGreater(meta["nsym"], 100)
        # 6-ary symbol: log2(6) = 2.585 bits per packet.
        bits_per_packet = meta["nbits"] / meta["nsym"]
        self.assertAlmostEqual(bits_per_packet, 2.585, delta=0.06)
        for spec in specs[:-1]:
            self.assertIn(channel._size_class(spec.size), (0, 1, 2))

    def test_emit_flow_shape(self) -> None:
        flow = channel.emit_flow("timing", seed=4, n_packets=64)
        self.assertEqual(len(flow), 64)
        self.assertTrue(all(direction == 0 for _, direction, _ in flow))
        times = [t for t, _, _ in flow]
        self.assertTrue(all(b >= a for a, b in zip(times, times[1:])))


if __name__ == "__main__":
    unittest.main()
