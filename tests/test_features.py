"""Unit tests for the per-flow feature extractor."""

from __future__ import annotations

import unittest

import numpy as np

from capacity_tunnel import capacity, features


def make_flow(sizes, gaps, direction=0):
    """Build a flow from sizes and inter-packet gaps (len(gaps) == len(sizes) - 1)."""
    t = 0.0
    records = []
    for i, size in enumerate(sizes):
        records.append((t, direction, int(size)))
        if i < len(gaps):
            t += float(gaps[i])
    return records


class TestEntropyFeatures(unittest.TestCase):
    def test_uniform_alphabet_entropy_is_two_bits(self) -> None:
        rng = np.random.default_rng(0)
        sizes = rng.choice([100, 300, 500, 700], size=400)
        flow = make_flow(sizes, [0.01] * (len(sizes) - 1))
        named = features.extract_named(flow)
        self.assertAlmostEqual(named["size_entropy"], 2.0, delta=0.05)
        self.assertEqual(named["size_alphabet"], 4.0)

    def test_constant_size_has_zero_entropy(self) -> None:
        flow = make_flow([120] * 200, [0.005] * 199)
        named = features.extract_named(flow)
        self.assertEqual(named["size_entropy"], 0.0)
        self.assertEqual(named["size_entropy_rate"], 0.0)
        self.assertEqual(named["size_mi"], 0.0)
        self.assertEqual(named["size_alphabet"], 1.0)

    def test_alternating_symbols_have_mutual_information(self) -> None:
        sizes = [100 if i % 2 == 0 else 300 for i in range(300)]
        flow = make_flow(sizes, [0.01] * 299)
        named = features.extract_named(flow)
        self.assertGreater(named["size_mi"], 0.8)

    def test_iid_symbols_have_near_zero_mutual_information(self) -> None:
        rng = np.random.default_rng(3)
        sizes = rng.choice([100, 300, 500, 700], size=400)
        flow = make_flow(sizes, [0.01] * 399)
        named = features.extract_named(flow)
        self.assertLess(named["size_mi"], 0.2)


class TestTimingFeatures(unittest.TestCase):
    def test_iat_cv_matches_alternating_gaps(self) -> None:
        gaps = [0.01 if i % 2 == 0 else 0.03 for i in range(299)]
        flow = make_flow([200] * 300, gaps)
        named = features.extract_named(flow)
        self.assertAlmostEqual(named["iat_cv"], 0.5, delta=0.02)

    def test_periodic_flow_scores_high_periodicity(self) -> None:
        flow = make_flow([100] * 300, [0.005] * 299)
        named = features.extract_named(flow)
        self.assertGreater(named["periodicity"], 0.5)
        self.assertLess(named["fano_factor"], 0.5)

    def test_bimodal_timing_is_detected(self) -> None:
        gaps = [0.005 if i % 2 == 0 else 0.025 for i in range(299)]
        flow = make_flow([200] * 300, gaps)
        named = features.extract_named(flow)
        self.assertGreaterEqual(named["timing_capacity"], 0.9)
        self.assertGreater(named["timing_separation"], 0.5)


class TestVectorContract(unittest.TestCase):
    def test_vector_length_and_order(self) -> None:
        flow = make_flow([100, 200, 300, 400] * 50, [0.01] * 199)
        vector = features.extract(flow)
        self.assertEqual(vector.shape, (len(features.FEATURE_NAMES),))
        named = features.extract_named(flow)
        for i, name in enumerate(features.FEATURE_NAMES):
            self.assertEqual(vector[i], named[name])

    def test_direction_fraction(self) -> None:
        ups = make_flow([100] * 50, [0.01] * 49, direction=0)
        both = ups + [(t + 1.0, 1, 200) for (t, _, _) in ups]
        self.assertEqual(features.extract_named(ups)["up_fraction"], 1.0)
        self.assertAlmostEqual(features.extract_named(sorted(both))["up_fraction"], 0.5, places=6)

    def test_short_flow_is_all_zero(self) -> None:
        named = features.extract_named(make_flow([100, 200], [0.01]))
        self.assertTrue(all(v == 0.0 for v in named.values()))

    def test_capacity_of_constant_channel_is_zero(self) -> None:
        flow = make_flow([64] * 200, [0.004] * 199)
        est = capacity.estimate_flow(flow)
        self.assertEqual(est.capacity_bps, 0.0)
        self.assertEqual(est.h_rate, 0.0)
        self.assertLess(est.timing_separation, capacity.MIN_TIMING_SEPARATION)
        self.assertEqual(est.c_timing, 0.0)


if __name__ == "__main__":
    unittest.main()
