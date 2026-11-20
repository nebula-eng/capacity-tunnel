"""Unit tests for the detector, ROC utilities and cross-validation."""

from __future__ import annotations

import unittest

import numpy as np

from capacity_tunnel import detector


def separable_data(n_per_class=120, seed=0):
    rng = np.random.default_rng(seed)
    tunnel = rng.normal(loc=[2.5, 2.0, -1.5], scale=0.6, size=(n_per_class, 3))
    benign = rng.normal(loc=[-2.0, -1.5, 1.5], scale=0.6, size=(n_per_class, 3))
    X = np.vstack([tunnel, benign])
    y = np.concatenate([np.ones(n_per_class, dtype=int), np.zeros(n_per_class, dtype=int)])
    labels = ["tunnel"] * n_per_class + ["benign"] * n_per_class
    return X, y, labels


class TestROC(unittest.TestCase):
    def test_perfect_separation(self) -> None:
        y = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertEqual(detector.roc_auc(y, scores), 1.0)
        self.assertEqual(detector.fpr_at_tpr(y, scores, 0.95), 0.0)

    def test_inverted_scores_have_zero_auc(self) -> None:
        y = np.array([1, 1, 0, 0])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(detector.roc_auc(y, scores), 0.0)

    def test_ties_give_half_auc(self) -> None:
        y = np.array([1, 0, 1, 0])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(detector.roc_auc(y, scores), 0.5, places=9)

    def test_random_scores_are_near_chance(self) -> None:
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 2000)
        scores = rng.normal(size=2000)
        self.assertLess(abs(detector.roc_auc(y, scores) - 0.5), 0.05)

    def test_fpr_at_tpr_operating_point(self) -> None:
        y = np.array([1] * 100 + [0] * 100)
        scores = np.concatenate([np.linspace(0.5, 1.0, 100), np.linspace(0.0, 0.6, 100)])
        value = detector.fpr_at_tpr(y, scores, 0.95)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertGreater(value, 0.0)


class TestCrossValidation(unittest.TestCase):
    def test_folds_partition_the_data(self) -> None:
        _, y, labels = separable_data(n_per_class=50)
        folds = detector.stratified_folds_by_class(labels, k=5, seed=1)
        self.assertEqual(len(folds), 5)
        union = np.concatenate(folds)
        self.assertEqual(sorted(union.tolist()), list(range(y.size)))
        self.assertEqual(len(set(union.tolist())), y.size)
        for fold in folds:
            self.assertEqual(int(y[fold].sum() * 2), fold.size)

    def test_cv_on_separable_data(self) -> None:
        X, y, labels = separable_data(n_per_class=80)
        result = detector.cross_validate(X, y, labels, k=5, seed=2)
        self.assertGreaterEqual(result.auc, 0.99)
        self.assertEqual(len(result.fold_aucs), 5)
        self.assertTrue(all(a >= 0.98 for a in result.fold_aucs))
        self.assertLess(result.fpr_at_95tpr, 0.1)


class TestModel(unittest.TestCase):
    def test_gradient_descent_fits_separable_data(self) -> None:
        X, y, _ = separable_data(n_per_class=60)
        model = detector.fit_logreg(X, y, iters=800)
        pred = (model.decision_function(X) > 0).astype(int)
        self.assertGreaterEqual(float((pred == y).mean()), 0.98)

    def test_standardization_is_invariant_to_affine_scaling(self) -> None:
        X, y, _ = separable_data(n_per_class=60)
        model_a = detector.fit_logreg(X, y, iters=600)
        shifted = X * 10.0 + 500.0
        model_b = detector.fit_logreg(shifted, y, iters=600)
        scores_a = model_a.decision_function(X)
        scores_b = model_b.decision_function(shifted)
        self.assertAlmostEqual(detector.roc_auc(y, scores_a), detector.roc_auc(y, scores_b), places=6)

    def test_model_serialization_round_trip(self) -> None:
        X, y, _ = separable_data(n_per_class=40)
        model = detector.fit_logreg(X, y, iters=400)
        copied = detector.LogReg.from_dict(model.to_dict())
        np.testing.assert_allclose(model.predict_proba(X), copied.predict_proba(X))

    def test_threshold_is_monotone(self) -> None:
        X, y, _ = separable_data(n_per_class=50)
        model = detector.fit_logreg(X, y, iters=400)
        scores = model.decision_function(X)
        # A higher TPR target requires a lower decision threshold.
        high = detector.threshold_at_tpr(y, scores, 0.95)
        low = detector.threshold_at_tpr(y, scores, 0.50)
        self.assertLessEqual(high, low)


if __name__ == "__main__":
    unittest.main()
