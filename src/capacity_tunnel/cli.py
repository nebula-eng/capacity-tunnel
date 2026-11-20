"""Command line interface for capacity-tunnel.

Subcommands
-----------
``demo``    run the real loopback channel in all three modes; print throughput/BER
``gen``     write a synthetic detection dataset (npz + json summary)
``train``   cross-validate, fit on all data, write model.json and metrics.json
``eval``    full pipeline: dataset -> CV -> baselines -> real channel -> robustness
``detect``  score a single flow from a JSON file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from . import channel, detector, features

DEFAULT_OUT = "results"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(payload: Dict[str, Any], path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")


def _save_dataset(ds: detector.Dataset, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez_compressed(
        path,
        X=ds.X,
        y=ds.y,
        class_labels=np.array(ds.class_labels),
        feature_names=np.array(features.FEATURE_NAMES),
        window=np.array([ds.window]),
        seed=np.array([ds.seed]),
    )


def _load_dataset(path: str) -> detector.Dataset:
    with np.load(path, allow_pickle=False) as data:
        return detector.Dataset(
            X=np.array(data["X"], dtype=float),
            y=np.array(data["y"], dtype=int),
            class_labels=[str(v) for v in data["class_labels"]],
            flows_per_class=int(sum(1 for v in data["class_labels"] if v == data["class_labels"][0])),
            window=int(data["window"][0]),
            seed=int(data["seed"][0]),
        )


def cmd_demo(args: argparse.Namespace) -> int:
    print("capacity-tunnel: real UDP loopback channel")
    print("payload: %d bytes per mode" % args.payload)
    print("%-10s %8s %6s %10s %9s %8s %8s" % ("mode", "packets", "recv", "bps", "wall_s", "BER", "sha256"))
    payload = np.random.default_rng(args.seed).integers(0, 256, args.payload, dtype=np.uint8).tobytes()
    for mode in channel.MODES:
        result = channel.run_channel(mode, payload=payload, seed=args.seed)
        print(
            "%-10s %8d %6d %10.1f %9.3f %8.4f %8s"
            % (
                mode,
                result.packets_sent,
                result.packets_received,
                result.throughput_bps,
                result.wall_duration_s,
                result.ber,
                "ok" if result.sha256_match else "FAIL",
            )
        )
    print("bps is protocol-normalized (payload bits / scheduled gaps); wall_s includes scheduling.")
    return 0


def cmd_gen(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    ds = detector.build_dataset(per_class=args.per_class, window=args.window, seed=args.seed)
    out = os.path.join(args.out, "dataset.npz")
    _save_dataset(ds, out)
    counts: Dict[str, int] = {}
    for label in ds.class_labels:
        counts[label] = counts.get(label, 0) + 1
    summary = {
        "path": out,
        "n_total": int(ds.X.shape[0]),
        "n_features": int(ds.X.shape[1]),
        "window": ds.window,
        "seed": ds.seed,
        "per_class": counts,
    }
    _write_json(summary, os.path.join(args.out, "dataset.json"))
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    print("elapsed: %.2f s" % (time.perf_counter() - started))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    ds = _load_dataset(args.data)
    cv = detector.cross_validate(ds.X, ds.y, ds.class_labels, k=args.folds, seed=args.seed)
    model = detector.fit_logreg(ds.X, ds.y, l2=args.l2)
    detector.save_model(model, os.path.join(args.out, "model.json"))
    metrics = {
        "auc": cv.auc,
        "fpr_at_95tpr": cv.fpr_at_95tpr,
        "threshold": cv.threshold,
        "fold_aucs": cv.fold_aucs,
        "n_total": int(ds.X.shape[0]),
        "n_features": int(ds.X.shape[1]),
    }
    _write_json(metrics, os.path.join(args.out, "metrics.json"))
    print("AUC %.4f  FPR@95TPR %.4f  folds %s" % (cv.auc, cv.fpr_at_95tpr, [round(a, 4) for a in cv.fold_aucs]))
    return 0


def _class_counts(labels: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def cmd_eval(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    out = args.out
    os.makedirs(out, exist_ok=True)

    ds = detector.build_dataset(per_class=args.per_class, window=args.window, seed=args.seed)
    labels = np.array(ds.class_labels)
    print("dataset: %d flows, %d features" % ds.X.shape)
    t_data = time.perf_counter() - started

    t0 = time.perf_counter()
    cv = detector.cross_validate(ds.X, ds.y, ds.class_labels, k=args.folds, seed=args.seed, l2=args.l2)
    t_cv = time.perf_counter() - t0

    model = detector.fit_logreg(ds.X, ds.y, l2=args.l2)
    detector.save_model(model, os.path.join(out, "model.json"))

    per_mode: Dict[str, Dict[str, float]] = {}
    for mode in detector.TUNNEL_CLASSES:
        mask = (labels == mode) | (ds.y == 0)
        per_mode[mode] = {
            "auc": detector.roc_auc(ds.y[mask], cv.oof_scores[mask]),
            "fpr_at_95tpr": detector.fpr_at_tpr(ds.y[mask], cv.oof_scores[mask], 0.95),
            "tpr_at_5fpr": detector.tpr_at_fpr(ds.y[mask], cv.oof_scores[mask], 0.05),
            "n_tunnel": int((labels == mode).sum()),
            "n_benign": int((ds.y == 0).sum()),
        }

    baselines: Dict[str, Dict[str, float]] = {}
    entropy_oof: Optional[np.ndarray] = None
    for name, names in detector.FEATURE_SETS.items():
        idx = detector.subset_indices(names)
        result = detector.cross_validate(
            ds.X[:, idx], ds.y, ds.class_labels, k=args.folds, seed=args.seed, l2=args.l2
        )
        if name == "entropy_only":
            entropy_oof = result.oof_scores
        baselines[name] = {
            "auc": result.auc,
            "fpr_at_95tpr": result.fpr_at_95tpr,
            "n_features": len(names),
            "features": list(names),
        }

    single_feature = {
        name: detector.roc_auc(ds.y, ds.X[:, j]) for j, name in enumerate(features.FEATURE_NAMES)
    }
    capacity_idx = features.FEATURE_NAMES.index("log_capacity_bps")
    ratio_idx = features.FEATURE_NAMES.index("log_capacity_ratio")
    entropy_idx = features.FEATURE_NAMES.index("size_entropy")
    rate_idx = features.FEATURE_NAMES.index("size_entropy_rate")
    alphabet_idx = features.FEATURE_NAMES.index("size_alphabet")
    per_class_medians: Dict[str, Dict[str, float]] = {}
    for cls in detector.ALL_CLASSES:
        mask = labels == cls
        per_class_medians[cls] = {
            "size_entropy": float(np.median(ds.X[mask, entropy_idx])),
            "size_entropy_rate": float(np.median(ds.X[mask, rate_idx])),
            "size_alphabet": float(np.median(ds.X[mask, alphabet_idx])),
            "capacity_bps": float(np.median(np.power(10.0, ds.X[mask, capacity_idx]) - 1.0)),
            "capacity_ratio": float(np.median(ds.X[mask, ratio_idx])),
        }

    t0 = time.perf_counter()
    channel_results = channel.channel_sweep(payload_bytes=args.channel_payload, seed=args.seed)
    t_channel = time.perf_counter() - t0
    channel_block: Dict[str, Dict[str, Any]] = {}
    for mode, result in channel_results.items():
        channel_block[mode] = {
            "payload_bytes": result.payload_bytes,
            "packets_sent": result.packets_sent,
            "packets_received": result.packets_received,
            "throughput_bps": round(result.throughput_bps, 6),
            "ber": result.ber,
            "sha256_match": result.sha256_match,
            "nominal_duration_s": round(result.nominal_duration_s, 6),
        }
        print(
            "channel %-9s %8.1f bps  BER %.4f  sha256 %s"
            % (mode, result.throughput_bps, result.ber, result.sha256_match)
        )

    robustness_block: Dict[str, Any] = {}
    t_rob = 0.0
    if args.robustness:
        t0 = time.perf_counter()
        train_idx, test_idx = detector.stratified_holdout(ds.class_labels, seed=args.seed)
        held_model = detector.fit_logreg(ds.X[train_idx], ds.y[train_idx], l2=args.l2)
        entropy_cols = detector.subset_indices(detector.FEATURE_SETS["entropy_only"])
        entropy_model = detector.fit_logreg(ds.X[train_idx][:, entropy_cols], ds.y[train_idx], l2=args.l2)
        benign_test = test_idx[ds.y[test_idx] == 0]
        benign_X = ds.X[benign_test]
        benign_y = ds.y[benign_test]
        padding_rows = detector.evasion_sweep(
            held_model,
            benign_X,
            benign_y,
            varying="padding",
            scales=(1.0, 1.5, 2.0, 4.0, 8.0, 16.0),
            per_scale=args.robust_per_scale,
            window=args.window,
            seed=args.seed + 505,
            entropy_model=entropy_model,
            entropy_indices=entropy_cols,
        )
        jitter_rows = detector.evasion_sweep(
            held_model,
            benign_X,
            benign_y,
            varying="timing_jitter",
            scales=(1.0, 2.0, 4.0, 8.0, 16.0),
            per_scale=args.robust_per_scale,
            window=args.window,
            seed=args.seed + 909,
            entropy_model=entropy_model,
            entropy_indices=entropy_cols,
        )
        robustness_block = {
            "protocol": "model trained on 70% stratified split; benign reference is the 30% hold-out; detector frozen",
            "n_benign_holdout": int(benign_test.size),
            "padding_sweep": padding_rows,
            "timing_jitter_sweep": jitter_rows,
        }
        t_rob = time.perf_counter() - t0

    elapsed = time.perf_counter() - started
    results = {
        "config": {
            "seed": args.seed,
            "per_class": ds.flows_per_class,
            "window": ds.window,
            "cv_folds": args.folds,
            "l2": args.l2,
            "channel_payload_bytes": args.channel_payload,
        },
        "dataset": {
            "n_total": int(ds.X.shape[0]),
            "n_tunnel": int(ds.y.sum()),
            "n_benign": int((ds.y == 0).sum()),
            "per_class": _class_counts(ds.class_labels),
        },
        "detector": {
            "auc": cv.auc,
            "fpr_at_95tpr": cv.fpr_at_95tpr,
            "tpr_at_5fpr": detector.tpr_at_fpr(ds.y, cv.oof_scores, 0.05),
            "threshold_at_95tpr": cv.threshold,
            "fold_aucs": cv.fold_aucs,
            "fold_fpr_at_95tpr": cv.fold_fpr,
            "n_features": len(features.FEATURE_NAMES),
            "features": list(features.FEATURE_NAMES),
        },
        "per_mode": per_mode,
        "baselines": baselines,
        "channel": channel_block,
        "robustness": robustness_block,
        "descriptive": {
            "single_feature_auc": single_feature,
            "per_class_medians": per_class_medians,
        },
    }
    _write_json(results, os.path.join(out, "results.json"))

    np.savez_compressed(
        os.path.join(out, "flow_scores.npz"),
        X=ds.X,
        y=ds.y,
        class_labels=np.array(ds.class_labels),
        oof_scores=cv.oof_scores,
        feature_names=np.array(features.FEATURE_NAMES),
        per_mode=np.array(list(detector.TUNNEL_CLASSES)),
        oof_scores_entropy_only=entropy_oof if entropy_oof is not None else np.zeros(ds.y.size),
    )
    _save_dataset(ds, os.path.join(out, "dataset.npz"))
    _write_json(
        {
            "n_total": int(ds.X.shape[0]),
            "n_features": int(ds.X.shape[1]),
            "window": ds.window,
            "seed": ds.seed,
            "per_class": _class_counts(ds.class_labels),
        },
        os.path.join(out, "dataset.json"),
    )

    print("overall AUC %.4f  FPR@95TPR %.4f" % (cv.auc, cv.fpr_at_95tpr))
    for mode, block in per_mode.items():
        print("  %-9s AUC %.4f  FPR@95TPR %.4f" % (mode, block["auc"], block["fpr_at_95tpr"]))
    print("baselines: " + "  ".join("%s %.4f" % (k, v["auc"]) for k, v in baselines.items()))
    if robustness_block:
        print(
            "padding:   "
            + "  ".join("%.4gx AUC %.4f" % (p["scale"], p["auc"]) for p in robustness_block["padding_sweep"])
        )
        print(
            "jitter:    "
            + "  ".join(
                "%.4gx AUC %.4f cap %.0f" % (p["scale"], p["auc"], p["median_capacity_bps"])
                for p in robustness_block["timing_jitter_sweep"]
            )
        )
    print(
        "timings: dataset %.2f s, cv %.2f s, channel %.2f s, robustness %.2f s, total %.2f s"
        % (t_data, t_cv, t_channel, t_rob, elapsed)
    )
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    with open(args.flow, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    records = payload.get("records")
    if not isinstance(records, list):
        raise SystemExit("flow JSON must contain a 'records' list of [t, direction, size]")
    flow = [(float(r[0]), int(r[1]), int(r[2])) for r in records]
    named = features.extract_named(flow, window=args.window)
    vector = np.array([named[name] for name in features.FEATURE_NAMES], dtype=float)[None, :]

    model_path = args.model
    if not os.path.exists(model_path):
        raise SystemExit("model not found at %s; run 'make train' or 'make eval' first" % model_path)
    model = detector.load_model(model_path)
    score = float(model.decision_function(vector)[0])
    probability = float(model.predict_proba(vector)[0])
    verdict = "tunnel" if probability >= args.threshold else "benign"

    output = {
        "flow_records": len(flow),
        "window": args.window,
        "score": round(score, 6),
        "probability": round(probability, 6),
        "threshold": args.threshold,
        "verdict": verdict,
        "features": {k: round(float(v), 6) for k, v in named.items()},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capacity-tunnel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the real loopback channel")
    demo.add_argument("--payload", type=int, default=96, help="payload bytes per mode (default 96)")
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(func=cmd_demo)

    gen = sub.add_parser("gen", help="write a synthetic dataset")
    gen.add_argument("--out", default=DEFAULT_OUT)
    gen.add_argument("--per-class", type=int, default=80)
    gen.add_argument("--window", type=int, default=512)
    gen.add_argument("--seed", type=int, default=2024)
    gen.set_defaults(func=cmd_gen)

    train = sub.add_parser("train", help="train and save the detector")
    train.add_argument("--data", default=os.path.join(DEFAULT_OUT, "dataset.npz"))
    train.add_argument("--out", default=DEFAULT_OUT)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--l2", type=float, default=1e-2)
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="full evaluation pipeline")
    ev.add_argument("--out", default=DEFAULT_OUT)
    ev.add_argument("--per-class", type=int, default=80)
    ev.add_argument("--window", type=int, default=512)
    ev.add_argument("--folds", type=int, default=5)
    ev.add_argument("--l2", type=float, default=1e-2)
    ev.add_argument("--seed", type=int, default=2024)
    ev.add_argument("--channel-payload", type=int, default=64)
    ev.add_argument("--robust-per-scale", type=int, default=40)
    ev.add_argument("--robustness", dest="robustness", action="store_true", default=True)
    ev.add_argument("--no-robustness", dest="robustness", action="store_false")
    ev.set_defaults(func=cmd_eval)

    de = sub.add_parser("detect", help="score one flow JSON")
    de.add_argument("--flow", required=True)
    de.add_argument("--model", default=os.path.join(DEFAULT_OUT, "model.json"))
    de.add_argument("--window", type=int, default=512)
    de.add_argument("--threshold", type=float, default=0.5)
    de.set_defaults(func=cmd_detect)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
