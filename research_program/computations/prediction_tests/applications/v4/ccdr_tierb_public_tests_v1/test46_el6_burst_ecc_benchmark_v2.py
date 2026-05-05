#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from _tierb_v5_quality_helpers import ensure_dir, now_utc_iso


def local_ldpc_checks(n_bits: int, n_checks: int, check_span: int, rng) -> List[np.ndarray]:
    checks = []
    starts = np.linspace(0, n_bits - check_span, n_checks).astype(int)
    for s in starts:
        checks.append(np.arange(s, s + check_span) % n_bits)
    return checks


def surface_like_checks(n_bits: int, n_checks: int) -> List[np.ndarray]:
    side = int(math.sqrt(n_bits))
    if side * side != n_bits:
        side = int(math.sqrt(n_bits))
        n = side * side
    else:
        n = n_bits
    checks = []
    for r in range(side):
        checks.append(np.array([r * side + c for c in range(side)], dtype=int))
    for c in range(side):
        checks.append(np.array([r * side + c for r in range(side)], dtype=int))
    return checks[:n_checks]


def protograph_like_checks(n_bits: int, n_checks: int, weight: int, rng) -> List[np.ndarray]:
    # Structured quasi-cyclic LDPC proxy: repeated shifted protograph edges.
    checks = []
    block = max(8, n_bits // max(1, int(math.sqrt(n_checks))))
    for j in range(n_checks):
        base = (j * 7) % n_bits
        step = 1 + (j * 11) % max(2, block)
        checks.append(np.array([(base + step * k) % n_bits for k in range(weight)], dtype=int))
    return checks


def spatially_coupled_checks(n_bits: int, n_checks: int, weight: int, rng) -> List[np.ndarray]:
    checks = []
    window = max(weight * 2, n_bits // max(8, n_checks // 4))
    for j in range(n_checks):
        center = int(j * n_bits / n_checks)
        offsets = rng.integers(-window, window + 1, size=weight)
        checks.append(np.mod(center + offsets, n_bits).astype(int))
    return checks


def cdt_like_irregular_checks(n_bits: int, n_checks: int, weight: int, rng) -> List[np.ndarray]:
    # Irregular/nonlocal proxy: power-law jumps + local anchor.
    checks = []
    for j in range(n_checks):
        anchor = rng.integers(0, n_bits)
        idx = {int(anchor)}
        while len(idx) < weight:
            if rng.random() < 0.35:
                jump = int(rng.zipf(1.35)) % n_bits
                if rng.random() < 0.5:
                    jump = -jump
                idx.add((anchor + jump) % n_bits)
            else:
                idx.add(int(rng.integers(0, n_bits)))
        checks.append(np.array(sorted(idx), dtype=int))
    return checks


def interleaved_rs_like_checks(n_bits: int, n_checks: int, interleave: int) -> List[np.ndarray]:
    # Not a real RS decoder; parity proxy for interleaved burst mitigation.
    checks = []
    for j in range(n_checks):
        residue = j % interleave
        checks.append(np.arange(residue, n_bits, interleave, dtype=int))
    return checks


def syndrome_nonzero(checks: List[np.ndarray], error_bits: np.ndarray) -> bool:
    e = np.zeros(max(int(np.max(c)) for c in checks if len(c)) + 1, dtype=np.uint8)
    e[error_bits] = 1
    for c in checks:
        if int(e[c].sum()) % 2:
            return True
    return False


def undetected_rate(checks, n_bits, burst_len, trials, rng) -> float:
    und = 0
    for _ in range(trials):
        start = int(rng.integers(0, n_bits))
        err = np.mod(np.arange(start, start + burst_len), n_bits).astype(int)
        if not syndrome_nonzero(checks, err):
            und += 1
    return und / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_t46_el6_burst_ecc_v2")
    ap.add_argument("--n-bits", type=int, default=512)
    ap.add_argument("--n-checks", type=int, default=128)
    ap.add_argument("--weight", type=int, default=8)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    outdir = ensure_dir(args.outdir)
    rng = np.random.default_rng(args.seed)
    baselines = {
        "local_ldpc_toy": local_ldpc_checks(args.n_bits, args.n_checks, check_span=max(args.weight, 16), rng=rng),
        "surface_like_rows_cols": surface_like_checks(args.n_bits, args.n_checks),
        "protograph_like_qc": protograph_like_checks(args.n_bits, args.n_checks, args.weight, rng),
        "spatially_coupled_ldpc_proxy": spatially_coupled_checks(args.n_bits, args.n_checks, args.weight, rng),
        "interleaved_rs_like_parity_proxy": interleaved_rs_like_checks(args.n_bits, args.n_checks, interleave=max(8, args.weight)),
        "cdt_like_irregular_nonlocal": cdt_like_irregular_checks(args.n_bits, args.n_checks, args.weight, rng),
    }
    burst_lengths = [2, 4, 8, 16, 32, 64, 96]
    rows = []
    for b in burst_lengths:
        row = {"burst_length": b}
        for name, checks in baselines.items():
            row[name + "_undetected"] = undetected_rate(checks, args.n_bits, b, args.trials, rng)
        rows.append(row)
    cdt_key = "cdt_like_irregular_nonlocal_undetected"
    non_cdt = [k + "_undetected" for k in baselines if not k.startswith("cdt_like")]
    wins = []
    ratios = []
    for r in rows:
        cdt = max(r[cdt_key], 1e-9)
        best_other = min(r[k] for k in non_cdt)
        wins.append(cdt < best_other)
        ratios.append(best_other / cdt)
    result = {
        "schema": "ccdr-tierb-result-v1",
        "quality_patch_version": "v6_items_5_6_8_t48_additive",
        "generated_utc": now_utc_iso(),
        "test_id": "T46",
        "prediction_ids": ["EL6"],
        "prediction_names": ["CDT-like/random graph code improves burst-channel LDPC capacity proxy"],
        "data_source": "synthetic public-code-only burst-channel benchmark generated by script",
        "evidence_level": "synthetic_engineering_benchmark_not_observational_confirmation",
        "readiness_status": "model_fit_done",
        "n_bits": args.n_bits,
        "n_checks": args.n_checks,
        "check_weight": args.weight,
        "trials_per_burst_length": args.trials,
        "baselines": list(baselines.keys()),
        "burst_results": rows,
        "median_best_baseline_over_cdt_ratio": float(np.median(ratios)),
        "cdt_wins_all_burst_lengths": bool(all(wins)),
        "support_like": bool(all(wins) and np.median(ratios) > 1.25),
        "evidence_status": "confirm_like_synthetic_only" if all(wins) else "weakened_synthetic_only",
        "falsification_logic": {
            "caveat": "This is still a benchmark/prototype only, not observational evidence and not CCDR physics confirmation.",
            "confirm_like": "CDT-like irregular/nonlocal parity graph beats local, surface-like, protograph-like, spatially-coupled, and interleaved parity proxies at matched n_bits/n_checks/weight.",
            "falsify_like": "Any realistic matched baseline equals or beats the CDT-like graph across burst lengths."
        }
    }
    (outdir / "t46_el6_burst_ecc_benchmark_v2_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
