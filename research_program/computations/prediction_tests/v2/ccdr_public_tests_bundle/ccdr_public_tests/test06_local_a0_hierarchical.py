#!/usr/bin/env python3
"""
Test 06: Local-a0 hierarchical-style robustness test on public SPARC RAR data.

This script uses the public SPARC RAR table and fits the standard one-parameter
RAR relation for a0 while allowing per-galaxy offsets as nuisance terms with a
Gaussian shrinkage prior. That is not a full Bayesian hierarchical model, but it
captures the main robustness question with only public data and standard Python
packages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from _common_public_data import load_sparc_rar, rar_relation, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test06_local_a0_hierarchical"))
    parser.add_argument("--offset-prior-dex", type=float, default=0.08)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_sparc_rar().copy()
    # Deduplicate/normalize column names first because some parsers emit repeated labels.
    norm_cols = []
    seen = {}
    for c in df.columns:
        base = str(c).strip()
        key = base.lower()
        seen[key] = seen.get(key, 0) + 1
        norm_cols.append(base if seen[key] == 1 else f"{base}__dup{seen[key]}")
    df.columns = norm_cols
    cols = {str(c).lower(): c for c in df.columns}

    def _pick_col(predicates):
        matches = []
        for lc, c in cols.items():
            score = 0
            for pred, weight in predicates:
                if pred(lc):
                    score += weight
            if score > 0:
                matches.append((score, c))
        if not matches:
            return None
        matches.sort(key=lambda x: (-x[0], str(x[1])))
        return matches[0][1]

    gobs_col = _pick_col([
        (lambda lc: lc == 'gobs', 10),
        (lambda lc: 'gobs' in lc and 'err' not in lc and 'e_' not in lc and 'sig' not in lc and 'sd' not in lc, 6),
        (lambda lc: 'observed' in lc and 'acc' in lc, 4),
    ])
    gbar_col = _pick_col([
        (lambda lc: lc == 'gbar', 10),
        (lambda lc: 'gbar' in lc and 'err' not in lc and 'e_' not in lc and 'sig' not in lc and 'sd' not in lc, 6),
        (lambda lc: 'gmon' in lc or 'gnew' in lc, 5),
        (lambda lc: 'bary' in lc and 'acc' in lc, 4),
    ])
    gal_col = _pick_col([
        (lambda lc: lc in {'gal', 'name', 'galaxy'}, 10),
        (lambda lc: 'galaxy' in lc or lc.startswith('name'), 6),
    ])
    if gobs_col is None or gbar_col is None:
        raise RuntimeError(f"Could not identify SPARC acceleration columns from: {list(df.columns)}")
    if gal_col is None:
        gal_col = '__all__'
        df[gal_col] = 'SPARC'

    work = df[[gal_col, gobs_col, gbar_col]].copy()
    work.columns = ['galaxy', 'gobs_raw', 'gbar_raw']
    work['galaxy'] = work['galaxy'].astype(str)
    work['gobs_raw'] = pd.to_numeric(work['gobs_raw'], errors='coerce')
    work['gbar_raw'] = pd.to_numeric(work['gbar_raw'], errors='coerce')
    work = work.dropna(subset=['galaxy', 'gobs_raw', 'gbar_raw']).copy()
    # SPARC RAR tables often store log10(acceleration / m s^-2). Detect that and convert.
    gobs_vals = work['gobs_raw'].to_numpy(float)
    gbar_vals = work['gbar_raw'].to_numpy(float)
    finite = np.isfinite(gobs_vals) & np.isfinite(gbar_vals)
    if not np.any(finite):
        raise RuntimeError('SPARC table yielded no finite gobs/gbar rows after coercion')
    if np.nanmedian(gobs_vals[finite]) < -3.0 and np.nanmedian(gbar_vals[finite]) < -3.0:
        work['gobs'] = 10.0 ** work['gobs_raw']
        work['gbar'] = 10.0 ** work['gbar_raw']
        input_units = 'log10(m/s^2) converted to linear m/s^2'
    else:
        work['gobs'] = work['gobs_raw']
        work['gbar'] = work['gbar_raw']
        input_units = 'linear m/s^2'
    work = work[(work['gobs'] > 0) & (work['gbar'] > 0)].copy()
    if len(work) == 0:
        raise RuntimeError(f"SPARC table produced zero positive rows using gobs={gobs_col}, gbar={gbar_col}, galaxy={gal_col}")
    galaxies = sorted(work['galaxy'].astype(str).unique())
    g_index = {g: i for i, g in enumerate(galaxies)}
    gid = work['galaxy'].astype(str).map(g_index).to_numpy(int)
    gobs = work['gobs'].to_numpy(float)
    gbar = work['gbar'].to_numpy(float)
    log_gobs = np.log10(gobs)
    log_gbar = np.log10(gbar)
    prior_sigma = args.offset_prior_dex

    def objective(theta: np.ndarray) -> float:
        log10_a0 = theta[0]
        offsets = theta[1:]
        a0 = 10.0 ** log10_a0
        pred = np.log10(rar_relation(gbar, a0)) + offsets[gid]
        resid = log_gobs - pred
        penalty = np.sum((offsets / prior_sigma) ** 2)
        return float(np.sum(resid**2) + penalty)

    x0 = np.zeros(1 + len(galaxies), dtype=float)
    x0[0] = np.log10(1.2e-10)
    res = optimize.minimize(objective, x0=x0, method="L-BFGS-B")
    best_log10_a0 = float(res.x[0])
    best_a0 = float(10.0 ** best_log10_a0)
    offsets = res.x[1:]

    # Compare against Milgrom and cH0 reference values.
    a0_milgrom = 1.2e-10
    h0_km_s_mpc = 70.0
    H0_si = h0_km_s_mpc * 1000.0 / 3.085677581491367e22
    cH0 = 299792458.0 * H0_si

    summary = {
        "test_name": "Local-a0 hierarchical-style robustness fit",
        "n_points": int(len(work)),
        "n_galaxies": int(len(galaxies)),
        "best_a0_m_per_s2": best_a0,
        "fractional_offset_from_milgrom": (best_a0 - a0_milgrom) / a0_milgrom,
        "fractional_offset_from_cH0": (best_a0 - cH0) / cH0,
        "offset_prior_dex": prior_sigma,
        "offset_rms_dex": float(np.sqrt(np.mean(offsets**2))),
        "optimizer_success": bool(res.success),
        "optimizer_message": str(res.message),
        "falsification_logic": {
            "confirm_like": "Best-fit a0 remains close to Milgrom and far from cH0 under hierarchical-style shrinkage.",
            "falsify_like": "The preferred a0 shifts materially or broadens so much that the local retreat loses robustness.",
        },
        "notes": [
            "This is a shrinkage-based approximation to a hierarchical fit, designed to stay lightweight and fully public-data-driven.",
            f"Detected SPARC acceleration input units: {input_units}.",
            f"Selected columns: galaxy={gal_col}, gobs={gobs_col}, gbar={gbar_col}.",
        ],
    }
    save_json(args.outdir / "test06_local_a0_hierarchical_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
