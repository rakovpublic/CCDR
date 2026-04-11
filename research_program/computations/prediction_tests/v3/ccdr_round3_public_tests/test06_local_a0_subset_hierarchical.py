#!/usr/bin/env python3
"""
Test 06: proper galaxy-level local-a0 subset test on SPARC.

Uses the public SPARC RAR and Table1 machine-readable tables. Fits a common a0
with per-galaxy shrinkage offsets, then reruns the fit on physically motivated
subsets (surface brightness, gas fraction, inclination) when those columns are
available. If the public RAR source exposes only compact 4-column data without
per-galaxy labels, the script falls back to a pooled fit and reports that the
subset stage could not be run honestly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common_public_data import fit_rar_hierarchical_like, load_sparc_rar, load_sparc_table1, pick_column, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test06_local_a0_subset_hierarchical"))
    parser.add_argument("--offset-prior-dex", type=float, default=0.08)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rar = load_sparc_rar().copy()
    notes = [
        "This is a lightweight galaxy-level shrinkage approximation to a hierarchical fit, using only public SPARC machine-readable tables.",
    ]

    try:
        galaxy_col = pick_column(rar.columns, ["Galaxy", "galaxy", "Name", "name", "gal"])
        has_galaxy_labels = True
    except Exception:
        galaxy_col = "__galaxy__"
        rar[galaxy_col] = "__all__"
        has_galaxy_labels = False
        notes.append(
            "Public SPARC RAR source did not expose per-galaxy labels; falling back to pooled fit with a synthetic single-galaxy label, so subset fits are skipped."
        )

    gobs_col = pick_column(rar.columns, ["gobs", "gobs2", "gobs_tot", "gobs_mond"])
    gbar_col = pick_column(rar.columns, ["gbar", "gbar2", "gbar_tot", "gnew"])

    base = fit_rar_hierarchical_like(rar, galaxy_col, gobs_col, gbar_col, offset_prior_dex=args.offset_prior_dex)
    base["galaxy_labels_available"] = bool(has_galaxy_labels)

    subsets = []
    if has_galaxy_labels:
        table1 = load_sparc_table1().copy()
        t1_name_col = pick_column(table1.columns, ["Name", "name", "Galaxy", "galaxy"])
        merged = rar.merge(table1, left_on=galaxy_col, right_on=t1_name_col, how="left", suffixes=("", "_t1"))

        for feature_candidates, label in [
            (["SBeff", "SB", "SB_eff"], "surface_brightness"),
            (["logMHI", "MHI", "M_HI"], "gas_mass"),
            (["Inc", "inclination"], "inclination"),
        ]:
            try:
                fcol = pick_column(merged.columns, feature_candidates)
            except Exception:
                continue
            vals = pd.to_numeric(merged[fcol], errors="coerce").to_numpy(float)
            if np.sum(np.isfinite(vals)) < 100:
                continue
            med = float(np.nanmedian(vals))
            for side, mask in [
                ("low", np.isfinite(vals) & (vals < med)),
                ("high", np.isfinite(vals) & (vals >= med)),
            ]:
                work = merged.loc[mask, [galaxy_col, gobs_col, gbar_col]].copy()
                if len(work) < 200:
                    continue
                res = fit_rar_hierarchical_like(work, galaxy_col, gobs_col, gbar_col, offset_prior_dex=args.offset_prior_dex)
                res.update({"subset": f"{label}_{side}", "threshold": med, "feature_column": fcol})
                subsets.append(res)

    a0_milgrom = 1.2e-10
    H0_si = 70.0 * 1000.0 / 3.085677581491367e22
    cH0 = 299792458.0 * H0_si
    base.update({
        "fractional_offset_from_milgrom": (base["best_a0_m_per_s2"] - a0_milgrom) / a0_milgrom,
        "fractional_offset_from_cH0": (base["best_a0_m_per_s2"] - cH0) / cH0,
    })

    summary = {
        "test_name": "Local-a0 subset hierarchical test",
        "base_fit": base,
        "subset_fits": subsets,
        "falsification_logic": {
            "confirm_like": "A0 remains near Milgrom and far from cH0 across physically distinct SPARC subsets under galaxy-level shrinkage fitting.",
            "falsify_like": "The preferred a0 drifts strongly across subsets or some subsets move back toward cH0.",
        },
        "notes": notes,
    }
    save_json(args.outdir / "test06_local_a0_subset_hierarchical_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
