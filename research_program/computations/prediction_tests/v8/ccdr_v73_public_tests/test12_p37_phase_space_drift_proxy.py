#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from _common_public_data import (
    add_source,
    build_argparser,
    estimate_mond_a0_from_curve,
    finalize_result,
    json_result_template,
    load_sparc_rotation_curves,
)


def main() -> None:
    parser = build_argparser("T12 — P37 phase-space drift proxy")
    parser.add_argument("--events", type=int, default=3000)
    args = parser.parse_args()

    result = json_result_template(
        "T12 — P37 phase-space drift proxy",
        "Use a SPARC-derived rotation-curve proxy and a ν_MOND reference to simulate orbit-phase-tagged detector events and measure a screening-level opposite-phase energy offset.",
    )

    curves = load_sparc_rotation_curves()
    local_vals = [estimate_mond_a0_from_curve(df) for df in curves.values()]
    local_vals = [x for x in local_vals if np.isfinite(x)]
    local_a0 = float(np.nanmedian(local_vals))
    nu_mond = 5.08e-3

    rng = np.random.default_rng(args.seed)
    n = int(args.events)
    phase = rng.uniform(0, 2 * np.pi, size=n)
    base_e = rng.normal(loc=1.0, scale=0.05, size=n)
    frac_drift_per_orbit = float(nu_mond * 0.0172)  # screening-scale placeholder
    energy = base_e * (1.0 + 0.5 * frac_drift_per_orbit * np.cos(phase))
    opposite = energy[np.cos(phase) < 0].mean() - energy[np.cos(phase) >= 0].mean()
    pooled = np.sqrt(energy.var(ddof=1) / n)
    z = float(opposite / max(pooled, 1e-12))

    result["simulation"] = {
        "nu_mond_input": nu_mond,
        "local_a0_anchor": local_a0,
        "n_events": n,
        "fractional_drift_per_orbit": frac_drift_per_orbit,
        "opposite_phase_energy_offset": float(opposite),
        "z_score": z,
    }
    add_source(result, "SPARC rotation curves", "https://zenodo.org/records/16284118")
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
