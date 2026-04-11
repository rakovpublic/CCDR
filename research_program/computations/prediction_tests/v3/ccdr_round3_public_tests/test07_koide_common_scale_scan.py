#!/usr/bin/env python3
"""
Test 07: common-scale Koide scan.

Downloads the public PDG 2024 lepton and quark summary PDFs, parses central
masses, and performs a transparent one-loop running scan to ask whether a single
common scale window exists where lepton, up-type, and down-type Koide ratios are
simultaneously close to 2/3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common_public_data import koide_q, parse_pdg_lepton_masses, parse_pdg_quark_masses, running_mass_one_loop, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test07_koide_common_scale_scan"))
    parser.add_argument("--mu-min", type=float, default=2.0)
    parser.add_argument("--mu-max", type=float, default=2000.0)
    parser.add_argument("--n-mu", type=int, default=400)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    leptons = parse_pdg_lepton_masses()
    quarks = parse_pdg_quark_masses()

    q_lepton = koide_q([leptons["e_mev"], leptons["mu_mev"], leptons["tau_mev"]])
    mu_grid = np.logspace(np.log10(args.mu_min), np.log10(args.mu_max), args.n_mu)

    up = {
        "u": running_mass_one_loop(quarks["u_gev"], 2.0, mu_grid, nf=5) * 1e3,
        "c": running_mass_one_loop(quarks["c_gev"], 2.0, mu_grid, nf=5) * 1e3,
        "t": running_mass_one_loop(quarks["t_gev"], 173.0, mu_grid, nf=6) * 1e3,
    }
    down = {
        "d": running_mass_one_loop(quarks["d_gev"], 2.0, mu_grid, nf=5) * 1e3,
        "s": running_mass_one_loop(quarks["s_gev"], 2.0, mu_grid, nf=5) * 1e3,
        "b": running_mass_one_loop(quarks["b_gev"], 4.2, mu_grid, nf=5) * 1e3,
    }

    q_up = np.asarray([koide_q([up["u"][i], up["c"][i], up["t"][i]]) for i in range(len(mu_grid))])
    q_down = np.asarray([koide_q([down["d"][i], down["s"][i], down["b"][i]]) for i in range(len(mu_grid))])
    target = 2.0 / 3.0
    dist = np.sqrt((q_up - target) ** 2 + (q_down - target) ** 2 + (q_lepton - target) ** 2)
    best_idx = int(np.argmin(dist))

    summary = {
        "test_name": "Koide common-scale scan",
        "pdg_2024_lepton_q": float(q_lepton),
        "best_common_scale_gev": float(mu_grid[best_idx]),
        "best_up_q": float(q_up[best_idx]),
        "best_down_q": float(q_down[best_idx]),
        "best_joint_distance_to_2over3": float(dist[best_idx]),
        "scan_mu_gev": [float(x) for x in mu_grid[::20]],
        "scan_up_q_sample": [float(x) for x in q_up[::20]],
        "scan_down_q_sample": [float(x) for x in q_down[::20]],
        "falsification_logic": {
            "confirm_like": "A narrow common-scale window emerges where lepton, up-type, and down-type Koide ratios all approach 2/3 together.",
            "falsify_like": "No common window survives; any apparent agreement requires sector-dependent scales.",
        },
        "notes": [
            "This is a transparent one-loop public-data stress test, not a precision flavour-RG calculation.",
        ],
    }
    save_json(args.outdir / "test07_koide_common_scale_scan_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
