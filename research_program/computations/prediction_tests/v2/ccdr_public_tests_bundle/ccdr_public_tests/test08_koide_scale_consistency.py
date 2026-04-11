#!/usr/bin/env python3
"""
Test 08: Koide-scale consistency test using public PDG 2024 summary tables.

This script downloads the public PDG lepton and quark mass summary tables, parses
central values, computes Koide Q, and then scans a simple one-loop QCD running
model to ask whether there is a distinct scale window where the quark-sector Q
becomes unusually close to 2/3.

This is intentionally a structural consistency test, not a claim that one-loop
running is sufficient for precision flavour phenomenology.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common_public_data import (
    koide_q,
    parse_pdg_lepton_masses,
    parse_pdg_quark_masses,
    running_mass_one_loop,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out_test08_koide_scale_consistency"))
    parser.add_argument("--mu-min", type=float, default=2.0)
    parser.add_argument("--mu-max", type=float, default=2000.0)
    parser.add_argument("--n-mu", type=int, default=400)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    try:
        leptons = parse_pdg_lepton_masses()
    except Exception:
        leptons = {"electron_MeV": 0.51099895, "muon_MeV": 105.6583755, "tau_MeV": 1776.86}
    try:
        quarks = parse_pdg_quark_masses()
    except Exception:
        quarks = {
            "up_GeV": 0.00216,
            "down_GeV": 0.00467,
            "strange_GeV": 0.0934,
            "charm_GeV": 1.27,
            "bottom_GeV": 4.18,
            "top_GeV": 172.57,
        }

    lepton_q = koide_q([leptons["electron_MeV"], leptons["muon_MeV"], leptons["tau_MeV"]])

    mu_grid = np.logspace(np.log10(args.mu_min), np.log10(args.mu_max), args.n_mu)
    up_q = []
    down_q = []
    for mu in mu_grid:
        up = running_mass_one_loop(quarks["up_GeV"], mu0_gev=2.0, mu_gev=mu, nf=3)
        charm = running_mass_one_loop(quarks["charm_GeV"], mu0_gev=quarks["charm_GeV"], mu_gev=mu, nf=4)
        top = running_mass_one_loop(quarks["top_GeV"], mu0_gev=quarks["top_GeV"], mu_gev=mu, nf=6)
        down = running_mass_one_loop(quarks["down_GeV"], mu0_gev=2.0, mu_gev=mu, nf=3)
        strange = running_mass_one_loop(quarks["strange_GeV"], mu0_gev=2.0, mu_gev=mu, nf=3)
        bottom = running_mass_one_loop(quarks["bottom_GeV"], mu0_gev=quarks["bottom_GeV"], mu_gev=mu, nf=5)
        up_q.append(koide_q([up * 1e3, charm * 1e3, top * 1e3]))
        down_q.append(koide_q([down * 1e3, strange * 1e3, bottom * 1e3]))

    up_q = np.asarray(up_q)
    down_q = np.asarray(down_q)
    target = 2.0 / 3.0
    i_up = int(np.argmin(np.abs(up_q - target)))
    i_down = int(np.argmin(np.abs(down_q - target)))

    summary = {
        "test_name": "Koide-scale consistency",
        "pdg_2024_lepton_q": float(lepton_q),
        "best_up_scale_gev": float(mu_grid[i_up]),
        "best_up_q": float(up_q[i_up]),
        "best_up_abs_delta_from_2over3": float(abs(up_q[i_up] - target)),
        "best_down_scale_gev": float(mu_grid[i_down]),
        "best_down_q": float(down_q[i_down]),
        "best_down_abs_delta_from_2over3": float(abs(down_q[i_down] - target)),
        "scan_mu_gev": [float(x) for x in mu_grid[::20]],
        "scan_up_q_sample": [float(x) for x in up_q[::20]],
        "scan_down_q_sample": [float(x) for x in down_q[::20]],
        "falsification_logic": {
            "confirm_like": "A physically narrow scale window emerges where the relevant sector is unusually close to 2/3 and that window is stable under reasonable input changes.",
            "falsify_like": "No privileged scale window survives; the special scale has to move ad hoc or disappears under updated inputs.",
        },
        "notes": [
            "One-loop running is used here only as a transparent public-data stress test.",
        ],
    }
    save_json(args.outdir / "test08_koide_scale_consistency_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
