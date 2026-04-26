# CCDR v7.3 / Synthesis v3.3 — public-data 12-test bundle

This bundle implements the **round-5 12-test screening battery** as separate Python files.

These scripts are designed to:
- download all needed inputs from public sources,
- avoid private collaboration products,
- stay honest about proxy status,
- emit JSON outputs under `outputs/`,
- fail loudly rather than silently when a public mirror changes.

## Files

- `_common_public_data.py` — shared download, parsing, statistics, and public-source helpers.
- `test01_p30_non_euclid_lensing_replication.py`
- `test02_p30_euclid_q1_systematics_audit.py`
- `test03_highz_a0_sparc_kmos3d_proxy.py`
- `test04_public_nu_redrive.py`
- `test05_euclid_q1_filament_ordering.py`
- `test06_p29_multi_redshift_growth.py`
- `test07_p36_mond_sequence_nu_extractor.py`
- `test08_p33_density_correlated_bao.py`
- `test09_cl2_p8c_reducing_volume_cross.py`
- `test10_p38_void_wall_cauchy_tail.py`
- `test11_hardcore_window_readiness.py`
- `test12_p37_phase_space_drift_proxy.py`
- `run_all.py`

## Setup

Python 3.10+ is recommended.

Base dependencies used directly by the scripts:
- numpy
- pandas
- scipy
- requests

Optional packages are auto-installed if needed:
- astropy
- astropy-healpix
- openpyxl

## Usage

Run one test:

```bash
python test01_p30_non_euclid_lensing_replication.py
```

Run the full battery:

```bash
python run_all.py
```

Each script prints a JSON result to stdout and writes the same JSON under `outputs/`.

## Important caveats

These are **screening / proxy** implementations, matching the stated role of the round-5 battery.

A few public inputs are stable and straightforward:
- Euclid Q1 via IRSA TAP
- ACT DR6 via LAMBDA
- Pantheon+ via GitHub DataRelease
- DESI DR2 BAO via public Cobaya data repo
- NANOGrav 15-year via Zenodo
- SPARC via Zenodo
- XENONnT WIMP curves via public GitHub repo
- PandaX-4T first-analysis points via public data page

A few public inputs are more fragile because filenames or release tags can change:
- Planck PR4 lensing asset names on GitHub releases
- KMOS3D tarball path stability on the MPE site
- Some direct-detection CSV/HEPData table identifiers

The helper layer therefore tries one or more public candidates and records the failure reason if none work.

## Why some tests are called “proxy”

This bundle follows the current CCDR/Synthesis framing:
- T1/T2/T3/T4/T5/T6/T7/T8/T9/T10/T11/T12 are public-data screening tests, not collaboration-grade likelihood analyses.
- T8 is a scaffold until density-binned DESI DR3 likelihoods are public.
- T11 is a readiness audit, not direct peak counting.
- T12 is a simulated phase-space proxy, not an event-level direct-detection measurement.

## Output discipline

Each script includes:
- the test name,
- a short description,
- public source URLs,
- quantitative outputs,
- simple support/falsify-like screening indicators where appropriate.
