# CCDR / Synthesis public-data falsifiability scripts

These scripts implement eight public-data tests as separate Python files. Every
script downloads its own input data from public sources at runtime.

They are written to be:
- reproducible
- explicit about assumptions
- honest about approximation level

They are **not** exact reproductions of collaboration likelihoods where those are
not publicly packaged in a lightweight way.

## Files

- `test01_unified_nu_audit.py`
- `test02_leave_one_out_nu.py`
- `test03_oscillation_only_wz.py`
- `test04_independent_filament_replication.py`
- `test05_filament_null_controls.py`
- `test06_local_a0_hierarchical.py`
- `test07_dm_tower_preregistered_scan.py`
- `test08_koide_scale_consistency.py`
- `_common_public_data.py`

## Suggested environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
```

## Example usage

```bash
python test01_unified_nu_audit.py --outdir out01
python test02_leave_one_out_nu.py --outdir out02
python test03_oscillation_only_wz.py --outdir out03
python test04_independent_filament_replication.py --outdir out04
python test05_filament_null_controls.py --outdir out05
python test06_local_a0_hierarchical.py --outdir out06
python test07_dm_tower_preregistered_scan.py --outdir out07
python test08_koide_scale_consistency.py --outdir out08
```

Each script writes a JSON summary to its output directory.
