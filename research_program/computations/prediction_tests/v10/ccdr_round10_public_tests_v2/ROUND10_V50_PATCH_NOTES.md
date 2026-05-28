# Round-10 v50 confirm-recovery / measurement-schema patch

v50 is built on top of v49. It keeps the stricter no-false-confirm policy, but adds concrete recovery paths and machine-readable measurement ingestors so blocked tests can become confirms when the required products exist.

Implemented improvements:

1. Dashboard bucket correctness: claim-grade non-SM confirms, SM consistency checks, coverage confirmations, readiness, blocked tests, and failed gates are separated.
2. P36 high-z raw-source recovery: unit_provenance paths are parsed and hashed/audited separately from generated `outputs/` artifacts.
3. P36 radius-quality reanalysis: writes large-radius summaries and refuses publication/global confirm unless radius>=0.5 kpc has >=2 source groups with >=20 rows each.
4. P30 route/global split: SDSS route-confirm, empirical-mask manifest, and global-confirm gates are separated.
5. P33 density BAO measurement ingestor: accepts a real `p33*alpha*measurement*.json/csv` and writes a required template.
6. P8/PTA source audit: parses cached `.par` files for RAJ/DECJ and hashes them; still requires weighted statistic + sky null for confirm.
7. CL2/PTA weighted-statistic ingestor: accepts `pta*weighted*statistic*.json/csv` and writes a template.
8. P32 ringdown likelihood ingestor: accepts `p32*strain*likelihood*.json` products and writes a template.
9. P40/P41 likelihood ingestors: accepts BB likelihood and q2/Wilson likelihood products and writes templates.
10. SMD derivation metadata ingestor: writes a preregistered derivation template and promotes only if all anti-postdiction fields pass.

Run:

```powershell
python run_all.py --allow-large --max-mb 80000 --script-timeout 720000
```

After the run, send `outputs/round10_summary.json`, `outputs/test51_round10_joint_dashboard.json`, and ideally the whole `outputs` folder.
