
# CCDR Round-10 Public Tests v2

Generated: 2026-05-03T20:39:31Z

This bundle contains 51 Python tests for CCDR v7.6 / Synthesis v3.6 public-data auditing.

## Goals

- Download public data automatically where endpoints are available.
- Never terminate without JSON: every test goes through `safe_json_main`.
- Use `data_limited` or `readiness_only` honestly when public data are too large, unavailable, or require a specialised parser.
- Keep SM-D tests separate from the main P1-P41 prediction list.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_all.py
```

Large products are disabled by default:

```bash
python run_all.py --allow-large --max-mb 5000
```

Filter by prediction or filename:

```bash
python run_all.py --only P41
python run_all.py --only SMD
python run_all.py --only pantheon
```

Outputs are written to:

```text
outputs/*.json
outputs/round10_summary.json
```

## Status semantics

- `partial`: a real public-data parser ran and produced a preliminary statistic.
- `diagnostic`: a real parser ran, but the statistic is intentionally not a final falsification statistic.
- `readiness_only`: public source exists/reachable; event-level or map-level parser is not yet decisive.
- `data_limited`: public endpoint unavailable, too large without `--allow-large`, or layout changed.
- `broken`: unexpected bug; should be fixed.

## Notes

This is a Round-10 starting bundle. The highest-priority upgrades are:
1. Replace inventory tests with exact row-schema parsers for DESI DR2 BAO and HEPData P41 tables.
2. Add ACT/Planck/Euclid map samplers behind `--allow-large`.
3. Add a true SPARC RAR/a0 parser using baryonic columns and galaxy metadata.
4. Add VAST/filament catalogue parsers for P3/P38 instead of metadata checks.
5. Add a dashboard that ingests all JSON outputs and classifies confirmed/plausible/null/falsified.
