# v59 implementation report

v59 converts the remaining manual-fill workflow into an explicit no-manual-fill public-parser workflow. Legacy fill/template artifacts are moved out of active paths, and the test runners search only current outputs/cache/public-like files for auto-built measurement artifacts.

Validation performed in the build environment:

```bash
python -m py_compile ccdr_r10_common.py run_all.py tests/*.py
```

Run with:

```powershell
python run_all.py --allow-large --max-mb 80000 --script-timeout 720000
```

Send back:

```text
outputs/round10_summary.json
outputs/test51_round10_joint_dashboard.json
```
