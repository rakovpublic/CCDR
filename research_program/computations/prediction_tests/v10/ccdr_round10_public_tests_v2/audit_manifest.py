
#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "round10_manifest.json").read_text(encoding="utf-8"))
by_pred = {}
for m in manifest:
    by_pred.setdefault(m["prediction_id"], 0)
    by_pred[m["prediction_id"]] += 1

print(json.dumps({
    "n_tests": len(manifest),
    "n_predictions_or_groups": len(by_pred),
    "counts_by_prediction": dict(sorted(by_pred.items())),
}, indent=2, sort_keys=True))
