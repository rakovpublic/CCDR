#!/usr/bin/env python3
"""T56: solar-sail residual proxy

Auto-download public-data Tier-B test. No manual input files.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tierb.tierb_common import base_argparser, emit_result, wrap_error
from tierb.tierb_runner import run_test


def main():
    parser = base_argparser("T56: solar-sail residual proxy")
    args = parser.parse_args()
    try:
        result = run_test("T56", args)
    except Exception as exc:
        result = wrap_error("T56", exc)
    emit_result(result, args.outdir, "T56")


if __name__ == "__main__":
    main()
