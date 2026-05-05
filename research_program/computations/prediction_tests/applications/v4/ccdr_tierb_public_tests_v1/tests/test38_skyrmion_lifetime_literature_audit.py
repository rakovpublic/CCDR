#!/usr/bin/env python3
"""T38: skyrmion lifetime literature audit

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
    parser = base_argparser("T38: skyrmion lifetime literature audit")
    args = parser.parse_args()
    try:
        result = run_test("T38", args)
    except Exception as exc:
        result = wrap_error("T38", exc)
    emit_result(result, args.outdir, "T38")


if __name__ == "__main__":
    main()
