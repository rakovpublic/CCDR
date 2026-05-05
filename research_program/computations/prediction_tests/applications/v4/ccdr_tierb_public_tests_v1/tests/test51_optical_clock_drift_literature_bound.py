#!/usr/bin/env python3
"""T51: optical-clock drift literature bound

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
    parser = base_argparser("T51: optical-clock drift literature bound")
    args = parser.parse_args()
    try:
        result = run_test("T51", args)
    except Exception as exc:
        result = wrap_error("T51", exc)
    emit_result(result, args.outdir, "T51")


if __name__ == "__main__":
    main()
