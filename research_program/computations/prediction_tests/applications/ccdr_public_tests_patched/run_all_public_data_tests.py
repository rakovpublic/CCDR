from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DEFAULT_SCRIPTS = [
    "test_fr4_transport_vs_collisionality_public.py",
    "test_fr7_mkss_vs_tauE_public.py",
    "test_mat1_ccdr_mu_vs_casimir_public.py",
    "test_mat3_lowT_sqrtT_public.py",
    "test_mat7_hbn_si_public.py",
    "test_el1_3d_nand_public.py",
]

OPTIONAL_DATA_GATE_SCRIPTS = [
    "test_fr3_elm_energy_scaling_public.py",
    "test_fr8_stellarator_vs_tokamak_public.py",
    "test_mat2_cos6theta_grain_boundary_public.py",
    "test_mat6_isotopic_diamond_public.py",
    "test_el3_si_area_volume_crossover_public.py",
    "test_el8_optical_vs_via_public.py",
]

ENV_FOR_SCRIPT = {
    "test_fr3_elm_energy_scaling_public.py": ["FR3_NUMERIC_URLS", "FR3_ELM_TABLE_URL"],
    "test_fr8_stellarator_vs_tokamak_public.py": ["FR8_STELLARATOR_TABLE_URL", "FR8_NUMERIC_URLS"],
    "test_mat2_cos6theta_grain_boundary_public.py": ["MAT2_NUMERIC_URLS"],
    "test_mat6_isotopic_diamond_public.py": ["MAT6_NUMERIC_URLS"],
    "test_el3_si_area_volume_crossover_public.py": ["EL3_NUMERIC_URLS"],
    "test_el8_optical_vs_via_public.py": ["EL8_NUMERIC_URLS"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CCDR public-data prediction tests.")
    p.add_argument(
        "--include-data-gates",
        action="store_true",
        help="Also run FR3/FR8/MAT2/MAT6/EL3/EL8 gate scripts even when no numeric URL env vars are set.",
    )
    p.add_argument("--only", nargs="*", help="Run only selected script names.")
    p.add_argument("--compact", action="store_true", help="Print compact verdict summaries instead of full reports.")
    p.add_argument(
        "--script-timeout",
        type=float,
        default=float(os.environ.get("CCDR_SCRIPT_TIMEOUT", "0") or 0),
        help="Optional per-script timeout in seconds. 0 means no subprocess timeout.",
    )
    p.add_argument(
        "--http-timeout",
        type=float,
        default=None,
        help="Override CCDR_HTTP_TIMEOUT for child scripts, in seconds.",
    )
    return p.parse_args()


def should_auto_include_optional(script: str) -> bool:
    return any(os.environ.get(k) for k in ENV_FOR_SCRIPT.get(script, []))


def extract_last_json(text: str) -> dict | None:
    """Return the last valid top-level JSON object printed by a child script.

    The old greedy regex could swallow warning dictionaries or tracebacks before
    the final report. This scanner tries every opening brace from the end and
    accepts the first suffix that parses.
    """
    stripped = text.strip()
    for i in range(len(stripped) - 1, -1, -1):
        if stripped[i] != "{":
            continue
        candidate = stripped[i:]
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def compact_summary(data: dict) -> dict:
    keep = ["prediction_id", "status", "verdict", "confirmation_level", "scientific_verdict", "partial_test_only"]
    return {k: data.get(k) for k in keep if k in data}


def run_child(path: Path, args: argparse.Namespace, child_env: dict[str, str]) -> subprocess.CompletedProcess:
    timeout = args.script_timeout if args.script_timeout and args.script_timeout > 0 else None
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        capture_output=bool(args.compact),
        text=True,
        timeout=timeout,
        env=child_env,
    )


def main() -> None:
    args = parse_args()
    if args.only:
        scripts = args.only
    else:
        scripts = list(DEFAULT_SCRIPTS)
        for script in OPTIONAL_DATA_GATE_SCRIPTS:
            if args.include_data_gates or should_auto_include_optional(script):
                scripts.append(script)

    skipped = [s for s in OPTIONAL_DATA_GATE_SCRIPTS if s not in scripts]
    if skipped:
        print("Skipping no-verdict data-gate tests by default:")
        for s in skipped:
            envs = ", ".join(ENV_FOR_SCRIPT.get(s, []))
            print(f" - {s}" + (f"  [enable with {envs}]" if envs else ""))
        print("Run with --include-data-gates, --only, or set the listed env vars to include them.\n")

    failures = []
    missing = []
    summaries = []
    child_env = os.environ.copy()
    if args.http_timeout is not None:
        child_env["CCDR_HTTP_TIMEOUT"] = str(args.http_timeout)
    for script in scripts:
        path = ROOT / script
        if not path.exists():
            missing.append(script)
            continue
        print(f"\n=== Running {script} ===")
        try:
            proc = run_child(path, args, child_env)
        except subprocess.TimeoutExpired as e:
            failures.append({"script": script, "returncode": "timeout", "timeout_s": args.script_timeout})
            print(f"TIMEOUT after {args.script_timeout:g}s: {script}")
            if args.compact:
                out = e.stdout or ""
                err = e.stderr or ""
                if isinstance(out, bytes):
                    out = out.decode("utf-8", errors="replace")
                if isinstance(err, bytes):
                    err = err.decode("utf-8", errors="replace")
                print((out + err)[-3000:])
            continue

        if args.compact:
            data = extract_last_json((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""))
            if data:
                summary = compact_summary(data)
                summaries.append({"script": script, **summary})
                print(json.dumps(summary, indent=2))
            else:
                print(((proc.stdout or "") + (proc.stderr or ""))[-3000:])
        if proc.returncode != 0:
            failures.append({"script": script, "returncode": proc.returncode})
    if args.compact and summaries:
        out = ROOT / "outputs" / "_batch_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"\nWrote compact summary to {out}")
    if missing:
        print("\nMissing scripts:")
        for s in missing:
            print(f" - {s}")
        raise SystemExit(1)
    if failures:
        print("\nSome scripts failed:")
        for f in failures:
            print(f" - {f['script']}: rc={f['returncode']}")
        raise SystemExit(1)
    print("\nAll selected scripts completed.")


if __name__ == "__main__":
    main()
