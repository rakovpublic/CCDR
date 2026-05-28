#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import os
import time
import uuid
from pathlib import Path

RUNNER_VERSION = "v80"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _status_summary(results):
    summary = {}
    for r in results:
        st = r.get("status", "unknown")
        summary[st] = summary.get(st, 0) + 1
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=".ccdr_round10_cache")
    ap.add_argument("--allow-large", action="store_true")
    ap.add_argument("--max-mb", type=float, default=250.0)
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="substring filter for script filename or prediction id/name")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--script-timeout", type=int, default=180, help="wall-clock timeout per test subprocess; prevents one data endpoint from hanging all-test runs")
    ap.add_argument("--resume", action="store_true", help="reuse existing current-run test JSON files when present and valid")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    tests = sorted((root / "tests").glob("test*.py"))
    manifest = json.loads((root / "round10_manifest.json").read_text(encoding="utf-8"))

    if args.only:
        keep = []
        for t in tests:
            m = next((x for x in manifest if x.get("file") == t.name), {})
            blob = " ".join([t.name, str(m.get("prediction_id", "")), str(m.get("prediction_name", ""))]).lower()
            if args.only.lower() in blob:
                keep.append(t)
        tests = keep


    def _runner_quarantine_manual_fill_v60():
        # Remove active legacy manual-fill/template artifacts before tests run.
        ex = root / "docs" / "examples" / "legacy_manual_fill_artifacts_quarantined_v60"
        ex.mkdir(parents=True, exist_ok=True)
        moved = []
        for active in [root / "inputs", root / "measurements"]:
            if not active.exists():
                continue
            for p in list(active.rglob("*")):
                if not p.is_file():
                    continue
                s = str(p).lower()
                if not any(tok in s for tok in ["template", "_fill", "fill_before", "fillbefore", "example", "placeholder", "manual_fill"]):
                    continue
                dest = ex / p.name
                try:
                    if dest.exists():
                        dest = ex / (p.stem + "_" + uuid.uuid4().hex[:6] + p.suffix)
                    p.replace(dest)
                    moved.append({"from": str(p), "to": str(dest)})
                except Exception as e:
                    moved.append({"from": str(p), "error": str(e)})
        if moved:
            _write_json(root / "outputs" / "v60_runner_quarantined_manual_fill_artifacts.json", {"n_moved": len(moved), "moved": moved})

    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)
    _runner_quarantine_manual_fill_v60()
    current_run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8]

    if not args.resume and os.environ.get("CCDR_KEEP_PREVIOUS_OUTPUTS", "0") not in {"1", "true", "TRUE", "yes"}:
        for old in out_dir.glob("test*.json"):
            try:
                old.unlink()
            except Exception:
                pass
        for name in [
            "round10_summary.json",
            "round10_partial_summary_v51.json", "round10_partial_summary_v52.json", "round10_partial_summary_v53.json",
            "round10_partial_summary_v54.json", "round10_partial_summary_v55.json", "round10_partial_summary_v56.json", "round10_partial_summary_v57.json", "round10_partial_summary_v58.json", "round10_partial_summary_v59.json", "round10_partial_summary_v60.json",
            "round10_partial_summary_v61.json", "round10_partial_summary_v62.json", "round10_partial_summary_v63.json", "round10_partial_summary_v64.json", "round10_partial_summary_v65.json", "round10_partial_summary_v66.json", "round10_partial_summary_v67.json", "round10_partial_summary_v68.json", "round10_partial_summary_v69.json", "round10_partial_summary_v70.json", "round10_partial_summary_v71.json", "round10_partial_summary_v72.json", "round10_partial_summary_v73.json", "round10_partial_summary_v74.json", "round10_partial_summary_v75.json", "round10_partial_summary_v76.json", "round10_partial_summary_v77.json", "round10_partial_summary_v78.json", "round10_partial_summary_v79.json", "round10_partial_summary_v80.json",
            "current_run_progress_v51.json", "current_run_progress_v52.json", "current_run_progress_v53.json",
            "current_run_progress_v54.json", "current_run_progress_v55.json", "current_run_progress_v56.json", "current_run_progress_v57.json", "current_run_progress_v58.json", "current_run_progress_v59.json", "current_run_progress_v60.json", "current_run_progress_v61.json", "current_run_progress_v62.json", "current_run_progress_v63.json", "current_run_progress_v64.json", "current_run_progress_v65.json", "current_run_progress_v66.json", "current_run_progress_v67.json", "current_run_progress_v68.json", "current_run_progress_v69.json", "current_run_progress_v70.json", "current_run_progress_v71.json", "current_run_progress_v72.json", "current_run_progress_v73.json", "current_run_progress_v74.json", "current_run_progress_v75.json", "current_run_progress_v76.json", "current_run_progress_v77.json", "current_run_progress_v78.json", "current_run_progress_v79.json", "current_run_progress_v80.json",
        ]:
            try:
                (out_dir / name).unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    for alias in ["current_run_id_v80.txt", "current_run_id_v79.txt", "current_run_id_v78.txt", "current_run_id_v77.txt", "current_run_id_v76.txt", "current_run_id_v75.txt", "current_run_id_v74.txt", "current_run_id_v73.txt", "current_run_id_v72.txt", "current_run_id_v71.txt", "current_run_id_v70.txt", "current_run_id_v69.txt", "current_run_id_v68.txt", "current_run_id_v67.txt", "current_run_id_v66.txt", "current_run_id_v65.txt", "current_run_id_v64.txt", "current_run_id_v63.txt", "current_run_id_v62.txt", "current_run_id_v61.txt", "current_run_id_v60.txt", "current_run_id_v59.txt", "current_run_id_v58.txt", "current_run_id_v57.txt", "current_run_id_v56.txt", "current_run_id_v55.txt", "current_run_id_v54.txt", "current_run_id_v53.txt", "current_run_id_v52.txt", "current_run_id_v51.txt", "current_run_id_v50.txt", "current_run_id_v49.txt"]:
        (out_dir / alias).write_text(current_run_id, encoding="utf-8")

    results = []
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    expected = len(tests)

    def stamp_ids(obj: dict) -> dict:
        for v in range(49, 81):
            obj[f"current_run_id_v{v}"] = current_run_id
        return obj

    def write_progress(completed: bool = False):
        bundle = {
            "round": 10,
            "runner_version": RUNNER_VERSION,
            "started_utc": started_utc,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "expected_tests": expected,
            "n_tests_run": len(results),
            "missing_tests_v51": [t.stem for t in tests if not any(r.get("script_stem") == t.stem for r in results)],
            "summary": _status_summary(results),
            "results": results,
            "v57_note": "v57 auto-builds strict confirm artifacts from public/cached data when possible; no manual artifact fill is required.",
            "v59_note": "v59 removes active manual-fill artifacts and uses only auto public/cache parsers for tests.",
            "v60_note": "v60 adds source-targeted public parsers, same-mask recomputation diagnostics, automated alpha/likelihood gates, and why-not-confirm classes; no manual filling.",
            "v61_note": "v61 adds concrete no-manual estimators for P36/P30/P33/PTA/P32/P40/P41.",
            "v63_note": "v63 adds source-targeted P36 fetch/parse, strict radius provenance and source bootstrap; P30 patch/confounding resolver; DESI LSS alpha proxy; PTA residual proxy; and likelihood gate refinements.",
            "v64_note": "v64 deepens behavior: P36 physical-radius claim rows and source-2 parser, P30 pre-sign patch rejection/recompute, DESI LSS alpha bootstrap proxy, PTA/P40/P41 statistic parsers; no manual filling.",
            "v65_note": "v65 implements deep KGES+KMOS3D source-2 parsers, P36 radius audit/bootstrap, P30 single empirical mask residualization, exact DESI LSS alpha proxy, PTA residual-kappa join, and P32 detector split focus; no manual filling.",
            "v66_note": "v66 implements report-driven confirm recovery: P36 clean physical-radius claim subset, durable P30/P33/PTA/P32/P40/P41/P3 recovery artifacts, and strict dashboard accounting.",
            "v67_note": "v67 implements all six requested recovery improvements: BK18 covariance parsing, P41 numeric fitter, DESI exact/compressed ingestion, PTA join scan, P32 detector discovery, and P30 strict mask/residualization gates.",
            "v68_note": "v68 implements six follow-up improvements: P41 public Supplementary.tex table fitter, P33 labeled compressed/exact DESI audit, P32 H1/L1 text-strain likelihoods, PTA join/coordinate audit, P30 control-subtracted mask resolver, and P39/P1 covariance plus model-penalty artifact.",
            "v69_note": "v69 implements six latest-run improvements: CL6 bridge confirm from confirmed P40/P41 anchors, P32 whitening/nulls/injection audit, P30 accepted-mask curl closure, P33 exact LSS/random ingestion, PTA par/tim residual-kappa join, and P39/P1 full-covariance model chain.",
            "v70_note": "v70 implements seven confirm-focused improvements: P36 memory-safe clean-row recovery, PTA signed residual-kappa gate, P33 DESI exact-input manifest, P30 SDSS route confirm, P32 multi-event strain manifest, P39 covariance audit, and P3 endpoint semantic search.",
            "v71_note": "v71 implements seven follow-up confirm improvements: P33 exact density-split alpha pipeline, P30 global route closure, P35 harmonic/P(k) parser, P3 strict endpoint parser, P32 multi-event strain gate, P39 covariance/penalty chain, and DCN source extraction.",
            "v72_note": "v72 implements seven latest confirm improvements: PTA public-only residual-kappa recovery, P33 exact LSS/random density-split gate, P30 cross-route closure, P35 LSS P(k)/xi harmonic parser, P3 endpoint semantic recovery, P32 distinct-event strain index, and P39/P1 publication likelihood chain.",
            "v73_note": "v73 implements seven follow-up improvements: P30 SDSS route repair plus global split, PTA public source scan, P33 exact LSS/random scan, P35 LSS P(k)/xi scan, P3 endpoint semantic scan, P32 multi-event strain source audit, and P39/P1 full covariance/systematics audit.",
            "v74_note": "v74 implements seven confirm-recovery improvements: P30 second-route audit, PTA residual-kappa join builder, P33 exact LSS/random ingestion audit, P35 P(k)/xi harmonic audit, P3 endpoint catalogue recovery, P32 multi-event strain source audit, and P39/P1 covariance/systematics likelihood chain.",
            "v75_note": "v75 implements concrete confirm-recovery parsers/statistics: P30 Euclid geometry and Planck map audit, PTA TOA-weight parsing, P33 archive-aware LSS/random scan, P35 extended P(k)/xi scan, P3 endpoint edge parser, P32 event API strain index, and P39 covariance/systematics classifier.",
            "v76_note": "v76 implements all eight confirm-focused improvements: strict P3 endpoint orientation, PTA residual-kappa pair scan, P30 Planck same-mask gate, P30 Euclid kappa-cell gate, P32 second-strain gate, P33 exact-alpha fit, P39 full-covariance likelihood gate, and P35 exact P(k)/xi harmonic gate.",
            "v77_note": "v77 implements deeper source-recovery manifests for all eight confirm paths: real P3 endpoint sources, PTA residual/postfit sources, P30 Planck/Euclid map/kappa source manifests, P32 second-strain sources, P33 exact LSS/random sources, P39 Pantheon covariance sources, and P35 exact P(k)/xi sources.",
            "v78_note": "v78 implements archive-aware confirm recovery for all eight paths: PTA residual-kappa member joins, P32 nested GWOSC URL resolution, P30 Planck FITS/ALM and Euclid kappa-cell parsers, P33 exact LSS/random archive parsing, P39 flattened Pantheon covariance parsing, P35 archive P(k)/xi parsing, and P3 endpoint archive/member parsing.",
            "v79_note": "v79 implements measured confirm-recovery diagnostics: low-nside public lensing same-mask sampling, strict P35 P(k)/xi filtering and bootstrap nulls, direct NANOGrav .res residual-kappa joining, widened P32 local strain discovery, expanded P33 LSS aliases, P39 covmat parsing, Euclid cross-route kappa cells, and P3 FITS endpoint parsing.",
            "v80_note": "v80 implements eight confirm-focused follow-ups: P8/T16 promotion from direct residual-kappa evidence, true Planck-source and Euclid-native P30 resolvers, P33 exact LSS classification, P35 FITS/table P(k)/xi aliases, P32 archive-member strain discovery, P3 graph endpoint parsing, and P39 full-covariance recovery.",
        }
        stamp_ids(bundle)
        for v in range(51, 81):
            bundle[f"run_complete_v{v}"] = bool(completed and len(results) == expected)

        for v in range(51, 81):
            _write_json(out_dir / f"round10_partial_summary_v{v}.json", bundle)
        progress_keys = ["runner_version", "expected_tests", "n_tests_run", "missing_tests_v51", "summary"]
        progress_obj = {k: bundle[k] for k in progress_keys if k in bundle}
        stamp_ids(progress_obj)
        for v in range(51, 81):
            progress_obj[f"run_complete_v{v}"] = bundle[f"run_complete_v{v}"]
            _write_json(out_dir / f"current_run_progress_v{v}.json", progress_obj)
        if completed:
            _write_json(out_dir / "round10_summary.json", bundle)
        return bundle

    try:
        for script in tests:
            existing_path = out_dir / (script.stem + ".json")
            if args.resume and existing_path.exists():
                try:
                    obj = json.loads(existing_path.read_text(encoding="utf-8"))
                    obj.setdefault("script_stem", script.stem)
                    stamp_ids(obj)
                    results.append(obj)
                    print(f"{script.name}: {obj.get('status')} (resumed)")
                    write_progress(False)
                    continue
                except Exception:
                    pass

            cmd = [sys.executable, str(script), "--cache-dir", args.cache_dir, "--max-mb", str(args.max_mb), "--timeout", str(args.timeout)]
            if args.allow_large:
                cmd.append("--allow-large")
            if args.quick:
                cmd.append("--quick")
            t0 = time.time()
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
            env["CCDR_R10_CURRENT_RUN_ID"] = current_run_id
            for v in range(49, 81):
                env[f"CCDR_R10_CURRENT_RUN_ID_V{v}"] = current_run_id
            try:
                proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, env=env, timeout=args.script_timeout)
            except subprocess.TimeoutExpired as e:
                proc = subprocess.CompletedProcess(cmd, 124, stdout=e.stdout or "", stderr=((e.stderr or "") + f"\nSCRIPT_TIMEOUT after {args.script_timeout}s"))
            elapsed = time.time() - t0
            stdout = proc.stdout.strip()
            try:
                obj = json.loads(stdout)
            except Exception:
                obj = {
                    "test_id": script.stem,
                    "status": "runner_parse_error",
                    "returncode": proc.returncode,
                    "stdout": stdout[-4000:],
                    "stderr": (proc.stderr or "")[-4000:],
                }
            obj["script_stem"] = script.stem
            obj["runner_elapsed_s"] = round(elapsed, 3)
            obj["runner_returncode"] = proc.returncode
            stamp_ids(obj)
            results.append(obj)
            _write_json(existing_path, obj)
            print(f"{script.name}: {obj.get('status')} ({elapsed:.1f}s)")
            write_progress(False)
            if args.stop_on_fail and obj.get("status") in {"broken", "runner_parse_error"}:
                break
    finally:
        completed = len(results) == expected
        bundle = write_progress(completed)
        if not completed:
            _write_json(out_dir / "round10_summary.json", bundle)

    final = {
        "runner_version": RUNNER_VERSION,
        "n_tests_run": len(results),
        "expected_tests": expected,
        "summary": _status_summary(results),
        "summary_path": str(out_dir / "round10_summary.json"),
    }
    for v in range(51, 81):
        final[f"run_complete_v{v}"] = len(results) == expected
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
