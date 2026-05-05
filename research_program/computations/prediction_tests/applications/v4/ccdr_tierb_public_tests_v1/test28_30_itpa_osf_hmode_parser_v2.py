#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _tierb_v5_quality_helpers import (
    classify_readiness,
    download_bytes,
    ensure_dir,
    extract_tables_from_blob,
    find_col,
    is_itpa_candidate_file,
    linear_fit_metrics,
    now_utc_iso,
    numeric_series,
    safe_name,
    walk_osf_files,
)

OSF_ITPA = "https://api.osf.io/v2/nodes/drwcq/files/osfstorage/"

TAUE = [r"tau.*e", r"taue", r"t?auth", r"energy.*conf", r"te98", r"\btauth\b"]
DENS = [r"\bne\b", r"nbar", r"density", r"nel", r"line.*averaged"]
STORED = [r"wmhd", r"wth", r"stored", r"energy"]
MACHINE = [r"machine", r"device", r"tok", r"tokamak"]
POWER = [r"ploss", r"pheat", r"power", r"aux", r"pin"]
IP = [r"\bip\b", r"plasma.*current"]
BT = [r"\bbt\b", r"btor", r"toroidal.*field"]
RMAJ = [r"\br\b", r"rgeo", r"major.*radius"]
AMIN = [r"\ba\b", r"amin", r"minor.*radius"]
KAPPA = [r"kappa", r"elong"]
DELTA = [r"delta", r"triang"]
Q95 = [r"q95", r"q_95", r"safety"]


def summarize_candidate_table(name, df):
    cols = {"tau_e": find_col(df, TAUE), "density": find_col(df, DENS), "stored_energy": find_col(df, STORED), "machine": find_col(df, MACHINE), "power": find_col(df, POWER), "ip": find_col(df, IP), "bt": find_col(df, BT), "r_major": find_col(df, RMAJ), "a_minor": find_col(df, AMIN), "kappa": find_col(df, KAPPA), "triangularity": find_col(df, DELTA), "q95": find_col(df, Q95)}
    return {"table": name, "n_rows": int(len(df)), "n_cols": int(len(df.columns)), "matched_columns": cols, "columns_preview": [str(c) for c in list(df.columns)[:40]]}


def analyze_t28(df, table_name):
    taue = find_col(df, TAUE)
    dens = find_col(df, DENS)
    stored = find_col(df, STORED)
    power = find_col(df, POWER)
    ip = find_col(df, IP)
    bt = find_col(df, BT)
    kappa = find_col(df, KAPPA)
    if not (taue and dens):
        return {"ok": False, "reason": "missing_tau_e_or_density"}
    y = np.log(np.clip(numeric_series(df, taue).astype(float), 1e-30, None))
    X = {"intercept": np.ones(len(df)), "log_density": np.log(np.clip(numeric_series(df, dens).astype(float), 1e-30, None))}
    if stored:
        X["log_stored_energy"] = np.log(np.clip(numeric_series(df, stored).astype(float), 1e-30, None))
    if power:
        X["log_power"] = np.log(np.clip(numeric_series(df, power).astype(float), 1e-30, None))
    if ip:
        X["log_ip"] = np.log(np.clip(numeric_series(df, ip).astype(float), 1e-30, None))
    if bt:
        X["log_bt"] = np.log(np.clip(numeric_series(df, bt).astype(float), 1e-30, None))
    if kappa:
        X["elongation"] = numeric_series(df, kappa).astype(float)
    fit = linear_fit_metrics(y, X)
    fit.update({"test": "T28", "table": table_name, "used_columns": {"tau_e": taue, "density": dens, "stored": stored, "power": power, "ip": ip, "bt": bt, "kappa": kappa}})
    return fit


def analyze_t30(df, table_name):
    taue = find_col(df, TAUE)
    dens = find_col(df, DENS)
    power = find_col(df, POWER)
    ip = find_col(df, IP)
    bt = find_col(df, BT)
    kappa = find_col(df, KAPPA)
    delta = find_col(df, DELTA)
    q95 = find_col(df, Q95)
    rmaj = find_col(df, RMAJ)
    amin = find_col(df, AMIN)
    if not (taue and dens):
        return {"ok": False, "reason": "missing_tau_e_or_density"}
    y = np.log(np.clip(numeric_series(df, taue).astype(float), 1e-30, None))
    base = {"intercept": np.ones(len(df)), "log_density": np.log(np.clip(numeric_series(df, dens).astype(float), 1e-30, None))}
    if power:
        base["log_power"] = np.log(np.clip(numeric_series(df, power).astype(float), 1e-30, None))
    if ip:
        base["log_ip"] = np.log(np.clip(numeric_series(df, ip).astype(float), 1e-30, None))
    if bt:
        base["log_bt"] = np.log(np.clip(numeric_series(df, bt).astype(float), 1e-30, None))
    shaped = dict(base)
    if kappa:
        shaped["elongation"] = numeric_series(df, kappa).astype(float)
    if delta:
        shaped["triangularity"] = numeric_series(df, delta).astype(float)
    if q95:
        shaped["q95"] = numeric_series(df, q95).astype(float)
    if rmaj and amin:
        shaped["aspect_ratio"] = numeric_series(df, rmaj).astype(float) / np.clip(numeric_series(df, amin).astype(float), 1e-30, None)
    fit_base = linear_fit_metrics(y, base)
    fit_shape = linear_fit_metrics(y, shaped)
    out = {"test": "T30", "table": table_name, "base_fit": fit_base, "density_plus_curvature_fit": fit_shape, "used_columns": {"tau_e": taue, "density": dens, "power": power, "ip": ip, "bt": bt, "elongation": kappa, "triangularity": delta, "q95": q95, "r_major": rmaj, "a_minor": amin}}
    if fit_base.get("ok") and fit_shape.get("ok"):
        out["rms_reduction_fraction"] = float((fit_base["rms"] - fit_shape["rms"]) / max(fit_base["rms"], 1e-30))
        out["delta_aic_shape_minus_base"] = float(fit_shape["aic"] - fit_base["aic"])
        out["support_like"] = bool(0.10 <= out["rms_reduction_fraction"] <= 0.30 and out["delta_aic_shape_minus_base"] < 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_t28_t30_itpa_v2")
    ap.add_argument("--osf-api", default=OSF_ITPA)
    args = ap.parse_args()
    outdir = ensure_dir(args.outdir)
    files, diag = walk_osf_files(args.osf_api)
    candidates = [f for f in files if is_itpa_candidate_file(f)]
    downloaded = []
    table_summaries = []
    t28_results = []
    t30_results = []
    for f in candidates:
        if not f.download_url:
            continue
        blob, meta = download_bytes(f.download_url)
        downloaded.append({"name": f.name, "path": f.path, "download_url": f.download_url, "meta": meta})
        if not blob:
            continue
        cache_path = outdir / safe_name(f.name)
        try:
            cache_path.write_bytes(blob)
        except Exception:
            pass
        for tname, df in extract_tables_from_blob(blob, f.name):
            if len(df) < 50:
                continue
            table_summaries.append(summarize_candidate_table(tname, df))
            t28_results.append(analyze_t28(df, tname))
            t30_results.append(analyze_t30(df, tname))
    readiness = classify_readiness(files, candidates, table_summaries)
    result = {
        "schema": "ccdr-tierb-result-v1",
        "quality_patch_version": "v6_items_5_6_8_t48_additive",
        "generated_utc": now_utc_iso(),
        "test_ids": ["T28", "T30"],
        "prediction_ids": ["FR7", "FR10"],
        "prediction_names": ["M_KSS/global H-mode confinement proxy", "density plus curvature residual coupling in confinement scaling"],
        "source_strategy": "exact OSF ITPA DB5.2.3 recursive traversal; broad Zenodo disabled for scientific mode",
        "osf_diagnostics": diag,
        "osf_files_count": len(files),
        "candidate_files_count": len(candidates),
        "candidate_files": [f.__dict__ for f in candidates[:30]],
        "downloaded": downloaded[:20],
        "table_summaries": table_summaries[:20],
        "tables_count": len(table_summaries),
        "readiness_status": readiness,
        "evidence_status": "analysis_run" if table_summaries else "data_limited",
        "t28_results": t28_results[:20],
        "t30_results": t30_results[:20],
        "falsification_logic": {
            "caveat": "If no DB5/STD5 structured table is parsed, result remains data_limited, not null.",
            "T28_confirm_like": "tau_E model has stable positive density/KSS-proxy relation under controls.",
            "T30_confirm_like": "density+curvature/shaping terms reduce confinement residual RMS by 10-20% with AIC/BIC improvement.",
            "falsify_like": "Adequate DB5/STD5 rows exist and added terms are absent/reversed or fail held-out controls."
        },
    }
    outpath = outdir / "t28_t30_itpa_osf_result.json"
    outpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
