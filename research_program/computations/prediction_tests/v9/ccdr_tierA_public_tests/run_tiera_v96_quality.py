#!/usr/bin/env python3
"""Supplemental Tier-A v9.6 quality runner.

Run from the root of ccdr_tierA_public_tests after applying the patch:
    python run_tiera_v96_quality.py --cache .cache --outdir out_v9_6_quality --allow-large
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path

from tiera_quality_v96 import *


def run_t03_fix_probe(args):
    out = {"test_id": "T03", "improvement": "pantheon_columns_helper_available", "status": "ok"}
    # Try to find Pantheon cached file and validate column selector.
    files = list(Path(args.cache).rglob("Pantheon*SH0ES*.dat"))
    if files and pd is not None:
        try:
            df = pd.read_csv(files[0], sep=r"\s+", comment="#")
            z, mu, dmu = pantheon_columns_v96(df)
            out.update(pantheon_file=str(files[0]), selected_columns={"z": z, "mu": mu, "dmu": dmu}, n_rows=int(len(df)))
        except Exception as e:
            out.update(status="probe_failed", error=f"{type(e).__name__}: {e}")
    else:
        out.update(status="helper_written_no_cached_pantheon_file")
    return out


def run_kappa_product_probe(args, test_id):
    best = find_best_kappa_product_v96(args.cache)
    out = {"test_id": test_id, "improvement": "robust_kappa_map_product_classification", "status": "diagnostic_only", "best_product": best.get("best"), "n_candidates": best.get("n_candidates"), "candidate_sample": best.get("candidates", [])[:10]}
    if best.get("best") and best["best"].get("product_type") == "alm":
        out["recommendation"] = "ALM product selected; install healpy and allow alm2map, or prefer real-space kappa/convergence map."
    elif best.get("best") and best["best"].get("usable_without_transform"):
        out["recommendation"] = "real-space map found; original T04/T05/T16 should use this product instead of alm-only files."
    else:
        out["recommendation"] = "no usable kappa map found in cache; inspect ACT/Planck real-space map URLs."
    return out


def run_euclid_depth_probe(args):
    files = list(Path(args.cache).rglob("euclid_enriched_sample*.csv")) + list(Path(args.cache).rglob("*euclid*q1*catalogue*.csv"))
    out = {"test_id": "T06_T07", "improvement": "euclid_depth_proxy_control", "status": "data_limited"}
    if not files or pd is None:
        out["reason"] = "no_cached_euclid_sample_or_pandas"; return out
    try:
        df = pd.read_csv(files[0])
        d, meta = add_euclid_depth_proxy_v96(df)
        out.update(status="ok" if meta.get("ok") else "depth_proxy_unavailable", euclid_file=str(files[0]), depth_meta=meta, n_rows=int(len(d)))
        if "ra" in d.columns and "dec" in d.columns:
            # patch-count/depth correlation diagnostic
            import numpy as np
            ra = pd.to_numeric(d["ra"], errors="coerce"); dec = pd.to_numeric(d["dec"], errors="coerce"); dep = pd.to_numeric(d["depth_proxy_v96"], errors="coerce")
            mask = ra.notna() & dec.notna()
            if mask.sum() > 100:
                bins_ra = np.linspace(float(ra[mask].min()), float(ra[mask].max()), 12)
                bins_dec = np.linspace(float(dec[mask].min()), float(dec[mask].max()), 12)
                grid = []
                for i in range(len(bins_ra)-1):
                    for j in range(len(bins_dec)-1):
                        m = mask & (ra>=bins_ra[i]) & (ra<bins_ra[i+1]) & (dec>=bins_dec[j]) & (dec<bins_dec[j+1])
                        if m.sum() >= 10:
                            grid.append((int(m.sum()), float(dep[m].median()) if dep[m].notna().sum() else None))
                if len(grid) > 5:
                    counts = np.array([g[0] for g in grid], float); depths = np.array([g[1] for g in grid], float)
                    ok = np.isfinite(depths)
                    if ok.sum() > 5:
                        out["patch_count_depth_corr"] = float(np.corrcoef(counts[ok], depths[ok])[0,1])
                        out["n_patches"] = int(ok.sum())
    except Exception as e:
        out.update(status="probe_failed", error=f"{type(e).__name__}: {e}")
    return out


def run_vizier_probe(args):
    out = {"test_id": "T08", "improvement": "catalogue_specific_vizier_parser", "status": "diagnostic_only", "sources": []}
    sources = EXTRA_SOURCE_SEEDS_V96.get("T08", [])
    for url in sources:
        blob, meta = http_get_bytes(url, Path(args.cache)/"v96_vizier", timeout=args.timeout)
        rec = {"url": url, "meta": meta, "tables": []}
        if blob:
            tabs = read_vizier_like_table_v96(blob, safe_name(url))
            for lbl, df in tabs[:10]:
                rec["tables"].append({"label": lbl, "shape": [int(len(df)), int(len(df.columns))], "columns": list(map(str, df.columns))[:40], "match": match_column_groups_v96(df, DATA_CONTRACTS_V96["T08"]["required_column_groups"])})
        out["sources"].append(rec)
    out["status"] = "candidate_tables_found" if any(r["tables"] for r in out["sources"]) else "data_limited"
    return out


def run_discovery_probe(args, test_id):
    seeds = EXTRA_SOURCE_SEEDS_V96.get(test_id, [])
    contract = DATA_CONTRACTS_V96.get(test_id, {"required_column_groups": []})
    out = {"test_id": test_id, "improvement": "extra_automated_source_discovery", "status": "data_limited", "sources": []}
    for url in seeds:
        blob, meta = http_get_bytes(url, Path(args.cache)/f"v96_{test_id}", timeout=args.timeout)
        kind = artifact_kind_v96(url, meta.get("content_type"))
        rec = {"url": url, "kind": kind, "meta": meta, "links_sample": [], "candidate_tables": []}
        if blob:
            if kind in ["html_article_or_landing_page", "pdf_article_or_report", "json_unknown_needs_physical_gate", "metadata_record"]:
                text = blob.decode("utf-8", errors="ignore") if kind != "pdf_article_or_report" else ""
                if text:
                    rec["links_sample"] = extract_links_from_text_v96(text, url)[:30]
            if kind != "metadata_record":
                for lbl, df in dataframe_from_bytes(blob, safe_name(url))[:20]:
                    tier = "primary_structured_public_table" if kind == "physical_data_artifact" else "secondary_auto_extracted_table"
                    score = score_candidate_table_v96(df, contract, url, tier)
                    score["label"] = lbl
                    score["columns_sample"] = list(map(str, df.columns))[:50]
                    rec["candidate_tables"].append(score)
        out["sources"].append(rec)
    quals = [c for r in out["sources"] for c in r["candidate_tables"] if c.get("qualifies_for_model")]
    out["qualifying_candidate_count"] = len(quals)
    out["status"] = "candidate_tables_found" if quals else "data_limited"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--outdir", default="out_v9_6_quality")
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--allow-large", action="store_true")
    args = ap.parse_args()
    outdir = ensure_dir(args.outdir)
    results = []
    tasks = [
        ("t03_helper_fix", lambda: run_t03_fix_probe(args)),
        ("t04_kappa_loader", lambda: run_kappa_product_probe(args, "T04")),
        ("t05_kappa_loader", lambda: run_kappa_product_probe(args, "T05")),
        ("t16_kappa_loader", lambda: run_kappa_product_probe(args, "T16")),
        ("t06_t07_euclid_depth", lambda: run_euclid_depth_probe(args)),
        ("t08_vizier_parser", lambda: run_vizier_probe(args)),
        ("t21_spectral_distortion_sources", lambda: run_discovery_probe(args, "T21")),
        ("t23_bmode_sources", lambda: run_discovery_probe(args, "T23")),
        ("t25_eta_s_sources", lambda: run_discovery_probe(args, "T25")),
        ("t15_posterior_sources", lambda: run_discovery_probe(args, "T15")),
        ("t17_posterior_sources", lambda: run_discovery_probe(args, "T17")),
        ("t24_posterior_sources", lambda: run_discovery_probe(args, "T24")),
    ]
    for name, fn in tasks:
        try:
            r = fn()
        except Exception as e:
            r = {"task": name, "status": "error", "error": f"{type(e).__name__}: {e}"}
        r["task"] = name
        write_json(outdir / f"{name}.json", r)
        results.append(r)
        print(json.dumps(json_safe({"task": name, "status": r.get("status"), "test_id": r.get("test_id")}), sort_keys=True))
    summary = {"schema": "ccdr-tierA-v9.6-quality-summary", "n_tasks": len(results), "results": results}
    write_json(outdir / "summary_v96.json", summary)
    print(json.dumps(json_safe(summary), indent=2)[:8000])

if __name__ == "__main__":
    main()
