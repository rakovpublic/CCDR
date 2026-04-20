#!/usr/bin/env python3
"""
T14 / CL2 / P8c — PTA × Planck PR4 kappa cross-correlation.

This script downloads public NANOGrav 15-year timing products and public
Planck PR4 lensing maps, samples the convergence field at pulsar positions,
and tests whether pulsars with larger residual/noise proxy show systematically
higher kappa values.

It is a public-data screening/proxy implementation. The ideal collaboration-
grade test would construct a true sky field and harmonic cross-spectrum.
Here we use pulsar-position sampling plus bootstrap robustness.
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr

from common_public_utils import (
    bootstrap_stat,
    download_github_release_asset,
    download_zenodo_file,
    ensure_dir,
    extract_archive,
    find_files,
    find_planck_pr4_files,
    log,
    read_tim_error_proxy,
    sample_healpix_map_value,
    save_json,
    skycoord_from_par,
    try_pint_wrms,
)


def download_nanograv(cache_dir: Path) -> Path:
    archive = cache_dir / "NANOGrav15yr_PulsarTiming.tar.gz"
    download_zenodo_file(
        record_ids=[16051178, 8423265],
        patterns=[r"NANOGrav15yr_PulsarTiming.*\.tar\.gz$"],
        dest=archive,
    )
    root = cache_dir / "nanograv15yr"
    return extract_archive(archive, root)


def download_planck_pr4(cache_dir: Path) -> Path:
    archive = cache_dir / "PR42018like_maps.tar"
    download_github_release_asset(
        owner="carronj",
        repo="planck_PR4_lensing",
        asset_name_regex=r"PR42018like_maps\.tar$",
        dest=archive,
        release_name_regex=r"Lensing maps and chains",
    )
    root = cache_dir / "planck_pr4"
    return extract_archive(archive, root)




def _norm_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s).upper()


def find_matching_tim(par: Path, tims: List[Path]) -> Optional[Path]:
    pstem = _norm_name(par.stem)
    same_branch = [t for t in tims if ('wideband' in str(t).lower()) == ('wideband' in str(par).lower())]
    search_space = same_branch or tims
    exact = [t for t in search_space if _norm_name(t.stem) == pstem]
    if exact:
        return sorted(exact, key=lambda x: len(str(x)))[0]
    contains = [t for t in search_space if pstem in _norm_name(t.stem) or _norm_name(t.stem) in pstem]
    if contains:
        return sorted(contains, key=lambda x: len(str(x)))[0]
    return None

def collect_pulsars(root: Path, max_pulsars: int, require_wrms: bool) -> List[Dict[str, object]]:
    pars = sorted(find_files(root, [r"\.par$"]))
    tims = sorted(find_files(root, [r"\.tim$"]))
    items: List[Dict[str, object]] = []
    for par in pars:
        name = par.stem
        tim = find_matching_tim(par, tims)
        if tim is None:
            continue
        coord = skycoord_from_par(par)
        if coord is None:
            continue
        wrms = try_pint_wrms(par, tim)
        if wrms is None:
            proxy = read_tim_error_proxy(tim)
            wrms = float(proxy.get("wrms_proxy_us", proxy.get("proxy_amplitude", np.nan)))
        if require_wrms and (not np.isfinite(wrms) or wrms <= 0):
            continue
        items.append(
            {
                "name": name,
                "par": str(par),
                "tim": str(tim),
                "ra_deg": float(coord.ra.deg),
                "dec_deg": float(coord.dec.deg),
                "wrms_us": float(wrms),
            }
        )
    items = [x for x in items if np.isfinite(x["wrms_us"]) and x["wrms_us"] > 0]
    items.sort(key=lambda x: x["wrms_us"], reverse=True)
    if max_pulsars and len(items) > max_pulsars:
        items = items[:max_pulsars]
    return items


def bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> Dict[str, object]:
    pear = []
    spear = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 3:
            continue
        pear.append(pearsonr(x[idx], y[idx])[0])
        spear.append(spearmanr(x[idx], y[idx]).correlation)
    return {
        "pearson_boot_mean": float(np.nanmean(pear)) if pear else float("nan"),
        "pearson_boot_ci95": [float(np.nanpercentile(pear, 2.5)), float(np.nanpercentile(pear, 97.5))] if pear else [float("nan"), float("nan")],
        "spearman_boot_mean": float(np.nanmean(spear)) if spear else float("nan"),
        "spearman_boot_ci95": [float(np.nanpercentile(spear, 2.5)), float(np.nanpercentile(spear, 97.5))] if spear else [float("nan"), float("nan")],
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="out_test14_cl2_p8c_pta_planck_pr4_kappa")
    ap.add_argument("--cache-dir", default="test14_cache")
    ap.add_argument("--max-pulsars", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--map-field", type=int, default=0)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    outdir = ensure_dir(args.outdir)
    cache_dir = ensure_dir(args.cache_dir)
    rng = np.random.default_rng(args.seed)

    log("[data] downloading NANOGrav 15-year archive")
    ng_root = download_nanograv(cache_dir)
    log("[data] downloading Planck PR4 maps archive")
    pr4_root = download_planck_pr4(cache_dir)

    map_path, mask_path = find_planck_pr4_files(pr4_root)
    pulsars = collect_pulsars(ng_root, max_pulsars=args.max_pulsars, require_wrms=False)
    if len(pulsars) < 5:
        raise RuntimeError("Too few pulsars with usable coordinates and timing data")

    ra = np.array([p["ra_deg"] for p in pulsars], dtype=float)
    dec = np.array([p["dec_deg"] for p in pulsars], dtype=float)
    wrms = np.array([p["wrms_us"] for p in pulsars], dtype=float)
    kappa = sample_healpix_map_value(map_path, ra, dec, field=args.map_field)
    if mask_path is not None:
        try:
            mask = sample_healpix_map_value(mask_path, ra, dec, field=0)
            good = np.isfinite(mask) & (mask > 0)
        except Exception:
            good = np.ones_like(kappa, dtype=bool)
    else:
        good = np.ones_like(kappa, dtype=bool)
    good &= np.isfinite(wrms) & np.isfinite(kappa)

    wrms = wrms[good]
    kappa = kappa[good]
    kept = [p for p, g in zip(pulsars, good) if g]
    if len(kept) < 5:
        raise RuntimeError("Too few pulsars survive map/mask sampling")

    pear_r, pear_p = pearsonr(wrms, kappa)
    spear = spearmanr(wrms, kappa)
    boot = bootstrap_corr(wrms, kappa, n_boot=args.n_boot, rng=rng)

    result = {
        "test_name": "T14 / CL2 / P8c PTA × Planck PR4 kappa",
        "analysis_mode": "pulsar-position public-data proxy",
        "n_pulsars": int(len(kept)),
        "map_path": str(map_path),
        "mask_path": str(mask_path) if mask_path else None,
        "mean_wrms_us": float(np.mean(wrms)),
        "mean_kappa": float(np.mean(kappa)),
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_r": float(spear.correlation),
        "spearman_p": float(spear.pvalue),
        **boot,
        "same_sign_as_p30": bool(pear_r > 0),
        "falsification_logic": {
            "confirm_like": "Positive PTA–kappa correlation with bootstrap support, same sign as P30.",
            "falsify_like": "Null or wrong-sign correlation.",
        },
        "pulsars": kept,
    }
    save_json(result, outdir / "result.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
