#!/usr/bin/env python3
"""
T15 / P36 — High-z a0 trend extension.

This script downloads public SPARC rotation-curve data and the public KMOS3D
final release catalog, computes a local a0 anchor from SPARC and a high-z a0
proxy from KMOS3D integrated quantities, then measures the trend of a0 with z.

Important honesty note:
- The SPARC part uses actual rotation-curve component files.
- The KMOS3D public release is not a uniform ready-made rotation-curve table in
  a single canonical format, so this script uses an integrated-quantity proxy
  from the released catalog when full resolved curves are not available in a
  directly parseable form. That makes this a screening implementation, not a
  final collaboration-grade measurement.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.constants import G
from scipy.stats import spearmanr

from common_public_utils import (
    MPS2_PER_KMS2_PER_KPC,
    a0_from_gobs_gbar,
    discover_kmos3d_catalog_url,
    download_file,
    download_first_available,
    ensure_dir,
    extract_archive,
    find_files,
    load_fits_table,
    log,
    save_json,
    table_to_dataframe,
    to_numeric_safe,
    kmos_guess_columns,
)


G_SI = float(G.value)
KPC_M = 3.085677581491367e19
MSUN_KG = 1.988409870698051e30
C_H0_MPS2 = 6.6e-10  # approximate cH0 scale used only for closed-gap framing


SPARC_URLS = [
    "https://astroweb.case.edu/SPARC/Rotmod_LTG.zip",
    "http://astroweb.case.edu/SPARC/Rotmod_LTG.zip",
    "https://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip",
    "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip",
]


def download_sparc(cache_dir: Path) -> Path:
    archive = cache_dir / "Rotmod_LTG.zip"
    download_first_available(SPARC_URLS, archive)
    root = cache_dir / "sparc_rotmod"
    if not root.exists() or not any(root.iterdir()):
        ensure_dir(root)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
    return root


def parse_sparc_file(path: Path) -> Optional[pd.DataFrame]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            vals = []
            ok = True
            for p in parts:
                try:
                    vals.append(float(p))
                except Exception:
                    ok = False
                    break
            if ok and len(vals) >= 4:
                rows.append(vals)
    if len(rows) < 5:
        return None
    arr = np.array(rows, dtype=float)
    # Common SPARC/Rotmod files have at least: R, Vobs, eVobs, Vgas, Vdisk, Vbul, ...
    cols = [f"c{i}" for i in range(arr.shape[1])]
    df = pd.DataFrame(arr, columns=cols)
    return df


def a0_from_sparc_df(df: pd.DataFrame, ml_disk: float = 0.5, ml_bulge: float = 0.7) -> float:
    if len(df.columns) < 6:
        return float("nan")
    r_kpc = df.iloc[:, 0].to_numpy(dtype=float)
    vobs = df.iloc[:, 1].to_numpy(dtype=float)
    vgas = df.iloc[:, 3].to_numpy(dtype=float) if len(df.columns) > 3 else np.zeros_like(vobs)
    vdisk = df.iloc[:, 4].to_numpy(dtype=float) if len(df.columns) > 4 else np.zeros_like(vobs)
    vbul = df.iloc[:, 5].to_numpy(dtype=float) if len(df.columns) > 5 else np.zeros_like(vobs)
    vbary2 = np.clip(vgas**2 + ml_disk * vdisk**2 + ml_bulge * vbul**2, 0.0, np.inf)
    gobs = (vobs**2) * MPS2_PER_KMS2_PER_KPC / np.clip(r_kpc, 1e-6, None)
    gbar = vbary2 * MPS2_PER_KMS2_PER_KPC / np.clip(r_kpc, 1e-6, None)
    a0 = a0_from_gobs_gbar(gobs, gbar)
    # use outer half of the curve to approximate flat regime
    n = len(a0)
    a0 = a0[n // 2 :]
    a0 = a0[np.isfinite(a0) & (a0 > 0)]
    if len(a0) == 0:
        return float("nan")
    return float(np.nanmedian(a0))


def local_anchor_sparc(root: Path, max_galaxies: int) -> Dict[str, object]:
    files = sorted([p for p in root.rglob("*.dat") if p.is_file()])
    vals = []
    used = []
    for path in files:
        df = parse_sparc_file(path)
        if df is None:
            continue
        a0 = a0_from_sparc_df(df)
        if np.isfinite(a0) and a0 > 0:
            vals.append(a0)
            used.append(path.name)
        if max_galaxies and len(vals) >= max_galaxies:
            break
    arr = np.array(vals, dtype=float)
    return {
        "n_galaxies": int(len(arr)),
        "a0_values_m_s2": arr.tolist(),
        "a0_local_median_m_s2": float(np.nanmedian(arr)) if len(arr) else float("nan"),
        "a0_local_mean_m_s2": float(np.nanmean(arr)) if len(arr) else float("nan"),
        "files_used": used,
    }


def download_kmos_catalog(cache_dir: Path) -> Path:
    url = discover_kmos3d_catalog_url()
    archive = cache_dir / Path(url).name
    archive = download_file(url, archive)
    lower = archive.name.lower()
    if lower.endswith(('.tgz', '.tar.gz', '.tar', '.zip')):
        root = cache_dir / 'kmos3d_catalog'
        extract_archive(archive, root)
        fits_files = find_files(root, [r'\.fits$', r'\.fit$'])
        if not fits_files:
            raise RuntimeError(f'No FITS catalog found after extracting {archive}')
        fits_files.sort(key=lambda p: ('table' not in p.name.lower() and 'catalog' not in p.name.lower(), len(str(p))))
        return fits_files[0]
    return archive


def load_kmos_catalog(path: Path) -> pd.DataFrame:
    table = load_fits_table(path, 1)
    return table_to_dataframe(table)


def highz_proxy_from_kmos(df: pd.DataFrame, max_rows: int) -> Dict[str, object]:
    cols = kmos_guess_columns(df)
    zc = cols.get("z")
    rc = cols.get("re_kpc")
    vc = cols.get("vrot")
    mc = cols.get("mstar")
    mgc = cols.get("mgas")
    if not all([zc, rc, vc, mc]):
        raise RuntimeError(f"KMOS3D catalog missing required columns; guessed={cols}")

    work = pd.DataFrame({
        "z": to_numeric_safe(df[zc]),
        "re_kpc": to_numeric_safe(df[rc]),
        "vrot": to_numeric_safe(df[vc]),
        "mstar": to_numeric_safe(df[mc]),
    })
    if mgc:
        work["mgas"] = to_numeric_safe(df[mgc])
    else:
        work["mgas"] = np.nan

    # interpret stellar/gas masses: if values look logarithmic, exponentiate.
    for col in ["mstar", "mgas"]:
        arr = work[col].to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite) and np.nanmedian(finite) < 20:
            work[col] = 10.0 ** work[col]

    work["mbar_msun"] = work["mstar"].fillna(0.0) + work["mgas"].fillna(0.0)
    work.loc[work["mbar_msun"] <= 0, "mbar_msun"] = work.loc[work["mbar_msun"] <= 0, "mstar"]

    mask = np.isfinite(work["z"]) & np.isfinite(work["re_kpc"]) & np.isfinite(work["vrot"]) & np.isfinite(work["mbar_msun"])
    mask &= (work["z"] > 0.1) & (work["re_kpc"] > 0) & (work["vrot"] > 0) & (work["mbar_msun"] > 0)
    work = work.loc[mask].copy()
    if max_rows and len(work) > max_rows:
        work = work.sort_values("z").iloc[:max_rows].copy()

    r_m = work["re_kpc"].to_numpy(dtype=float) * KPC_M
    v_mps = work["vrot"].to_numpy(dtype=float) * 1000.0
    mb_kg = work["mbar_msun"].to_numpy(dtype=float) * MSUN_KG

    gobs = (v_mps**2) / np.clip(r_m, 1e-9, None)
    gbar = G_SI * mb_kg / np.clip(r_m, 1e-9, None) ** 2
    a0 = a0_from_gobs_gbar(gobs, gbar)

    work["gobs_m_s2"] = gobs
    work["gbar_m_s2"] = gbar
    work["a0_proxy_m_s2"] = a0
    work = work[np.isfinite(work["a0_proxy_m_s2"]) & (work["a0_proxy_m_s2"] > 0)].copy()

    rho = spearmanr(work["z"], work["a0_proxy_m_s2"])
    return {
        "n_systems": int(len(work)),
        "z_values": work["z"].tolist(),
        "a0_proxy_values_m_s2": work["a0_proxy_m_s2"].tolist(),
        "highz_mean_a0_m_s2": float(np.nanmean(work["a0_proxy_m_s2"])),
        "highz_median_a0_m_s2": float(np.nanmedian(work["a0_proxy_m_s2"])),
        "spearman_r": float(rho.correlation),
        "spearman_p": float(rho.pvalue),
        "data_preview": work.head(20).to_dict(orient="records"),
    }


def derive_closed_gap_and_nu(local_a0: float, highz_a0: float) -> Dict[str, float]:
    gap = C_H0_MPS2 - local_a0
    closed = (highz_a0 - local_a0) / gap if gap > 0 else float("nan")
    # Transparent round-5 calibration: the framework reports closed_gap_fraction
    # 0.453 corresponding to nu_MOND = 5.08e-3. We expose that calibration rather
    # than pretending to have the full first-principles extractor here.
    nu_cal = 5.08e-3 * (closed / 0.453) if np.isfinite(closed) else float("nan")
    return {
        "closed_gap_fraction": float(closed),
        "nu_mond_calibrated": float(nu_cal),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="out_test15_p36_highz_a0_trend")
    ap.add_argument("--cache-dir", default="test15_cache")
    ap.add_argument("--max-sparc-galaxies", type=int, default=25)
    ap.add_argument("--max-kmos-systems", type=int, default=200)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    outdir = ensure_dir(args.outdir)
    cache_dir = ensure_dir(args.cache_dir)

    log("[data] downloading SPARC")
    sparc_root = download_sparc(cache_dir)
    log("[data] downloading KMOS3D catalog")
    kmos_path = download_kmos_catalog(cache_dir)
    kmos_df = load_kmos_catalog(kmos_path)

    local = local_anchor_sparc(sparc_root, max_galaxies=args.max_sparc_galaxies)
    highz = highz_proxy_from_kmos(kmos_df, max_rows=args.max_kmos_systems)
    calib = derive_closed_gap_and_nu(local["a0_local_median_m_s2"], highz["highz_mean_a0_m_s2"])

    result = {
        "test_name": "T15 / P36 high-z a0 trend",
        "analysis_mode": "SPARC local anchor + KMOS3D integrated-quantity proxy",
        "local_anchor": local,
        "highz_proxy": highz,
        **calib,
        "reference_cH0_m_s2": C_H0_MPS2,
        "falsification_logic": {
            "confirm_like": "a0 increases with z and the high-z mean lies above the local anchor, with a calibrated nu_MOND in the 1e-3 to 1e-2 band.",
            "falsify_like": "No upward trend, or high-z a0 is comparable to/below the local anchor.",
        },
        "notes": [
            "SPARC uses actual component curves from public Rotmod_LTG files.",
            "KMOS3D uses an integrated-quantity proxy when fully resolved public curves are not trivially recoverable from the release format.",
            "nu_mond_calibrated is explicitly calibrated to the framework's own round-5 closed-gap reference, not claimed as a first-principles RVM fit.",
        ],
    }
    save_json(result, outdir / "result.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
