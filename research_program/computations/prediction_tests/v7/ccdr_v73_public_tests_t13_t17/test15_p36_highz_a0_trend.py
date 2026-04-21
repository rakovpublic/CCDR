#!/usr/bin/env python3
"""
T15 / P36 — High-z a0 trend extension.

Public-data screening implementation:
- SPARC local anchor from public rotation-curve component files.
- KMOS3D high-z proxy from the public release tarball.

Honesty note:
This is a screening proxy, not a collaboration-grade resolved-curve reanalysis.
The public KMOS3D release tables are heterogeneous. This script now supports:
1) structural/kinematic proxy when stellar-mass and radius-like columns are present
2) sigma/aperture fallback when the released table only exposes redshift, velocity
   dispersion, and aperture radius (which is exactly what appeared in the user's log)
"""
from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.constants import G
from astropy.cosmology import Planck18
from astropy.io import fits
from scipy.stats import spearmanr

from common_public_utils import (
    MPS2_PER_KMS2_PER_KPC,
    a0_from_gobs_gbar,
    discover_kmos3d_catalog_url,
    download_file,
    download_first_available,
    ensure_dir,
    kmos_guess_columns,
    log,
    save_json,
    to_numeric_safe,
)

G_SI = float(G.value)
KPC_M = 3.085677581491367e19
MSUN_KG = 1.988409870698051e30
C_H0_MPS2 = 6.6e-10
VERSION = "T15 v11"

SPARC_URLS = [
    "https://astroweb.case.edu/SPARC/Rotmod_LTG.zip",
    "http://astroweb.case.edu/SPARC/Rotmod_LTG.zip",
    "https://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip",
    "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip",
]


def _to_native(arr_like):
    arr = np.asarray(arr_like)
    if getattr(arr.dtype, "kind", "O") == "O":
        return arr
    if getattr(arr.dtype, "byteorder", "=") not in ("=", "|"):
        arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
    return arr


def fits_table_to_dataframe_native(data) -> pd.DataFrame:
    names = list(getattr(getattr(data, "dtype", None), "names", []) or [])
    if not names:
        raise RuntimeError("FITS table has no named columns")
    out: Dict[str, object] = {}
    for name in names:
        col = _to_native(data[name])
        if np.asarray(col).ndim == 1 and getattr(np.asarray(col).dtype, "kind", "O") != "O":
            out[name] = pd.Series(np.asarray(col))
        elif np.asarray(col).ndim == 1:
            out[name] = pd.Series(np.asarray(col, dtype=object))
        else:
            out[name] = [np.asarray(v).tolist() for v in np.asarray(col, dtype=object)]
    return pd.DataFrame(out)


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
    return pd.DataFrame(arr, columns=[f"c{i}" for i in range(arr.shape[1])])


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
    a0 = a0[len(a0) // 2 :]
    a0 = a0[np.isfinite(a0) & (a0 > 0)]
    return float(np.nanmedian(a0)) if len(a0) else float("nan")


def download_sparc(cache_dir: Path) -> Path:
    archive = cache_dir / "Rotmod_LTG.zip"
    download_first_available(SPARC_URLS, archive)
    root = cache_dir / "sparc_rotmod"
    if not root.exists() or not any(root.iterdir()):
        ensure_dir(root)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
    return root


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
    return download_file(url, archive)


def safe_extract_tar(archive: Path, root: Path) -> None:
    ensure_dir(root)
    with tarfile.open(archive, "r:*") as tf:
        members = []
        root_resolved = root.resolve()
        for m in tf.getmembers():
            target = (root / m.name).resolve()
            if root_resolved not in target.parents and target != root_resolved:
                raise RuntimeError(f"Unsafe archive member path: {m.name}")
            members.append(m)
        try:
            tf.extractall(root, members=members, filter="data")
        except TypeError:
            tf.extractall(root, members=members)


def extract_kmos_catalog(archive: Path) -> Path:
    root = archive.parent / "kmos3d_catalog_extracted"
    marker = root / ".done"
    if marker.exists() and any(root.rglob("*.fits")):
        return root
    if root.exists():
        for p in sorted(root.rglob("*"), reverse=True):
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            except Exception:
                pass
    ensure_dir(root)
    safe_extract_tar(archive, root)
    marker.write_text("ok", encoding="utf-8")
    return root


def _score_cols(cols: List[str]) -> int:
    u = {c.upper() for c in cols}
    score = 0
    for k in ["Z", "FIELD", "ID"]:
        if k in u:
            score += 5
    for k in ["LMSTAR", "MSTAR", "LOGMSTAR", "MASS"]:
        if k in u:
            score += 5
    for k in ["RHALF", "RE", "RE_KPC", "R_E", "RADIUS", "AP_RADIUS"]:
        if k in u:
            score += 4
    for k in ["VROT", "VCIRC", "VMAX", "SIG", "SIGMA"]:
        if k in u:
            score += 3
    for k in ["FLUX_HA", "SFR"]:
        if k in u:
            score += 1
    return score


def load_kmos_catalog(archive: Path) -> pd.DataFrame:
    root = extract_kmos_catalog(archive)
    fits_paths = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".fits", ".fit"}])
    if not fits_paths:
        raise RuntimeError(f"No extracted FITS files found under {root}")
    best_df = None
    best_score = -1
    best_src = None
    best_cols = None
    errors: List[str] = []
    for fp in fits_paths:
        try:
            with fits.open(fp, memmap=False, ignore_missing_end=True) as hdul:
                for i, hdu in enumerate(hdul):
                    data = getattr(hdu, "data", None)
                    cols = list(getattr(getattr(data, "dtype", None), "names", []) or [])
                    if not cols:
                        continue
                    score = _score_cols(cols)
                    if score < best_score:
                        continue
                    try:
                        df = fits_table_to_dataframe_native(data)
                    except Exception as e:
                        errors.append(f"{fp.name}[HDU {i}] dataframe conversion failed: {e}")
                        continue
                    best_df = df
                    best_score = score
                    best_src = f"{fp.name}[HDU {i}]"
                    best_cols = cols
        except Exception as e:
            errors.append(f"{fp.name} open failed: {e}")
    if best_df is None:
        raise RuntimeError("Could not load any KMOS FITS table. Last errors: " + " | ".join(errors[-5:]))
    best_df.attrs["kmos_table_source"] = best_src
    best_df.attrs["kmos_table_score"] = best_score
    log(f"[{VERSION}] selected KMOS table source={best_src} score={best_score}")
    log(f"[{VERSION}] selected KMOS columns preview={list(best_cols[:14])}")
    return best_df


def infer_mass_msun(series: pd.Series) -> pd.Series:
    s = to_numeric_safe(series)
    arr = s.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) and np.nanmedian(finite) < 20:
        return 10.0 ** s
    return s


def find_col(df: pd.DataFrame, *tokens: str) -> Optional[str]:
    names = {str(c).lower(): c for c in df.columns}
    for token in tokens:
        tl = token.lower()
        for lc, c in names.items():
            if tl == lc:
                return c
        for lc, c in names.items():
            if tl in lc:
                return c
    return None


def choose_kmos_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    # Exact science-column matches first; only then fall back to fuzzy guesses.
    names = {str(c).upper(): c for c in df.columns}
    cols = dict(kmos_guess_columns(df))
    if 'Z' in names:
        cols['z'] = names['Z']
    elif 'REDSHIFT' in names:
        cols['z'] = names['REDSHIFT']
    if 'FIELD' in names:
        cols['field'] = names['FIELD']
    if 'ID' in names:
        cols['id'] = names['ID']
    if 'SIG' in names:
        cols['sigma'] = names['SIG']
    elif 'SIGMA' in names:
        cols['sigma'] = names['SIGMA']
    if 'AP_RADIUS' in names:
        cols['re_arcsec'] = names['AP_RADIUS']
    elif 'RHALF' in names:
        cols['re_arcsec'] = names['RHALF']
    elif 'RE' in names:
        cols['re_arcsec'] = names['RE']
    for key in ('re_kpc','vrot','mstar','mgas'):
        cols.setdefault(key, None)
    return cols


def kpc_per_arcsec_from_z(z: pd.Series) -> pd.Series:
    z = pd.to_numeric(z, errors="coerce").astype(float)
    vals = np.full(len(z), np.nan, dtype=float)
    arr = z.to_numpy(dtype=float)
    ok = np.isfinite(arr) & (arr > 0)
    if np.any(ok):
        da_kpc = Planck18.angular_diameter_distance(arr[ok]).to_value("kpc")
        vals[ok] = da_kpc * (np.pi / 648000.0)
    return pd.Series(vals, index=z.index)


def highz_proxy_from_kmos(df: pd.DataFrame, max_rows: int) -> Dict[str, object]:
    cols = choose_kmos_columns(df)
    cols.setdefault("sigma", find_col(df, "sigma", "sig"))
    cols.setdefault("re_arcsec", find_col(df, "rhalf", "ap_radius", "radius", "re_arc", "reff_arc"))

    zc = cols.get("z") or find_col(df, "z")
    if not zc:
        raise RuntimeError(
            f"KMOS3D catalog missing redshift column; table_source={df.attrs.get('kmos_table_source')}; "
            f"columns_preview={list(df.columns[:40])}"
        )

    z = to_numeric_safe(df[zc])
    rc_kpc = cols.get("re_kpc")
    rc_arcsec = cols.get("re_arcsec")
    vc = cols.get("vrot")
    sc = cols.get("sigma")
    mc = cols.get("mstar")
    mgc = cols.get("mgas")

    re_kpc = None
    radius_source = None
    if rc_kpc:
        re_kpc = to_numeric_safe(df[rc_kpc])
        radius_source = rc_kpc
    elif rc_arcsec:
        re_arcsec = to_numeric_safe(df[rc_arcsec])
        re_kpc = re_arcsec * kpc_per_arcsec_from_z(z)
        radius_source = f"{rc_arcsec}->kpc"

    # Preferred structural/kinematic mode when baryonic proxies exist.
    if re_kpc is not None and mc:
        work = pd.DataFrame({
            "z": z,
            "re_kpc": re_kpc,
            "mstar": infer_mass_msun(df[mc]),
        })
        work["mgas"] = infer_mass_msun(df[mgc]) if mgc else np.nan
        work["vrot"] = to_numeric_safe(df[vc]) if vc else np.nan
        work["sigma"] = to_numeric_safe(df[sc]) if sc else np.nan
        work["mbar_msun"] = work["mstar"].fillna(0.0) + work["mgas"].fillna(0.0)
        work.loc[work["mbar_msun"] <= 0, "mbar_msun"] = work.loc[work["mbar_msun"] <= 0, "mstar"]
        mask = np.isfinite(work["z"].to_numpy(dtype=float)) & np.isfinite(work["re_kpc"].to_numpy(dtype=float)) & np.isfinite(work["mbar_msun"].to_numpy(dtype=float))
        mask &= (work["z"].to_numpy(dtype=float) > 0.1) & (work["re_kpc"].to_numpy(dtype=float) > 0) & (work["mbar_msun"].to_numpy(dtype=float) > 0)
        work = work.iloc[np.flatnonzero(mask)].copy()
        if max_rows and len(work) > max_rows:
            work = work.sort_values("z").iloc[:max_rows].copy()
        if len(work):
            r_m = work["re_kpc"].to_numpy(dtype=float) * KPC_M
            mb_kg = work["mbar_msun"].to_numpy(dtype=float) * MSUN_KG
            gbar = G_SI * mb_kg / np.clip(r_m, 1e-9, None) ** 2
            analysis_mode = "structural_acceleration_proxy"
            if vc and np.isfinite(work["vrot"]).sum() >= max(8, int(0.1 * len(work))):
                kin = work[np.isfinite(work["vrot"]) & (work["vrot"] > 0)].copy()
                if len(kin):
                    r_m = kin["re_kpc"].to_numpy(dtype=float) * KPC_M
                    mb_kg = kin["mbar_msun"].to_numpy(dtype=float) * MSUN_KG
                    v_mps = kin["vrot"].to_numpy(dtype=float) * 1000.0
                    gobs = (v_mps ** 2) / np.clip(r_m, 1e-9, None)
                    gbar_kin = G_SI * mb_kg / np.clip(r_m, 1e-9, None) ** 2
                    kin["a0_proxy_m_s2"] = a0_from_gobs_gbar(gobs, gbar_kin)
                    kin = kin[np.isfinite(kin["a0_proxy_m_s2"]) & (kin["a0_proxy_m_s2"] > 0)].copy()
                    if len(kin):
                        work = kin
                        analysis_mode = "kinematic_catalog_proxy"
                    else:
                        work["a0_proxy_m_s2"] = gbar
                else:
                    work["a0_proxy_m_s2"] = gbar
            else:
                work["a0_proxy_m_s2"] = gbar
            rho = spearmanr(work["z"], work["a0_proxy_m_s2"])
            return {
                "n_systems": int(len(work)),
                "analysis_mode": analysis_mode,
                "column_guess": cols,
                "radius_source": radius_source,
                "table_source": df.attrs.get("kmos_table_source"),
                "table_score": df.attrs.get("kmos_table_score"),
                "z_values": work["z"].tolist(),
                "a0_proxy_values_m_s2": work["a0_proxy_m_s2"].tolist(),
                "highz_mean_a0_m_s2": float(np.nanmean(work["a0_proxy_m_s2"])),
                "highz_median_a0_m_s2": float(np.nanmedian(work["a0_proxy_m_s2"])),
                "spearman_r": float(rho.correlation),
                "spearman_p": float(rho.pvalue),
                "data_preview": work.head(20).to_dict(orient="records"),
            }

    # Fallback for the actual public H-alpha/sigma table seen in the user's logs.
    if sc and re_kpc is not None:
        work = pd.DataFrame({
            "z": z,
            "re_kpc": re_kpc,
            "sigma_kms": to_numeric_safe(df[sc]),
        })
        mask = np.isfinite(work["z"].to_numpy(dtype=float)) & np.isfinite(work["re_kpc"].to_numpy(dtype=float)) & np.isfinite(work["sigma_kms"].to_numpy(dtype=float))
        mask &= (work["z"].to_numpy(dtype=float) > 0.1) & (work["re_kpc"].to_numpy(dtype=float) > 0) & (work["sigma_kms"].to_numpy(dtype=float) > 0)
        work = work.iloc[np.flatnonzero(mask)].copy()
        if max_rows and len(work) > max_rows:
            work = work.sort_values("z").iloc[:max_rows].copy()
        if len(work) == 0:
            raise RuntimeError(
                f"No usable KMOS3D rows after sigma/aperture fallback; guessed={cols}; source={df.attrs.get('kmos_table_source')}"
            )
        r_m = work["re_kpc"].to_numpy(dtype=float) * KPC_M
        sigma_mps = work["sigma_kms"].to_numpy(dtype=float) * 1000.0
        work["a0_proxy_m_s2"] = (sigma_mps ** 2) / np.clip(r_m, 1e-9, None)
        rho = spearmanr(work["z"], work["a0_proxy_m_s2"])
        return {
            "n_systems": int(len(work)),
            "analysis_mode": "sigma_aperture_proxy",
            "column_guess": cols,
            "radius_source": radius_source,
            "table_source": df.attrs.get("kmos_table_source"),
            "table_score": df.attrs.get("kmos_table_score"),
            "z_values": work["z"].tolist(),
            "a0_proxy_values_m_s2": work["a0_proxy_m_s2"].tolist(),
            "highz_mean_a0_m_s2": float(np.nanmean(work["a0_proxy_m_s2"])),
            "highz_median_a0_m_s2": float(np.nanmedian(work["a0_proxy_m_s2"])),
            "spearman_r": float(rho.correlation),
            "spearman_p": float(rho.pvalue),
            "data_preview": work.head(20).to_dict(orient="records"),
        }

    raise RuntimeError(
        f"KMOS3D catalog missing essential usable columns after extraction; table_source={df.attrs.get('kmos_table_source')}; "
        f"columns_preview={list(df.columns[:40])}"
    )


def derive_closed_gap_and_nu(local_a0: float, highz_a0: float, analysis_mode: str) -> Dict[str, float]:
    gap = C_H0_MPS2 - local_a0
    closed = (highz_a0 - local_a0) / gap if gap > 0 else float("nan")
    nu_cal = 5.08e-3 * (closed / 0.453) if analysis_mode == "kinematic_catalog_proxy" and np.isfinite(closed) else float("nan")
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

    log(f"[{VERSION}] starting high-z a0 screening")
    log("[data] downloading SPARC")
    sparc_root = download_sparc(cache_dir)
    log("[data] downloading KMOS3D catalog")
    kmos_archive = download_kmos_catalog(cache_dir)
    log(f"[{VERSION}] loading extracted KMOS3D catalog")
    kmos_df = load_kmos_catalog(kmos_archive)

    local = local_anchor_sparc(sparc_root, max_galaxies=args.max_sparc_galaxies)
    highz = highz_proxy_from_kmos(kmos_df, max_rows=args.max_kmos_systems)
    calib = derive_closed_gap_and_nu(
        local["a0_local_median_m_s2"],
        highz["highz_mean_a0_m_s2"],
        highz["analysis_mode"],
    )

    result = {
        "test_name": "T15 / P36 high-z a0 trend",
        "analysis_mode": "SPARC local anchor + KMOS3D public-release proxy",
        "local_anchor": local,
        "highz_proxy": highz,
        **calib,
        "reference_cH0_m_s2": C_H0_MPS2,
        "falsification_logic": {
            "confirm_like": "a0 proxy increases with z and the high-z mean lies above the local anchor.",
            "falsify_like": "No upward trend appears, or the high-z proxy lies at or below the local anchor.",
        },
        "notes": [
            "SPARC uses actual component curves from public Rotmod_LTG files.",
            "KMOS3D uses a public-release proxy; sigma_aperture_proxy is fallback mode when the release table lacks stellar-mass and resolved radius columns.",
            "nu_mond_calibrated is only reported for the kinematic_catalog_proxy mode.",
        ],
    }
    save_json(result, outdir / "result.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
