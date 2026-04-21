#!/usr/bin/env python3
"""Shared helpers for CCDR v7.3 / Synthesis v3.3 public-data screening tests.

These helpers favor:
- reproducible public-data downloads
- small, query-based access where possible
- graceful degradation when a public mirror is temporarily unavailable
- transparent JSON outputs rather than hidden state

The tests in this bundle are *screening/proxy* scripts. They are designed to be
runnable from public sources without private collaboration products.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from scipy import optimize, spatial, stats


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
OUT_DIR = ROOT / "outputs"
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (compatible; CCDR-v7.3-public-tests/1.0; +https://openai.com)"
)
TIMEOUT = 120


def ensure_package(package: str, import_name: str | None = None) -> None:
    """Best-effort runtime installer for optional dependencies.

    Used only for packages that make public-data access much easier, mainly
    astropy-related FITS/WCS helpers.
    """
    import importlib.util

    name = import_name or package
    if importlib.util.find_spec(name) is not None:
        return
    print(f"[deps] Installing optional package: {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


class DownloadError(RuntimeError):
    pass


class DataUnavailable(RuntimeError):
    pass


@dataclass
class DownloadCandidate:
    name: str
    url: str


@dataclass
class FitResult:
    params: dict[str, float]
    chi2: float
    ndof: int


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def write_json(path: os.PathLike[str] | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def robust_spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": int(m.sum())}
    rho, p = stats.spearmanr(x[m], y[m])
    return {"rho": float(rho), "pvalue": float(p), "n": int(m.sum())}


def robust_pearson(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return {"r": float("nan"), "pvalue": float("nan"), "n": int(m.sum())}
    r, p = stats.pearsonr(x[m], y[m])
    return {"r": float(r), "pvalue": float(p), "n": int(m.sum())}


def sigma_from_p_two_sided(p: float) -> float:
    if not np.isfinite(p) or p <= 0 or p >= 1:
        return float("nan")
    return float(stats.norm.isf(p / 2.0))


def bootstrap_statistic(
    values: Sequence[float], func: Callable[[np.ndarray], float], n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    est = float(func(arr))
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(arr, size=arr.size, replace=True)
        boots.append(float(func(samp)))
    lo, hi = np.percentile(boots, [16, 84])
    return {"value": est, "lo": float(lo), "hi": float(hi), "n": int(arr.size)}


def request_text(url: str, timeout: int = TIMEOUT) -> str:
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def discover_hrefs(page_url: str, patterns: Sequence[str]) -> list[str]:
    html = request_text(page_url)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    out: list[str] = []
    for href in hrefs:
        full = urljoin(page_url, href)
        if any(re.search(pat, full, flags=re.IGNORECASE) for pat in patterns):
            out.append(full)
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def download_file(
    url: str,
    dest: os.PathLike[str] | str,
    *,
    timeout: int = TIMEOUT,
    overwrite: bool = False,
    chunk_size: int = 1 << 20,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite and dest.stat().st_size > 0:
        return dest
    with SESSION.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
    return dest


def download_first_available(candidates: Sequence[DownloadCandidate], dest_dir: os.PathLike[str] | str) -> Path:
    dest_dir = Path(dest_dir)
    errors: list[str] = []
    for cand in candidates:
        try:
            filename = sanitize_filename(Path(cand.url.split("?")[0]).name or cand.name)
            return download_file(cand.url, dest_dir / filename)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{cand.name}: {exc}")
    raise DownloadError("All download candidates failed:\n" + "\n".join(errors))


# ---------- Public-source manifests ----------


def pantheon_plus_paths() -> dict[str, str]:
    base = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR"
    return {
        "data": f"{base}/Pantheon%2BSH0ES.dat",
        "cov_stat_sys": f"{base}/Pantheon%2BSH0ES_STAT%2BSYS.cov",
        "cov_stat": f"{base}/Pantheon%2BSH0ES_STATONLY.cov",
    }


def desi_bao_paths() -> dict[str, str]:
    base = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master"
    return {
        "consensus": f"{base}/final_consensus_covtot_dM_Hz_fsig.txt",
        "lrg_0p6_0p8": f"{base}/desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt",
        "lrg_elg_0p8_1p1": f"{base}/desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_mean.txt",
        "elg_1p1_1p6": f"{base}/desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt",
        "qso_0p8_2p1": f"{base}/desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_mean.txt",
        "lya": f"{base}/desi_2024_gaussian_bao_Lya_GCcomb_mean.txt",
        "all_gccomb": f"{base}/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",
        "bgs_0p1_0p4": f"{base}/desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_mean.txt",
    }


def sparc_paths() -> dict[str, str]:
    base = "https://zenodo.org/records/16284118/files"
    return {
        "rotmod": f"{base}/Rotmod_LTG.zip?download=1",
        "database": f"{base}/sparc_database.zip?download=1",
    }


def nanograv_path() -> str:
    return "https://zenodo.org/records/16051178/files/NANOGrav15yr_PulsarTiming_v2.1.0.tar.gz?download=1"


def act_dr6_path() -> str:
    return "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/dr6_lensing_release.tar.gz"


def planck_pr4_candidates() -> list[DownloadCandidate]:
    return [
        DownloadCandidate(
            "github_release_data",
            "https://github.com/carronj/planck_PR4_lensing/releases/download/Data/PR42018like_maps.tar",
        ),
        DownloadCandidate(
            "github_release_lensing_maps_and_chains",
            "https://github.com/carronj/planck_PR4_lensing/releases/download/Lensing-maps-and-chains/PR42018like_maps.tar",
        ),
    ]


def kmos3d_candidates() -> list[DownloadCandidate]:
    return [
        DownloadCandidate(
            "kmos3d_catalog_page",
            "https://www.mpe.mpg.de/ir/KMOS3D/data",
        ),
        DownloadCandidate(
            "kmos3d_catalog_direct",
            "https://www.mpe.mpg.de/resources/KMOS3D/3d_fnlsp_table_v3.fits.tgz",
        ),
        DownloadCandidate(
            "kmos3d_catalog_direct_no_www",
            "https://mpe.mpg.de/resources/KMOS3D/3d_fnlsp_table_v3.fits.tgz",
        ),
    ]


def xnt_wimp_paths() -> dict[str, str]:
    base = "https://raw.githubusercontent.com/XENONnT/wimp_data_release/master"
    return {
        "xenonnt_2025_si_wimp": f"{base}/SR0%2BSR1/xenonnt_2025_si_wimp.csv",
        "xenonnt_2025_sd_neutron_wimp": f"{base}/SR0%2BSR1/xenonnt_2025_sd_neutron_wimp.csv",
        "xenonnt_2025_sd_proton_wimp": f"{base}/SR0%2BSR1/xenonnt_2025_sd_proton_wimp.csv",
    }


def pandax_paths() -> dict[str, str]:
    base = "https://static.pandax.sjtu.edu.cn/download/data-share/p4-first-analysis"
    return {
        "data_xlsx": f"{base}/PandaX4T_Data_ne.xlsx",
        "eff_root": f"{base}/eff_RDQ_graph.root",
    }


def lz_hepdata_paths() -> dict[str, str]:
    # Public HEPData table downloads.
    # CSV exports are plain text and easy to parse. These are enough for T11.
    return {
        "lz_si": "https://www.hepdata.net/record/resource/2683193?format=csv&download=1",
        "lz_sdn": "https://www.hepdata.net/record/resource/2683195?format=csv&download=1",
        "lz_sdp": "https://www.hepdata.net/record/resource/2683194?format=csv&download=1",
    }


# ---------- Light-weight cosmology helpers ----------

C_LIGHT_KM_S = 299792.458
H0_DEFAULT = 70.0
OMEGA_M_DEFAULT = 0.3
OMEGA_L_DEFAULT = 0.7


def e_z(z: np.ndarray | float, om: float = OMEGA_M_DEFAULT, ol: float = OMEGA_L_DEFAULT) -> np.ndarray | float:
    return np.sqrt(om * (1.0 + np.asarray(z)) ** 3 + ol)


def comoving_distance_mpc(z: Sequence[float] | np.ndarray, h0: float = H0_DEFAULT) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    grid = np.linspace(0.0, max(1e-6, float(np.nanmax(z))), 4096)
    integ = np.cumsum(np.r_[0.0, 0.5 * (1.0 / e_z(grid[1:]) + 1.0 / e_z(grid[:-1])) * np.diff(grid)])
    return np.interp(z, grid, integ) * (C_LIGHT_KM_S / h0)


def angle_separation_deg(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    ra1r = np.deg2rad(ra1)
    ra2r = np.deg2rad(ra2)
    dec1r = np.deg2rad(dec1)
    dec2r = np.deg2rad(dec2)
    cosang = np.sin(dec1r) * np.sin(dec2r) + np.cos(dec1r) * np.cos(dec2r) * np.cos(ra1r - ra2r)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosang))


def unit_sphere_xyz(ra_deg: Sequence[float], dec_deg: Sequence[float]) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])


def local_density_proxy(ra_deg: Sequence[float], dec_deg: Sequence[float], k: int = 10) -> np.ndarray:
    xyz = unit_sphere_xyz(ra_deg, dec_deg)
    tree = spatial.cKDTree(xyz)
    d, _ = tree.query(xyz, k=min(k + 1, len(xyz)))
    # Use the kth neighbor chord distance as inverse-density proxy.
    r = np.clip(d[:, -1], 1e-8, None)
    return 1.0 / (r ** 2)


def density_proxy_at_targets(
    source_ra_deg: Sequence[float],
    source_dec_deg: Sequence[float],
    target_ra_deg: Sequence[float],
    target_dec_deg: Sequence[float],
    k: int = 64,
) -> np.ndarray:
    src = unit_sphere_xyz(source_ra_deg, source_dec_deg)
    tgt = unit_sphere_xyz(target_ra_deg, target_dec_deg)
    tree = spatial.cKDTree(src)
    kk = int(max(1, min(k, len(src))))
    d, _ = tree.query(tgt, k=kk)
    if kk == 1:
        d = np.asarray(d, dtype=float)
    else:
        d = np.asarray(d, dtype=float)[:, -1]
    r = np.clip(d, 1e-8, None)
    return 1.0 / (r ** 2)


def _circular_ra_window(ra_deg: np.ndarray) -> tuple[float, float] | None:
    ra = np.asarray(ra_deg, dtype=float)
    ra = ra[np.isfinite(ra)] % 360.0
    if ra.size < 2:
        return None
    ra_sorted = np.sort(ra)
    gaps = np.diff(np.r_[ra_sorted, ra_sorted[0] + 360.0])
    j = int(np.argmax(gaps))
    span = 360.0 - float(gaps[j])
    if span >= 300.0:
        return None
    lo = float(ra_sorted[(j + 1) % ra_sorted.size])
    hi = float(ra_sorted[j])
    if hi < lo:
        hi += 360.0
    return lo, hi


def sampler_query_window(sampler: Any, threshold: float | None = None, max_points: int = 50000) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Approximate RA/Dec window for querying catalogs inside a lensing footprint.

    For screening tests this window should be *loose*, not exact: use a relaxed
    mask threshold and generous padding so follow-up filtering with the sampler can
    recover enough public objects to proceed.
    """
    ensure_astropy_stack()
    from astropy_healpix import HEALPix
    from astropy import units as u

    src = sampler.mask_sampler if hasattr(sampler, 'mask_sampler') else sampler
    if getattr(src, 'mode', None) != 'healpix':
        return None, None
    data = np.ravel(np.asarray(getattr(src, 'data', []), dtype=float))
    if data.size == 0:
        return None, None
    # Use a looser threshold for query-window estimation than for the final
    # sampling itself; otherwise sparse ACT masks can yield windows that are too
    # tight and starve the public-catalog overlap.
    thr = float((0.3 if threshold is None else threshold))
    valid = np.isfinite(data) & (data >= thr)
    idx = np.flatnonzero(valid)
    if idx.size == 0 and threshold is None:
        # progressively relax only for building the query window
        for thr2 in (0.7, 0.5, 0.3, 0.2, 0.1):
            idx = np.flatnonzero(np.isfinite(data) & (data >= thr2))
            if idx.size:
                break
    if idx.size == 0:
        return None, None
    if idx.size > max_points:
        step = int(np.ceil(idx.size / max_points))
        idx = idx[::step]
    hp = HEALPix(nside=int(src.nside), order=str(src.order), frame='icrs')
    lon, lat = hp.healpix_to_lonlat(idx)
    ra = np.asarray(lon.to_value(u.deg), dtype=float) % 360.0
    dec = np.asarray(lat.to_value(u.deg), dtype=float)
    dec_pad = 5.0
    dec_range = (max(-90.0, float(np.nanmin(dec) - dec_pad)), min(90.0, float(np.nanmax(dec) + dec_pad)))
    ra_window = _circular_ra_window(ra)
    return ra_window, dec_range


def sample_euclid_overlap_with_sampler(
    sampler: Any,
    max_rows: int = 5000,
    zmin: float = 0.2,
    zmax: float = 1.5,
    seed: int = 0,
    oversample: int = 8,
) -> tuple[pd.DataFrame, np.ndarray]:
    # Try several increasingly loose catalog-query windows before giving up.
    windows = []
    for thr in (None, 0.5, 0.2):
        rw, dw = sampler_query_window(sampler, threshold=thr)
        windows.append((rw, dw))
    windows.append((None, None))
    attempts = [max(2, oversample), max(4, oversample * 2), max(8, oversample * 4)]
    best_gal = None
    best_kappa = None
    best_n = -1
    for ra_window, dec_window in windows:
        for mult in attempts:
            request_rows = int(min(max(max_rows * mult, 400), 3000))
            try:
                gal = load_euclid_q1_sample(
                    max_rows=request_rows,
                    zmin=zmin,
                    zmax=zmax,
                    seed=seed,
                    ra_range=ra_window,
                    dec_range=dec_window,
                )
            except Exception:
                continue
            kappa = np.asarray(sampler.sample(gal["ra"], gal["dec"]), dtype=float)
            good = np.isfinite(kappa)
            n_good = int(np.sum(good))
            if n_good > best_n:
                best_gal = gal.loc[good].reset_index(drop=True)
                best_kappa = kappa[good]
                best_n = n_good
            # Screening tests can proceed with a modest overlap sample.
            if n_good >= max(8, min(max_rows, 60)):
                gal = gal.loc[good].reset_index(drop=True)
                kappa = kappa[good]
                if len(gal) > max_rows:
                    take = gal.sample(max_rows, random_state=seed).index.to_numpy()
                    gal = gal.loc[take].reset_index(drop=True)
                    kappa = kappa[take]
                return gal.reset_index(drop=True), np.asarray(kappa, dtype=float)
    if best_n >= 5 and best_gal is not None and best_kappa is not None:
        gal = best_gal
        kappa = np.asarray(best_kappa, dtype=float)
        if len(gal) > max_rows:
            take = gal.sample(min(max_rows, len(gal)), random_state=seed).index.to_numpy()
            gal = gal.loc[take].reset_index(drop=True)
            kappa = kappa[take]
        return gal.reset_index(drop=True), kappa
    raise DataUnavailable("Too few Euclid galaxies overlap the requested lensing footprint")


def _random_sky_points_in_window(
    ra_window: tuple[float, float] | None,
    dec_window: tuple[float, float] | None,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if ra_window is None:
        ra = rng.uniform(0.0, 360.0, int(n))
    else:
        lo, hi = float(ra_window[0]), float(ra_window[1])
        if hi < lo:
            hi += 360.0
        ra = (lo + rng.uniform(0.0, hi - lo, int(n))) % 360.0
    if dec_window is None:
        s = rng.uniform(-1.0, 1.0, int(n))
        dec = np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))
    else:
        d0, d1 = float(dec_window[0]), float(dec_window[1])
        s0 = np.sin(np.deg2rad(d0))
        s1 = np.sin(np.deg2rad(d1))
        s = rng.uniform(min(s0, s1), max(s0, s1), int(n))
        dec = np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))
    return pd.DataFrame({'ra': np.asarray(ra, dtype=float), 'dec': np.asarray(dec, dtype=float)})


def sample_targets_from_sampler(
    sampler: Any,
    max_rows: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """Draw target sky positions robustly from the sampler footprint.

    Prefer random positions inside a loose footprint window and keep only those
    that return finite sampler values. This is much more robust for sparse ACT
    masks than assuming all mask-pixel centers are valid screening targets.
    """
    rng = np.random.default_rng(seed)
    windows = []
    for thr in (0.05, 0.02, None):
        windows.append(sampler_query_window(sampler, threshold=thr))
    windows.append((None, None))
    best = None
    best_n = -1
    need = max(8, min(int(max_rows), 80))
    for ra_window, dec_window in windows:
        for mult in (20, 50, 100):
            ntry = int(min(max(max_rows * mult, 2000), 120000))
            cand = _random_sky_points_in_window(ra_window, dec_window, ntry, rng)
            vals = np.asarray(sampler.sample(cand['ra'], cand['dec']), dtype=float)
            good = np.isfinite(vals)
            n_good = int(np.sum(good))
            if n_good > best_n:
                best = cand.loc[good].reset_index(drop=True)
                best_n = n_good
            if n_good >= need:
                out = cand.loc[good].reset_index(drop=True)
                if len(out) > max_rows:
                    take = out.sample(int(max_rows), random_state=seed).index.to_numpy()
                    out = out.loc[take].reset_index(drop=True)
                return out
    # Fallback to HEALPix pixel centers when random sampling fails.
    ensure_astropy_stack()
    from astropy_healpix import HEALPix
    from astropy import units as u
    src = sampler.mask_sampler if hasattr(sampler, 'mask_sampler') else sampler
    if getattr(src, 'mode', None) == 'healpix':
        data = np.ravel(np.asarray(getattr(src, 'data', []), dtype=float))
        valid = np.isfinite(data) & (data > 0)
        idx = np.flatnonzero(valid)
        if idx.size:
            take_n = int(min(max(max_rows * 10, 2000), idx.size))
            choose = rng.choice(idx, size=take_n, replace=(idx.size < take_n))
            hp = HEALPix(nside=int(src.nside), order=str(src.order), frame='icrs')
            lon, lat = hp.healpix_to_lonlat(choose)
            cand = pd.DataFrame({
                'ra': np.asarray(lon.to_value(u.deg), dtype=float) % 360.0,
                'dec': np.asarray(lat.to_value(u.deg), dtype=float),
            })
            vals = np.asarray(sampler.sample(cand['ra'], cand['dec']), dtype=float)
            good = np.isfinite(vals)
            if int(np.sum(good)) >= 5:
                out = cand.loc[good].reset_index(drop=True)
                if len(out) > max_rows:
                    take = out.sample(int(max_rows), random_state=seed).index.to_numpy()
                    out = out.loc[take].reset_index(drop=True)
                return out
    if best is not None and len(best) >= 5:
        out = best.reset_index(drop=True)
        if len(out) > max_rows:
            take = out.sample(min(int(max_rows), len(out)), random_state=seed).index.to_numpy()
            out = out.loc[take].reset_index(drop=True)
        return out
    raise DataUnavailable('No valid target positions found inside the requested lensing footprint')


def _assign_target_z_from_catalog(catalog: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    if len(catalog) == 0:
        return np.full(len(target), np.nan)
    src = unit_sphere_xyz(catalog['ra'], catalog['dec'])
    tgt = unit_sphere_xyz(target['ra'], target['dec'])
    tree = spatial.cKDTree(src)
    _, idx = tree.query(tgt, k=1)
    zsrc = np.asarray(catalog['z'], dtype=float) if 'z' in catalog.columns else np.full(len(catalog), np.nan)
    if zsrc.size == 0:
        return np.full(len(target), np.nan)
    return zsrc[np.asarray(idx, dtype=int)]


def query_public_density_catalog_for_sampler(
    sampler: Any,
    max_rows: int = 12000,
    zmin: float = 0.2,
    zmax: float = 1.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, str]:
    """Get a public galaxy catalog around the sampler footprint for density estimation.

    Prefer Euclid when available, but fall back to SDSS without requiring objects
    themselves to overlap the lensing footprint.
    """
    windows = []
    for thr in (None, 0.3, 0.1):
        windows.append(sampler_query_window(sampler, threshold=thr))
    windows.append((None, None))
    # Try Euclid first with modest request sizes.
    for ra_window, dec_window in windows:
        for rows in (min(max_rows, 2000), min(max_rows, 4000), min(max_rows, 8000)):
            try:
                gal = load_euclid_q1_sample(max_rows=max(400, rows), zmin=zmin, zmax=zmax, seed=seed, ra_range=ra_window, dec_range=dec_window)
                if len(gal) >= 100:
                    gal = gal[['ra','dec','z']].dropna().reset_index(drop=True)
                    return gal, 'Euclid Q1 IRSA TAP'
            except Exception:
                pass
    # Fall back to SDSS for robust screening coverage.
    for ra_window, dec_window in windows:
        for rows in (min(max_rows, 3000), min(max_rows, 8000), min(max_rows, 16000), min(max_rows, 30000)):
            try:
                gal = query_sdss_galaxies(max(300, rows), max(0.05, zmin), min(0.7, max(0.7, zmax)), cache_key=f'density_{rows}_{ra_window}_{dec_window}', ra_range=ra_window, dec_range=dec_window)
                if len(gal) >= 100:
                    gal = gal[['ra','dec','z']].dropna().reset_index(drop=True)
                    return gal, 'SDSS DR17 SkyServer SQL'
            except Exception:
                pass
    raise DataUnavailable('Could not load a sufficient public galaxy density catalog for the sampler footprint')


def _mask_based_target_positions(sampler: Any, max_rows: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Build target positions directly from the sampler mask footprint when available."""
    ensure_astropy_stack()
    from astropy_healpix import HEALPix
    from astropy import units as u

    rng = np.random.default_rng(seed)
    src = sampler.mask_sampler if hasattr(sampler, 'mask_sampler') else sampler
    if getattr(src, 'mode', None) != 'healpix':
        raise DataUnavailable('Mask-based target fallback requires a HEALPix sampler')
    data = np.ravel(np.asarray(getattr(src, 'data', []), dtype=float))
    if data.size == 0:
        raise DataUnavailable('Sampler mask footprint is empty')
    valid = np.isfinite(data) & (data > 0)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise DataUnavailable('Sampler mask footprint has no positive pixels')
    take_n = int(min(max(max_rows * 8, 2000), idx.size))
    choose = rng.choice(idx, size=take_n, replace=(idx.size < take_n))
    hp = HEALPix(nside=int(src.nside), order=str(src.order), frame='icrs')
    lon, lat = hp.healpix_to_lonlat(choose)
    return pd.DataFrame({
        'ra': np.asarray(lon.to_value(u.deg), dtype=float) % 360.0,
        'dec': np.asarray(lat.to_value(u.deg), dtype=float),
    })


def build_public_density_targets_with_sampler(
    sampler: Any,
    max_rows: int = 5000,
    zmin: float = 0.2,
    zmax: float = 1.5,
    seed: int = 0,
    density_k: int = 64,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Build screening targets from the footprint itself and evaluate public density there.

    This function is intentionally permissive: for screening tests we prefer a
    modest but usable footprint sample over aborting whenever the ACT overlap is sparse.
    """
    best_targets = None
    best_kappa = None
    best_n = -1
    target_note = None

    def _consider(candidate_targets: pd.DataFrame, *, allow_mask_proxy: bool = False, note: str | None = None):
        nonlocal best_targets, best_kappa, best_n, target_note
        if candidate_targets is None or len(candidate_targets) == 0:
            return False
        kappa_local = np.asarray(sampler.sample(candidate_targets['ra'], candidate_targets['dec']), dtype=float)
        good_local = np.isfinite(kappa_local)
        n_good_local = int(np.sum(good_local))
        if n_good_local > best_n:
            best_targets = candidate_targets.loc[good_local].reset_index(drop=True)
            best_kappa = kappa_local[good_local]
            best_n = n_good_local
            target_note = note
        if n_good_local >= max(4, min(int(max_rows), 20)):
            return candidate_targets.loc[good_local].reset_index(drop=True), kappa_local[good_local], note
        if allow_mask_proxy and hasattr(sampler, 'mask_sampler'):
            maskvals = np.asarray(sampler.mask_sampler.sample(candidate_targets['ra'], candidate_targets['dec']), dtype=float)
            mgood = np.isfinite(maskvals)
            n_mgood = int(np.sum(mgood))
            if n_mgood > best_n:
                best_targets = candidate_targets.loc[mgood].reset_index(drop=True)
                best_kappa = maskvals[mgood]
                best_n = n_mgood
                target_note = (note + ' [ACT-mask fallback]') if note else 'ACT-mask fallback'
            if n_mgood >= max(4, min(int(max_rows), 20)):
                return candidate_targets.loc[mgood].reset_index(drop=True), maskvals[mgood], ((note + ' [ACT-mask fallback]') if note else 'ACT-mask fallback')
        return False

    for mult in (1, 3, 8):
        try:
            targets = sample_targets_from_sampler(sampler, max_rows=max_rows * mult, seed=seed + mult)
        except Exception:
            continue
        res = _consider(targets, allow_mask_proxy=False, note='sampler targets')
        if res is not False:
            targets, kappa, target_note = res
            break
    else:
        # Strong fallback: use the mask footprint directly and, if needed, the mask values themselves
        try:
            mask_targets = _mask_based_target_positions(sampler, max_rows=max(max_rows * 4, 1000), seed=seed + 101)
            res = _consider(mask_targets, allow_mask_proxy=True, note='mask-footprint targets')
            if res is not False:
                targets, kappa, target_note = res
            elif best_n >= 3 and best_targets is not None and best_kappa is not None:
                targets = best_targets
                kappa = best_kappa
            else:
                raise DataUnavailable('Too few valid target positions inside the requested lensing footprint')
        except Exception:
            if best_n >= 3 and best_targets is not None and best_kappa is not None:
                targets = best_targets
                kappa = best_kappa
            else:
                raise DataUnavailable('Too few valid target positions inside the requested lensing footprint')
    if len(targets) > max_rows:
        take = targets.sample(min(int(max_rows), len(targets)), random_state=seed).index.to_numpy()
        targets = targets.loc[take].reset_index(drop=True)
        kappa = np.asarray(kappa, dtype=float)[take]
    gal, source = query_public_density_catalog_for_sampler(sampler, max_rows=max(4*max_rows, 4000), zmin=zmin, zmax=zmax, seed=seed)
    dens = density_proxy_at_targets(gal['ra'], gal['dec'], targets['ra'], targets['dec'], k=max(2, min(int(density_k), len(gal))))
    targets['density_proxy'] = dens
    targets['z'] = _assign_target_z_from_catalog(gal, targets)
    if np.all(~np.isfinite(targets['z'])):
        targets['z'] = 0.5 * (float(zmin) + float(zmax))
    else:
        medz = float(np.nanmedian(targets['z']))
        targets['z'] = np.where(np.isfinite(targets['z']), targets['z'], medz)
    if target_note:
        source = f"{source} ({target_note})"
    return targets.reset_index(drop=True), np.asarray(kappa, dtype=float), source


def quantile_split(values: Sequence[float], q: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    lo = np.nanquantile(arr, q)
    hi = np.nanquantile(arr, 1.0 - q)
    return arr <= lo, arr >= hi


def permute_within_groups(values: Sequence[float], groups: Sequence[Any], rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values)
    groups = np.asarray(groups)
    out = values.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = rng.permutation(out[idx])
    return out


# ---------- IRSA TAP / Euclid helpers ----------

IRSA_TAP_SYNC = "https://irsa.ipac.caltech.edu/TAP/sync"


def irsa_tap_query_csv(adql: str, maxrec: int | None = None, *, timeout: int = TIMEOUT, use_cache: bool = True) -> pd.DataFrame:
    payload = {"QUERY": adql, "LANG": "ADQL", "REQUEST": "doQuery", "FORMAT": "csv"}
    if maxrec is not None:
        payload["MAXREC"] = str(int(maxrec))
    cache_path = None
    if use_cache:
        key = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        cache_path = CACHE_DIR / "irsa_tap" / f"{key}.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return normalize_columns(pd.read_csv(cache_path))
    last_exc = None
    for read_timeout in (timeout, int(timeout * 1.5), int(timeout * 2.0)):
        try:
            r = SESSION.post(IRSA_TAP_SYNC, data=payload, timeout=(30, read_timeout))
            r.raise_for_status()
            text = r.text.strip()
            if not text:
                raise DataUnavailable("IRSA TAP returned empty response")
            df = normalize_columns(pd.read_csv(io.StringIO(text)))
            if len(df.columns) == 1 and ("error" in str(df.columns[0]).lower() or "exception" in str(df.columns[0]).lower()):
                raise DataUnavailable(f"IRSA TAP error: {df.iloc[0, 0] if len(df) else df.columns[0]}")
            if cache_path is not None:
                df.to_csv(cache_path, index=False)
            return df
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.0)
    raise last_exc


def tap_schema_columns(table_name: str) -> list[str]:
    q = f"SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name = '{table_name}'"
    try:
        df = irsa_tap_query_csv(q)
        return [str(x).strip() for x in df.iloc[:, 0].dropna().tolist()]
    except Exception:
        return []


EUCLID_MER_TABLE = "euclid_q1_mer_catalogue"
EUCLID_PHZ_TABLE = "euclid_q1_phz_photo_z"


def choose_euclid_photoz_column(columns: Sequence[str]) -> str:
    cols = list(columns)
    preferred = ["phz_median", "phz_best", "phz_mode", "phz_weighted_mean", "photo_z", "z_phot", "zbest"]
    for cand in preferred:
        for col in cols:
            if col.lower() == cand.lower():
                return col
    for col in cols:
        cl = col.lower()
        if "phz" in cl and any(k in cl for k in ["median", "best", "mode", "weighted", "mean"]):
            return col
    for col in cols:
        cl = col.lower()
        if cl.startswith("z") or "redshift" in cl:
            return col
    raise DataUnavailable(f"Could not discover a Euclid photo-z column from: {cols[:20]}")


def _split_linear_range(lo: float, hi: float, n: int) -> list[tuple[float, float]]:
    if n <= 1 or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [(float(lo), float(hi))]
    edges = np.linspace(float(lo), float(hi), int(n) + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]


def _split_ra_ranges(ra_range: tuple[float, float] | None, n: int) -> list[tuple[float, float] | None]:
    if ra_range is None:
        return [None]
    lo, hi = float(ra_range[0]), float(ra_range[1])
    if hi < lo:
        segs = [(lo, 360.0), (0.0, hi)]
    else:
        segs = [(lo, hi)]
    per_seg = max(1, int(math.ceil(n / max(1, len(segs)))))
    out: list[tuple[float, float]] = []
    for slo, shi in segs:
        out.extend(_split_linear_range(slo, shi, per_seg))
    return out[: max(1, n)]


def _euclid_where_sql(zcol: str, zlo: float, zhi: float, ra_sub: tuple[float, float] | None, dec_sub: tuple[float, float] | None) -> str:
    where = [
        f"p.{zcol} BETWEEN {float(zlo)} AND {float(zhi)}",
        "m.ra IS NOT NULL",
        "m.dec IS NOT NULL",
    ]
    if ra_sub is not None:
        ra_lo, ra_hi = float(ra_sub[0]), float(ra_sub[1])
        if ra_hi < ra_lo:
            ra_hi += 360.0
        if ra_hi - ra_lo < 359.0:
            if ra_hi <= 360.0:
                where.append(f"m.ra BETWEEN {ra_lo} AND {ra_hi}")
            else:
                hi_wrap = ra_hi - 360.0
                where.append(f"(m.ra >= {ra_lo} OR m.ra <= {hi_wrap})")
    if dec_sub is not None:
        dec_lo, dec_hi = float(dec_sub[0]), float(dec_sub[1])
        where.append(f"m.dec BETWEEN {dec_lo} AND {dec_hi}")
    return "\n      AND ".join(where)


def _euclid_query_block(limit: int, zcol: str, zlo: float, zhi: float, ra_sub: tuple[float, float] | None, dec_sub: tuple[float, float] | None) -> pd.DataFrame:
    where_sql = _euclid_where_sql(zcol, zlo, zhi, ra_sub, dec_sub)
    query = f"""
    SELECT TOP {int(limit)}
        m.object_id AS object_id,
        m.ra AS ra,
        m.dec AS dec,
        p.{zcol} AS z
    FROM {EUCLID_MER_TABLE} AS m
    JOIN {EUCLID_PHZ_TABLE} AS p
      ON m.object_id = p.object_id
    WHERE {where_sql}
    """
    return normalize_columns(irsa_tap_query_csv(query, maxrec=limit, timeout=TIMEOUT))


def load_euclid_q1_sample(
    max_rows: int = 5000,
    zmin: float = 0.2,
    zmax: float = 1.5,
    seed: int = 0,
    ra_range: tuple[float, float] | None = None,
    dec_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Query a modest Euclid Q1 sample directly from IRSA TAP.

    Uses TAP_SCHEMA discovery first so the script survives minor catalog-schema changes.
    The returned columns are normalized to ``ra``, ``dec``, and ``z``.

    To avoid IRSA sync timeouts, queries are chunked into small z/sky blocks and then
    concatenated locally.
    """
    phz_cols = tap_schema_columns(EUCLID_PHZ_TABLE)
    zcol = choose_euclid_photoz_column(phz_cols) if phz_cols else "phz_median"

    want = int(max_rows)
    # Keep sync TAP requests small and let local concatenation do the work.
    per_query = int(min(180, max(60, want // 12)))
    z_splits = 4 if want >= 1200 else 3 if want >= 500 else 2
    ra_splits = 6 if ra_range is not None and want >= 1200 else 4 if ra_range is not None and want >= 500 else 2 if ra_range is not None else 1
    dec_splits = 4 if dec_range is not None and want >= 1200 else 2 if dec_range is not None and want >= 400 else 1

    z_chunks = _split_linear_range(float(zmin), float(zmax), z_splits)
    ra_chunks = _split_ra_ranges(ra_range, ra_splits)
    dec_chunks = _split_linear_range(float(dec_range[0]), float(dec_range[1]), dec_splits) if dec_range is not None else [None]

    frames: list[pd.DataFrame] = []
    enough = max(int(0.35 * want), min(250, want))
    for zlo, zhi in z_chunks:
        for ra_sub in ra_chunks:
            for dec_sub in dec_chunks:
                try:
                    part = _euclid_query_block(per_query, zcol, zlo, zhi, ra_sub, dec_sub)
                except Exception:
                    continue
                if not part.empty:
                    frames.append(part)
                cur_n = sum(len(x) for x in frames)
                if cur_n >= enough:
                    break
            if sum(len(x) for x in frames) >= enough:
                break
        if sum(len(x) for x in frames) >= enough:
            break

    if not frames:
        # final fallback over the full requested range, but still keep it small
        frames = [_euclid_query_block(min(max(want, 120), 300), zcol, float(zmin), float(zmax), ra_range, dec_range)]

    df = normalize_columns(pd.concat(frames, ignore_index=True))
    if df.empty:
        raise DataUnavailable("No Euclid Q1 rows returned from IRSA TAP")
    if "object_id" in df.columns:
        df = df.drop_duplicates(subset=["object_id"])
    ra_col = first_existing_column(df, ["ra"])
    dec_col = first_existing_column(df, ["dec"])
    z_col = first_existing_column(df, ["z", zcol, "phz_median", "phz_best", "phz_mode", "phz_weighted_mean"])
    df["z"] = pd.to_numeric(df[z_col], errors="coerce")
    df["ra"] = pd.to_numeric(df[ra_col], errors="coerce")
    df["dec"] = pd.to_numeric(df[dec_col], errors="coerce")
    df = df[np.isfinite(df["ra"]) & np.isfinite(df["dec"]) & np.isfinite(df["z"])]
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    return df.reset_index(drop=True)


# ---------- FITS / WCS / HEALPix access ----------


def ensure_astropy_stack() -> None:
    ensure_package("astropy")
    ensure_package("astropy-healpix", "astropy_healpix")


def open_fits(path: os.PathLike[str] | str):
    ensure_astropy_stack()
    from astropy.io import fits

    return fits.open(path)


def extract_tar_member(tar_path: os.PathLike[str] | str, member_pattern: str, dest_dir: os.PathLike[str] | str) -> Path:
    tar_path = Path(tar_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    rx = re.compile(member_pattern)
    with tarfile.open(tar_path, "r:*") as tf:
        matches = [m for m in tf.getmembers() if rx.search(m.name)]
        if not matches:
            raise DataUnavailable(f"No tar member matching /{member_pattern}/ in {tar_path.name}")
        member = matches[0]
        out = dest_dir / Path(member.name).name
        if out.exists() and out.stat().st_size > 0:
            return out
        with tf.extractfile(member) as src, out.open("wb") as dst:
            if src is None:
                raise DataUnavailable(f"Failed to extract {member.name}")
            shutil.copyfileobj(src, dst)
        return out


def extract_zip_member(zip_path: os.PathLike[str] | str, member_pattern: str, dest_dir: os.PathLike[str] | str) -> Path:
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    rx = re.compile(member_pattern)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        matches = [n for n in names if rx.search(n)]
        if not matches:
            raise DataUnavailable(f"No zip member matching /{member_pattern}/ in {zip_path.name}")
        name = matches[0]
        out = dest_dir / Path(name).name
        if out.exists() and out.stat().st_size > 0:
            return out
        with zf.open(name) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return out


class SkyMapSampler:
    def __init__(self, path: os.PathLike[str] | str):
        self.path = Path(path)
        self.mode = None
        self.data = None
        self.wcs = None
        self.nside = None
        self.order = "ring"
        self.mask = None
        self.mask_threshold = 0.95
        self._load()

    def _load(self) -> None:
        ensure_astropy_stack()
        from astropy.io import fits
        from astropy.wcs import WCS

        with fits.open(self.path) as hdul:
            # Try image HDU + WCS first.
            for hdu in hdul:
                if getattr(hdu, "data", None) is None:
                    continue
                arr = np.asarray(hdu.data)
                if arr.ndim >= 2 and hdu.header.get("CTYPE1"):
                    self.mode = "wcs"
                    self.data = np.squeeze(arr)
                    self.wcs = WCS(hdu.header)
                    return
            # Fall back to HEALPix-style binary table.
            for hdu in hdul:
                data = getattr(hdu, "data", None)
                if data is None:
                    continue
                names = list(getattr(data, "names", []) or [])
                if names:
                    col = names[0]
                    # HEALPix FITS tables often store the map as a single vector-valued
                    # column with shape like (1, npix) or (npix, 1). Flatten to the actual
                    # 1D pixel vector; downstream samplers expect a plain 1D array.
                    arr = np.ravel(np.asarray(data[col], dtype=float))
                    nside = hdu.header.get("NSIDE")
                    ordering = str(hdu.header.get("ORDERING", "RING")).strip().lower()
                    if nside is None:
                        # infer from npix = 12 nside^2
                        npix = arr.size
                        nside = int(round((npix / 12.0) ** 0.5))
                    self.mode = "healpix"
                    self.data = arr
                    self.nside = int(nside)
                    self.order = ordering
                    return
        raise DataUnavailable(f"Could not interpret sky map format: {self.path}")

    def sample(self, ra_deg: Sequence[float], dec_deg: Sequence[float]) -> np.ndarray:
        ensure_astropy_stack()
        ra = np.asarray(ra_deg, dtype=float)
        dec = np.asarray(dec_deg, dtype=float)
        if self.mode == "wcs":
            from astropy.coordinates import SkyCoord
            from astropy import units as u

            coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            xpix, ypix = self.wcs.world_to_pixel(coords)
            xpix = np.asarray(np.round(xpix), dtype=int)
            ypix = np.asarray(np.round(ypix), dtype=int)
            vals = np.full(ra.shape, np.nan, dtype=float)
            good = (
                (xpix >= 0)
                & (ypix >= 0)
                & (ypix < self.data.shape[-2])
                & (xpix < self.data.shape[-1])
            )
            vals[good] = self.data[ypix[good], xpix[good]]
            return vals
        if self.mode == "healpix":
            from astropy_healpix import HEALPix
            from astropy import units as u

            hp = HEALPix(nside=self.nside, order=self.order, frame="icrs")
            lon = ra * u.deg
            lat = dec * u.deg
            idx = hp.lonlat_to_healpix(lon, lat)
            idx = np.asarray(idx, dtype=int)
            vals = np.full(ra.shape, np.nan, dtype=float)
            good = (idx >= 0) & (idx < len(self.data))
            vals[good] = np.ravel(self.data)[idx[good]]
            if self.mask is not None:
                flat_mask = np.ravel(self.mask)
                mgood = good & (idx >= 0) & (idx < len(flat_mask))
                bad = np.zeros_like(good, dtype=bool)
                bad[mgood] = flat_mask[idx[mgood]] < self.mask_threshold
                vals[bad] = np.nan
            return vals
        raise DataUnavailable("SkyMapSampler not initialized")




class HealpixArraySampler:
    def __init__(self, data: np.ndarray, nside: int, order: str = "ring", mask: np.ndarray | None = None, mask_threshold: float = 0.95):
        self.mode = "healpix"
        self.data = np.asarray(data, dtype=float)
        self.nside = int(nside)
        self.order = str(order).lower()
        self.mask = None if mask is None else np.asarray(mask, dtype=float)
        self.mask_threshold = float(mask_threshold)

    def sample(self, ra_deg: Sequence[float], dec_deg: Sequence[float]) -> np.ndarray:
        ensure_astropy_stack()
        from astropy_healpix import HEALPix
        from astropy import units as u

        ra = np.asarray(ra_deg, dtype=float)
        dec = np.asarray(dec_deg, dtype=float)
        hp = HEALPix(nside=self.nside, order=self.order, frame="icrs")
        idx = np.asarray(hp.lonlat_to_healpix(ra * u.deg, dec * u.deg), dtype=int)
        vals = np.full(ra.shape, np.nan, dtype=float)
        good = (idx >= 0) & (idx < len(self.data))
        vals[good] = np.ravel(self.data)[idx[good]]
        if self.mask is not None:
            flat_mask = np.ravel(self.mask)
            mgood = good & (idx >= 0) & (idx < len(flat_mask))
            bad = np.zeros_like(good, dtype=bool)
            bad[mgood] = flat_mask[idx[mgood]] < self.mask_threshold
            vals[bad] = np.nan
        return vals


class ActAlmSampler:
    """Sample ACT DR6 kappa alm data directly at arbitrary sky positions.

    The ACT DR6 LAMBDA release distributes baseline products as spherical-harmonic
    coefficients (alm) plus a HEALPix mask, not as a ready-made pixel map. We
    therefore evaluate the alm directly at the requested positions using ducc0's
    arbitrary-position spherical-harmonic synthesis and then apply the ACT mask.
    """

    def __init__(self, alm: np.ndarray, lmax: int, mask_sampler: SkyMapSampler, mask_threshold: float = 0.99):
        self.alm = np.asarray(alm, dtype=np.complex128).reshape(1, -1)
        self.lmax = int(lmax)
        self.mask_sampler = mask_sampler
        self.mask_threshold = float(mask_threshold)

    def sample(self, ra_deg: Sequence[float], dec_deg: Sequence[float]) -> np.ndarray:
        ensure_package("ducc0")
        from ducc0 import sht

        ra = np.asarray(ra_deg, dtype=float)
        dec = np.asarray(dec_deg, dtype=float)
        mask = np.asarray(self.mask_sampler.sample(ra, dec), dtype=float)
        vals = np.full(ra.shape, np.nan, dtype=float)
        good = np.isfinite(mask) & (mask >= self.mask_threshold) & np.isfinite(ra) & np.isfinite(dec)
        if not np.any(good):
            return vals
        loc = np.column_stack([
            np.deg2rad(90.0 - dec[good]),
            np.mod(np.deg2rad(ra[good]), 2.0 * np.pi),
        ]).astype(np.float64)
        out = sht.synthesis_general(
            alm=self.alm,
            spin=0,
            lmax=self.lmax,
            loc=loc,
            epsilon=1e-6,
            nthreads=0,
        )
        vals[good] = np.asarray(out[0], dtype=float)
        return vals


def _infer_healpy_lmax_from_nalm(nalm: int) -> int:
    # healpy indexing: nalm = (lmax+1)(lmax+2)/2 for mmax=lmax
    lmax = int((math.isqrt(8 * int(nalm) + 1) - 3) // 2)
    if (lmax + 1) * (lmax + 2) // 2 != int(nalm):
        raise DataUnavailable(f"Could not infer lmax from nalm={nalm}")
    return lmax


def _read_complex_alm_from_fits(path: os.PathLike[str] | str) -> np.ndarray:
    ensure_astropy_stack()
    from astropy.io import fits

    with fits.open(path) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            arr = np.asarray(data)
            if np.iscomplexobj(arr) and arr.size > 0:
                return np.ravel(arr).astype(np.complex128)
            names = list(getattr(data, "names", []) or [])
            if names:
                lower = {str(n).lower(): n for n in names}
                for cand in ("alm", "kappa_alm", "alms"):
                    if cand in lower:
                        col = np.asarray(data[lower[cand]])
                        if np.iscomplexobj(col):
                            return np.ravel(col).astype(np.complex128)
                real_name = None
                imag_name = None
                for key in lower:
                    if key in {"real", "re", "alm_real"}:
                        real_name = lower[key]
                    if key in {"imag", "im", "alm_imag"}:
                        imag_name = lower[key]
                if real_name is not None and imag_name is not None:
                    re = np.asarray(data[real_name], dtype=float)
                    im = np.asarray(data[imag_name], dtype=float)
                    return (np.ravel(re) + 1j * np.ravel(im)).astype(np.complex128)
                # common FITS convention: one vector column with two components [real, imag]
                for name in names:
                    col = np.asarray(data[name])
                    if col.ndim >= 2 and col.shape[-1] == 2:
                        col = np.asarray(col, dtype=float)
                        return (np.ravel(col[..., 0]) + 1j * np.ravel(col[..., 1])).astype(np.complex128)
        raise DataUnavailable(f"Could not parse complex alm coefficients from {path}")


def _build_act_baseline_alm_sampler(tar_path: os.PathLike[str] | str, dest: os.PathLike[str] | str) -> ActAlmSampler:
    dest = Path(dest)
    alm_path = extract_tar_member(tar_path, r"baseline/.*/?kappa_alm_data_act_dr6_lensing_v1_baseline\.fits$|baseline/kappa_alm_data_act_dr6_lensing_v1_baseline\.fits$", dest)
    mask_path = extract_tar_member(tar_path, r"baseline/.*/?mask_act_dr6_lensing_v1_healpix_nside_4096_baseline\.fits$|baseline/mask_act_dr6_lensing_v1_healpix_nside_4096_baseline\.fits$", dest)
    alm = _read_complex_alm_from_fits(alm_path)
    lmax = _infer_healpy_lmax_from_nalm(len(alm))
    mask_sampler = SkyMapSampler(mask_path)
    return ActAlmSampler(alm=alm, lmax=lmax, mask_sampler=mask_sampler, mask_threshold=0.5)


def load_act_dr6_kappa_sampler(cache_subdir: str = "act_dr6") -> ActAlmSampler:
    dest = CACHE_DIR / cache_subdir
    dest.mkdir(exist_ok=True)
    tar_path = download_file(act_dr6_path(), dest / "dr6_lensing_release.tar.gz")
    try:
        return _build_act_baseline_alm_sampler(tar_path, dest)
    except Exception as exc:
        raise DataUnavailable(
            "ACT DR6 baseline products are provided as alm coefficients plus a HEALPix mask. "
            f"The bundle could not build a valid alm sampler from the public release: {exc}"
        ) from exc


def load_planck_pr4_kappa_sampler(cache_subdir: str = "planck_pr4") -> SkyMapSampler:
    dest = CACHE_DIR / cache_subdir
    dest.mkdir(exist_ok=True)
    tar_path = download_first_available(planck_pr4_candidates(), dest)
    patterns = [r".*mv.*fits$", r".*lensing.*fits$", r".*kappa.*fits$"]
    last_exc = None
    for pat in patterns:
        try:
            f = extract_tar_member(tar_path, pat, dest)
            return SkyMapSampler(f)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise DataUnavailable(f"Could not locate Planck PR4 kappa map in archive: {last_exc}")


# ---------- Data ingestion helpers ----------


def load_pantheon_plus(cache_subdir: str = "pantheon_plus") -> tuple[pd.DataFrame, np.ndarray]:
    dest = CACHE_DIR / cache_subdir
    dest.mkdir(exist_ok=True)
    paths = pantheon_plus_paths()
    data_path = download_file(paths["data"], dest / "PantheonPlusSH0ES.dat")
    cov_path = download_file(paths["cov_stat_sys"], dest / "PantheonPlusSH0ES_STAT+SYS.cov")
    df = pd.read_csv(data_path, sep=r"\s+", engine="python", comment="#")
    with cov_path.open("r", encoding="utf-8") as f:
        vals = [float(x) for x in f.read().split()]
    n = int(vals[0])
    cov = np.array(vals[1:], dtype=float).reshape(n, n)
    return df, cov


def parse_simple_three_column_file(path: os.PathLike[str] | str) -> pd.DataFrame:
    rows = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) >= 3:
            rows.append((float(parts[0]), float(parts[1]), str(parts[2])))
    return pd.DataFrame(rows, columns=["z", "value", "quantity"])


def load_desi_consensus(cache_subdir: str = "desi_bao") -> pd.DataFrame:
    dest = CACHE_DIR / cache_subdir
    dest.mkdir(exist_ok=True)
    path = download_file(desi_bao_paths()["consensus"], dest / "final_consensus_covtot_dM_Hz_fsig.txt")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    rows = [[float(x) for x in ln.split()] for ln in lines]
    return pd.DataFrame(rows)


def load_desi_bao_measurements(cache_subdir: str = "desi_bao") -> pd.DataFrame:
    dest = CACHE_DIR / cache_subdir
    dest.mkdir(exist_ok=True)
    tables = []
    keys = ["bgs_0p1_0p4", "lrg_0p6_0p8", "lrg_elg_0p8_1p1", "elg_1p1_1p6", "qso_0p8_2p1", "lya", "all_gccomb"]
    for key in keys:
        url = desi_bao_paths()[key]
        p = download_file(url, dest / Path(url).name)
        df = parse_simple_three_column_file(p)
        df["source_key"] = key
        tables.append(df)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["z", "value", "quantity", "source_key"])


def public_growth_paths() -> dict[str, str]:
    base = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master"
    return {
        "sdss_dr16_lrg": f"{base}/sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat",
        "sdss_dr16_qso": f"{base}/sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8.dat",
    }


def load_public_growth_points(cache_subdir: str = "public_growth") -> pd.DataFrame:
    dest = CACHE_DIR / cache_subdir
    dest.mkdir(exist_ok=True)
    tables = []
    for key, url in public_growth_paths().items():
        p = download_file(url, dest / Path(url).name)
        df = parse_simple_three_column_file(p)
        df["source_key"] = key
        tables.append(df)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["z", "value", "quantity", "source_key"])


def extract_desi_fs8_points(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {str(c).lower() for c in df.columns}
    if {"z", "value", "quantity"}.issubset(cols_lower):
        d = normalize_columns(df)
        qcol = first_existing_column(d, ["quantity"])
        zcol = first_existing_column(d, ["z"])
        vcol = first_existing_column(d, ["value"])
        mask = d[qcol].astype(str).str.lower().isin(["f_sigma8", "fsigma8", "fσ8", "f_sig8"])
        out = d.loc[mask, [zcol, vcol]].copy()
        out.columns = ["z", "fs8"]
        out["z"] = pd.to_numeric(out["z"], errors="coerce")
        out["fs8"] = pd.to_numeric(out["fs8"], errors="coerce")
        out = out[np.isfinite(out["z"]) & np.isfinite(out["fs8"])].sort_values("z").reset_index(drop=True)
        return out
    num = df.apply(pd.to_numeric, errors="coerce")
    zcol = None
    for c in num.columns:
        vals = num[c].dropna()
        if len(vals) >= 3 and vals.between(0.0, 5.0).all():
            zcol = c
            break
    if zcol is None:
        raise DataUnavailable("Could not identify redshift column in DESI consensus file")
    out = pd.DataFrame({"z": num[zcol]})
    candidates = []
    for c in num.columns:
        if c == zcol:
            continue
        vals = num[c].dropna()
        if len(vals) >= 3 and vals.between(0.0, 2.0).all() and vals.std() > 0:
            candidates.append((float(vals.mean()), c))
    if not candidates:
        raise DataUnavailable("Could not identify fs8-like column in DESI consensus file")
    _, fs8col = min(candidates, key=lambda t: abs(t[0] - 0.5))
    out["fs8"] = num[fs8col]
    out = out[np.isfinite(out["z"]) & np.isfinite(out["fs8"])].sort_values("z").reset_index(drop=True)
    return out


def load_sparc_rotation_curves(cache_subdir: str = "sparc") -> dict[str, pd.DataFrame]:
    dest = CACHE_DIR / cache_subdir
    dest.mkdir(exist_ok=True)
    rot_zip = download_file(sparc_paths()["rotmod"], dest / "Rotmod_LTG.zip")
    download_file(sparc_paths()["database"], dest / "sparc_database.zip")
    curves = {}
    with zipfile.ZipFile(rot_zip) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(('.dat', '.txt')):
                continue
            with zf.open(name) as f:
                text = io.TextIOWrapper(f, encoding="utf-8", errors="ignore").read()
            rows = []
            for ln in text.splitlines():
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = re.split(r"\s+", ln)
                if len(parts) < 3:
                    continue
                rows.append(parts)
            if rows:
                df = pd.DataFrame(rows)
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                curves[Path(name).stem] = df
    return curves


def load_nanograv_archive(cache_subdir: str = "nanograv15") -> Path:
    dest = CACHE_DIR / cache_subdir
    dest.mkdir(exist_ok=True)
    return download_file(nanograv_path(), dest / "NANOGrav15yr_PulsarTiming_v2.1.0.tar.gz", timeout=600)


def load_kmos3d_catalog(cache_subdir: str = "kmos3d") -> pd.DataFrame:
    dest = CACHE_DIR / cache_subdir
    dest.mkdir(exist_ok=True)
    tar_path = None
    errors: list[str] = []
    try:
        hrefs = discover_hrefs(
            "https://www.mpe.mpg.de/ir/KMOS3D/data",
            [r"3d_fnlsp_table.*fits(?:\.tgz|\.tar\.gz|\.gz)$", r"table.*fits(?:\.tgz|\.tar\.gz|\.gz)$"],
        )
        for href in hrefs:
            try:
                tar_path = download_file(href, dest / sanitize_filename(Path(href.split("?")[0]).name))
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"discovered href {href}: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"page discovery failed: {exc}")
    if tar_path is None:
        for cand in kmos3d_candidates()[1:]:
            try:
                tar_path = download_file(cand.url, dest / sanitize_filename(Path(cand.url.split("?")[0]).name))
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{cand.name}: {exc}")
    if tar_path is None:
        raise DownloadError("All KMOS3D catalog download attempts failed:\n" + "\n".join(errors))
    fits_path = extract_tar_member(tar_path, r"3d_fnlsp_table_v3.*\.fits$|.*table.*\.fits$|.*catalog.*\.fits$", dest)
    ensure_astropy_stack()
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is not None and getattr(data, "names", None):
                arr = np.array(data)
                try:
                    arr = arr.byteswap().newbyteorder()
                except AttributeError:
                    arr = arr.byteswap().view(arr.dtype.newbyteorder('='))
                return normalize_columns(pd.DataFrame(arr))
    raise DataUnavailable("Could not locate tabular FITS extension in KMOS3D catalog")


def _read_csv_like_table(text: str) -> pd.DataFrame:
    snip = text.lstrip()[:4096]
    low = snip.lower()
    if "no objects have been found" in low:
        return pd.DataFrame()
    if low.startswith("<!doctype html") or low.startswith("<html"):
        raise DataUnavailable("SDSS SkyServer returned HTML instead of CSV/JSON")
    if low.startswith("error"):
        raise DataUnavailable(snip.splitlines()[0])
    lines = text.splitlines()
    while lines and lines[0].strip().startswith("#"):
        lines.pop(0)
    cleaned = "\n".join(lines).strip()
    if not cleaned:
        return pd.DataFrame()
    try:
        return normalize_columns(pd.read_csv(io.StringIO(cleaned)))
    except Exception as exc:
        raise DataUnavailable(f"Could not parse SDSS CSV response: {exc}") from exc

def _rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return normalize_columns(pd.DataFrame(rows))


def query_sdss_sql(sql: str, cache_key: str | None = None) -> pd.DataFrame:
    cache_path = CACHE_DIR / "sdss_sql" / f"{sanitize_filename(cache_key or str(abs(hash(sql))))}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            cached = normalize_columns(pd.read_csv(cache_path))
            bad_cols = {str(c).strip().lower() for c in cached.columns}
            if bad_cols in ({"#table1"}, {"tablename", "rows"}) or any(str(c).strip().lower().startswith("#table") for c in cached.columns):
                cache_path.unlink(missing_ok=True)
            else:
                return cached
        except Exception:
            cache_path.unlink(missing_ok=True)

    endpoints = [
        "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch",
        "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch",
    ]
    errors: list[str] = []
    for endpoint in endpoints:
        for fmt in ("json", "csv"):
            try:
                r = SESSION.get(endpoint, params={"cmd": sql, "format": fmt}, timeout=TIMEOUT)
                r.raise_for_status()
                text = r.text
                if fmt == "json":
                    payload = r.json()
                    if isinstance(payload, dict) and "Rows" in payload:
                        rows = payload.get("Rows") or []
                        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                            df = _rows_to_dataframe(rows)
                        elif isinstance(rows, list):
                            df = normalize_columns(pd.DataFrame(rows))
                        else:
                            raise DataUnavailable("Unrecognized SDSS JSON 'Rows' payload")
                    elif isinstance(payload, list):
                        df = _rows_to_dataframe(payload)
                    else:
                        raise DataUnavailable("Unrecognized JSON response shape from SDSS")
                else:
                    df = _read_csv_like_table(text)
                bad_cols = {str(c).strip().lower() for c in df.columns}
                if bad_cols in ({"#table1"}, {"tablename", "rows"}) or any(str(c).strip().lower().startswith("#table") for c in df.columns):
                    raise DataUnavailable(f"SDSS returned metadata wrapper columns {list(df.columns)}")
                if not df.empty:
                    df.to_csv(cache_path, index=False)
                return df
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{endpoint} [{fmt}]: {exc}")
    raise DataUnavailable("All SDSS SQL attempts failed: " + " | ".join(errors))


def _sdss_where_window(zmin: float, zmax: float, ra_range: tuple[float, float] | None = None, dec_range: tuple[float, float] | None = None, table_prefix: str = "") -> str:
    p = f"{table_prefix}." if table_prefix and not table_prefix.endswith(".") else table_prefix
    where = [f"{p}z BETWEEN {float(zmin)} AND {float(zmax)}"]
    if ra_range is not None:
        ra_lo, ra_hi = float(ra_range[0]), float(ra_range[1])
        if ra_hi < ra_lo:
            where.append(f"({p}ra >= {ra_lo} OR {p}ra <= {ra_hi})")
        else:
            where.append(f"{p}ra BETWEEN {ra_lo} AND {ra_hi}")
    if dec_range is not None:
        dec_lo, dec_hi = float(dec_range[0]), float(dec_range[1])
        where.append(f"{p}dec BETWEEN {dec_lo} AND {dec_hi}")
    return " AND ".join(where)


def query_sdss_galaxies(
    n_rows: int,
    zmin: float,
    zmax: float,
    cache_key: str | None = None,
    ra_range: tuple[float, float] | None = None,
    dec_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    spec_where = _sdss_where_window(zmin, zmax, ra_range=ra_range, dec_range=dec_range, table_prefix="")
    spec_where_s = _sdss_where_window(zmin, zmax, ra_range=ra_range, dec_range=dec_range, table_prefix="s")
    queries = [
        f"SELECT TOP {int(n_rows)} ra, dec, z FROM SpecPhotoAll WHERE class = 'GALAXY' AND {spec_where}",
        f"SELECT TOP {int(n_rows)} s.ra AS ra, s.dec AS dec, s.z AS z FROM SpecObj AS s WHERE s.class = 'GALAXY' AND {spec_where_s}",
        f"SELECT TOP {int(n_rows)} p.ra AS ra, p.dec AS dec, s.z AS z FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestObjID = p.objID WHERE s.class = 'GALAXY' AND {_sdss_where_window(zmin, zmax, ra_range=ra_range, dec_range=dec_range, table_prefix='s')}",
    ]
    errors = []
    key_base = cache_key or 'sdss'
    for i, sql in enumerate(queries):
        try:
            df = query_sdss_sql(sql, cache_key=f"{key_base}_v4_{i}")
            if df.empty:
                errors.append(f"query {i}: empty result")
                continue
            cols = {str(c).lower(): c for c in df.columns}
            if {"ra", "dec"}.issubset(cols):
                out = df[[cols["ra"], cols["dec"]]].copy()
                out["z"] = pd.to_numeric(df[cols["z"]], errors="coerce") if "z" in cols else np.nan
                out.columns = ["ra", "dec", "z"]
                out[["ra", "dec", "z"]] = out[["ra", "dec", "z"]].apply(pd.to_numeric, errors="coerce")
                out = out[np.isfinite(out["ra"]) & np.isfinite(out["dec"])]
                if not out.empty:
                    return out.reset_index(drop=True)
            errors.append(f"query {i}: unexpected columns {list(df.columns)}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"query {i}: {exc}")

    # Last-resort public SDSS photometric proxy: positions only, with z filled by the bin midpoint.
    photo_where = []
    if ra_range is not None:
        ra_lo, ra_hi = float(ra_range[0]), float(ra_range[1])
        if ra_hi < ra_lo:
            photo_where.append(f"(ra >= {ra_lo} OR ra <= {ra_hi})")
        else:
            photo_where.append(f"ra BETWEEN {ra_lo} AND {ra_hi}")
    if dec_range is not None:
        dec_lo, dec_hi = float(dec_range[0]), float(dec_range[1])
        photo_where.append(f"dec BETWEEN {dec_lo} AND {dec_hi}")
    photo_where_sql = (' WHERE ' + ' AND '.join(photo_where)) if photo_where else ''
    photo_queries = [
        f"SELECT TOP {int(n_rows)} ra, dec FROM PhotoObj{photo_where_sql}",
        f"SELECT TOP {int(n_rows)} ra, dec FROM Galaxy{photo_where_sql}",
    ]
    for j, sql in enumerate(photo_queries):
        try:
            df = query_sdss_sql(sql, cache_key=f"{key_base}_v4_photo_{j}")
            cols = {str(c).lower(): c for c in df.columns}
            if not {"ra", "dec"}.issubset(cols):
                errors.append(f"photo query {j}: unexpected columns {list(df.columns)}")
                continue
            out = df[[cols["ra"], cols["dec"]]].copy()
            out.columns = ["ra", "dec"]
            out[["ra", "dec"]] = out[["ra", "dec"]].apply(pd.to_numeric, errors="coerce")
            out = out[np.isfinite(out["ra"]) & np.isfinite(out["dec"])]
            if not out.empty:
                out["z"] = 0.5 * (float(zmin) + float(zmax))
                return out.reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"photo query {j}: {exc}")

    raise DataUnavailable("All SDSS SQL galaxy queries failed: " + " | ".join(errors))


def sample_sdss_overlap_with_sampler(
    sampler: Any,
    max_rows: int = 5000,
    zmin: float = 0.05,
    zmax: float = 0.7,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    windows = []
    for thr in (None, 0.5, 0.2):
        rw, dw = sampler_query_window(sampler, threshold=thr)
        windows.append((rw, dw))
    windows.append((None, None))
    best_gal = None
    best_kappa = None
    best_n = -1
    for ra_window, dec_window in windows:
        for mult in (2, 4, 8, 12):
            request_rows = int(min(max(max_rows * mult, 1200), 30000))
            try:
                gal = query_sdss_galaxies(request_rows, zmin, zmax, cache_key=f"sdss_overlap_{request_rows}_{ra_window}_{dec_window}", ra_range=ra_window, dec_range=dec_window)
            except Exception:
                continue
            kappa = np.asarray(sampler.sample(gal["ra"], gal["dec"]), dtype=float)
            good = np.isfinite(kappa)
            n_good = int(np.sum(good))
            if n_good > best_n:
                best_gal = gal.loc[good].reset_index(drop=True)
                best_kappa = kappa[good]
                best_n = n_good
            if n_good >= max(8, min(max_rows, 60)):
                gal = gal.loc[good].reset_index(drop=True)
                kappa = kappa[good]
                if len(gal) > max_rows:
                    take = gal.sample(max_rows, random_state=seed).index.to_numpy()
                    gal = gal.loc[take].reset_index(drop=True)
                    kappa = kappa[take]
                return gal.reset_index(drop=True), np.asarray(kappa, dtype=float)
    if best_n >= 5 and best_gal is not None and best_kappa is not None:
        gal = best_gal
        kappa = np.asarray(best_kappa, dtype=float)
        if len(gal) > max_rows:
            take = gal.sample(min(max_rows, len(gal)), random_state=seed).index.to_numpy()
            gal = gal.loc[take].reset_index(drop=True)
            kappa = kappa[take]
        return gal.reset_index(drop=True), kappa
    raise DataUnavailable("Too few SDSS galaxies overlap the requested lensing footprint")


def sample_public_overlap_with_sampler(
    sampler: Any,
    max_rows: int = 5000,
    zmin: float = 0.2,
    zmax: float = 1.5,
    seed: int = 0,
    oversample: int = 8,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    try:
        gal, kappa = sample_euclid_overlap_with_sampler(sampler, max_rows=max_rows, zmin=zmin, zmax=zmax, seed=seed, oversample=oversample)
        gal = gal.copy()
        gal["catalog_source"] = "Euclid Q1 IRSA TAP"
        return gal, kappa, "Euclid Q1 IRSA TAP"
    except Exception:
        # Robust public-data fallback so screening tests can still run when Euclid IRSA is flaky
        # or the overlap is too sparse.
        gal, kappa = sample_sdss_overlap_with_sampler(sampler, max_rows=max_rows, zmin=max(0.05, zmin), zmax=min(0.7, max(0.7, zmax)), seed=seed)
        gal = gal.copy()
        gal["catalog_source"] = "SDSS DR17 SkyServer SQL"
        return gal, kappa, "SDSS DR17 SkyServer SQL"


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # relaxed search by substring
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    raise KeyError(f"None of the candidate columns found: {candidates}")


def json_result_template(test_name: str, description: str) -> dict[str, Any]:
    return {
        "test_name": test_name,
        "description": description,
        "generated_utc": now_utc(),
        "data_sources": [],
        "notes": [],
        "status": "ok",
    }


def add_source(result: dict[str, Any], label: str, url: str, local_path: os.PathLike[str] | str | None = None) -> None:
    entry = {"label": label, "url": url}
    if local_path is not None:
        entry["local_path"] = str(local_path)
    result.setdefault("data_sources", []).append(entry)


def get_output_path(script_path: os.PathLike[str] | str, suffix: str = ".json") -> Path:
    stem = Path(script_path).stem
    return OUT_DIR / f"{stem}{suffix}"


# ---------- Simple domain-specific proxies ----------


def characteristic_curve_acceleration(df: pd.DataFrame) -> float:
    num = df.apply(pd.to_numeric, errors="coerce")
    if num.shape[1] < 2:
        return float("nan")
    r = num.iloc[:, 0].to_numpy(dtype=float)
    v = num.iloc[:, 1].to_numpy(dtype=float)
    m = np.isfinite(r) & np.isfinite(v) & (r > 0) & (v > 0)
    if m.sum() < 5:
        return float("nan")
    g_obs = (v[m] ** 2) / np.clip(r[m], 1e-6, None)
    return float(np.nanmedian(g_obs))


def estimate_mond_a0_from_curve(df: pd.DataFrame) -> float:
    """Estimate a MOND-like acceleration scale from a SPARC-style rotation curve.

    Proxy-level estimate based on the observed-vs-baryonic turnover, with a
    fallback to a compressed characteristic-acceleration proxy when the full
    baryonic decomposition is not usable.
    """
    num = df.apply(pd.to_numeric, errors="coerce")
    if num.shape[1] < 2:
        return float("nan")
    r = num.iloc[:, 0].to_numpy(dtype=float)
    v = num.iloc[:, 1].to_numpy(dtype=float)
    m = np.isfinite(r) & np.isfinite(v) & (r > 0) & (v > 0)
    if m.sum() < 5:
        return float("nan")
    g_obs = (v[m] ** 2) / np.clip(r[m], 1e-6, None)
    if num.shape[1] >= 3:
        vb = num.iloc[:, min(2, num.shape[1] - 1)].to_numpy(dtype=float)
        mb = m & np.isfinite(vb) & (vb > 0)
        if mb.sum() >= 5:
            g_obs_b = (v[mb] ** 2) / np.clip(r[mb], 1e-6, None)
            g_bar = (vb[mb] ** 2) / np.clip(r[mb], 1e-6, None)
            ratio = np.clip(g_obs_b / np.clip(g_bar, 1e-12, None), 1e-6, None)
            a0 = np.nanmedian(g_obs_b / np.clip(ratio - 1.0, 1e-6, None))
            if np.isfinite(a0) and a0 > 0:
                return float(a0)
    # Fallback: compress the characteristic observed acceleration into the same
    # proxy scale used by the SPARC turnover estimate.
    g_char = np.nanmedian(g_obs)
    return float(np.exp(0.25 * np.log(np.clip(g_char, 1e-12, None))))


def calibrate_sparc_a0_mapping(curves: dict[str, pd.DataFrame]) -> dict[str, float]:
    g_chars = []
    a0s = []
    for df in curves.values():
        g = characteristic_curve_acceleration(df)
        a0 = estimate_mond_a0_from_curve(df)
        if np.isfinite(g) and np.isfinite(a0) and g > 0 and a0 > 0:
            g_chars.append(g)
            a0s.append(a0)
    if len(g_chars) < 5:
        raise DataUnavailable("Not enough SPARC curves for a0 mapping calibration")
    x = np.log10(np.asarray(g_chars, dtype=float))
    y = np.log10(np.asarray(a0s, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    slope = float(np.clip(slope, 0.15, 1.0))
    return {
        "slope": slope,
        "intercept": float(intercept),
        "local_a0": float(np.nanmedian(a0s)),
        "local_g_char": float(np.nanmedian(g_chars)),
        "n_curves": int(len(a0s)),
    }


def _convert_radius_to_kpc(radius: np.ndarray, z: np.ndarray) -> np.ndarray:
    r = np.asarray(radius, dtype=float)
    z = np.asarray(z, dtype=float)
    finite = r[np.isfinite(r) & (r > 0)]
    if finite.size == 0:
        return r
    median_r = float(np.nanmedian(finite))
    if median_r > 20.0:
        return r
    # Many resolved high-z catalogs report effective radius in arcsec.
    da_mpc = comoving_distance_mpc(z) / np.clip(1.0 + z, 1e-6, None)
    kpc_per_arcsec = da_mpc * 1000.0 * (np.pi / 648000.0)
    return r * kpc_per_arcsec


def simple_exponential_scale(r: Sequence[float], corr: Sequence[float]) -> float:
    r = np.asarray(r, dtype=float)
    y = np.asarray(corr, dtype=float)
    m = np.isfinite(r) & np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return float("nan")
    coeff = np.polyfit(r[m], np.log(y[m]), 1)
    if coeff[0] >= 0:
        return float("inf")
    return float(-1.0 / coeff[0])


def fit_live_fraction(z: Sequence[float], fs8: Sequence[float]) -> FitResult:
    z = np.asarray(z, dtype=float)
    y = np.asarray(fs8, dtype=float)
    m = np.isfinite(z) & np.isfinite(y)
    z = z[m]
    y = y[m]
    if len(y) < 3:
        raise DataUnavailable("Need at least three fσ8 points for live/frozen fit")

    lookback = 1.0 - 1.0 / np.clip(1.0 + z, 1e-6, None)

    def model(p: np.ndarray) -> np.ndarray:
        amp, alpha, live = p
        base = 0.8 / np.clip((1.0 + z) ** 0.55, 1e-6, None)
        drift = np.exp(alpha * (1.0 - lookback / max(float(np.nanmax(lookback)), 1e-6)))
        return amp * ((1.0 - live) * base + live * base * drift)

    def obj(p: np.ndarray) -> float:
        amp, alpha, live = p
        pred = model(p)
        var = max(float(np.nanvar(y)), 1e-4)
        chi = float(np.sum((y - pred) ** 2 / var))
        regularizer = (alpha / 0.35) ** 2 + 0.25 * ((live - 0.2) / 0.4) ** 2
        return chi + regularizer

    bounds = [(0.7, 1.3), (-0.6, 0.6), (0.0, 1.0)]
    res = optimize.differential_evolution(obj, bounds=bounds, seed=0, polish=True)
    amp, alpha, live = res.x
    chi2 = obj(res.x)
    return FitResult(params={"amplitude": float(amp), "alpha": float(alpha), "live_fraction": float(live), "frozen_fraction": float(1 - live)}, chi2=float(chi2), ndof=max(len(y) - 3, 1))


def infer_kmos_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = list(df.columns)
    zcol = None
    for cand in ["z", "zbest", "z_ha", "redshift", "zneb", "zspec"]:
        try:
            zcol = first_existing_column(df, [cand])
            break
        except Exception:
            pass
    if zcol is None:
        for c in cols:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals) >= 10 and vals.between(0.0, 5.0).mean() > 0.9:
                zcol = c
                break
    if zcol is None:
        raise DataUnavailable("Could not infer KMOS3D redshift column")

    vcol = None
    for group in [["vrot", "vcirc", "v_circ", "vmax", "vc", "v22", "v_2.2", "vrot_re"], ["vel", "rot"]]:
        try:
            vcol = first_existing_column(df, group)
            break
        except Exception:
            pass
    if vcol is None:
        for c in cols:
            cl = c.lower()
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals) >= 10 and 20 < vals.median() < 500 and not any(bad in cl for bad in ["err", "sig", "disp"]):
                vcol = c
                if any(k in cl for k in ["v", "rot", "circ", "max"]):
                    break
    rcol = None
    for group in [["re", "r_e", "rhalf", "reff", "radius", "r2.2", "r_2.2"], ["rad", "size"]]:
        try:
            rcol = first_existing_column(df, group)
            break
        except Exception:
            pass
    if rcol is None:
        for c in cols:
            cl = c.lower()
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals) >= 10 and 0.05 < vals.median() < 30 and not any(bad in cl for bad in ["err", "sig"]):
                rcol = c
                if any(k in cl for k in ["r", "re", "rad", "size"]):
                    break
    if vcol is None or rcol is None:
        raise DataUnavailable(f"Could not infer usable KMOS3D kinematic columns from columns: {cols[:30]}")
    return {"z": zcol, "vrot": vcol, "radius": rcol}


def kmos_highz_a0_proxy(df: pd.DataFrame, sparc_mapping: dict[str, float] | None = None) -> pd.DataFrame:
    cols = infer_kmos_columns(df)
    z = pd.to_numeric(df[cols["z"]], errors="coerce")
    v = pd.to_numeric(df[cols["vrot"]], errors="coerce")
    r = pd.to_numeric(df[cols["radius"]], errors="coerce")
    out = pd.DataFrame({"z": z, "vrot": v, "radius": r})
    out = out[np.isfinite(out["z"]) & np.isfinite(out["vrot"]) & np.isfinite(out["radius"]) & (out["radius"] > 0) & (out["vrot"] > 0)]
    if out.empty:
        raise DataUnavailable("Could not infer usable KMOS3D kinematic columns")
    out["radius_kpc"] = _convert_radius_to_kpc(out["radius"].to_numpy(dtype=float), out["z"].to_numpy(dtype=float))
    out = out[np.isfinite(out["radius_kpc"]) & (out["radius_kpc"] > 0)].copy()
    out["g_char"] = (out["vrot"] ** 2) / out["radius_kpc"]
    if sparc_mapping is None:
        out["a0_proxy"] = np.exp(0.25 * np.log(np.clip(out["g_char"], 1e-12, None)))
    else:
        slope = float(sparc_mapping["slope"])
        intercept = float(sparc_mapping["intercept"])
        out["a0_proxy"] = 10.0 ** (intercept + slope * np.log10(np.clip(out["g_char"], 1e-12, None)))
    return out


def pairwise_ratio_summary(values: Sequence[float]) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals != 0)]
    if len(vals) < 2:
        return {"max_pairwise_ratio": float("nan")}
    ratios = []
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            ratios.append(max(abs(vals[i] / vals[j]), abs(vals[j] / vals[i])))
    return {"max_pairwise_ratio": float(np.max(ratios))}


def read_csv_flexible(path: os.PathLike[str] | str) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, comment="#")


def load_direct_detection_curves(cache_subdir: str = "dd_curves") -> dict[str, pd.DataFrame]:
    dest = CACHE_DIR / cache_subdir
    dest.mkdir(exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    for key, url in xnt_wimp_paths().items():
        p = download_file(url, dest / f"{key}.csv")
        out[key] = read_csv_flexible(p)

    for key, url in lz_hepdata_paths().items():
        try:
            p = download_file(url, dest / f"{key}.csv")
            out[key] = read_csv_flexible(p)
        except Exception:
            pass

    try:
        ensure_package("openpyxl")
        p = download_file(pandax_paths()["data_xlsx"], dest / "PandaX4T_Data_ne.xlsx")
        out["pandax4t_xlsx"] = pd.read_excel(p)
    except Exception:
        pass

    return out


def detect_mass_cross_section_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = list(df.columns)
    mass = first_existing_column(df, ["mass", "mx", "mchi", "wimp_mass", "m_dm", "x"])
    xs = first_existing_column(df, ["sigma", "cross", "limit", "y"])
    return mass, xs


def curve_window_coverage(df: pd.DataFrame, mass_window: tuple[float, float] = (500.0, 3000.0)) -> dict[str, float]:
    mass_col, xs_col = detect_mass_cross_section_columns(df)
    m = pd.to_numeric(df[mass_col], errors="coerce").to_numpy(dtype=float)
    xs = pd.to_numeric(df[xs_col], errors="coerce").to_numpy(dtype=float)
    g = np.isfinite(m) & np.isfinite(xs)
    m = np.sort(m[g])
    if m.size == 0:
        return {"coverage_gev": 0.0, "min_mass": float("nan"), "max_mass": float("nan")}
    low, high = mass_window
    inw = m[(m >= low) & (m <= high)]
    if inw.size == 0:
        return {"coverage_gev": 0.0, "min_mass": float("nan"), "max_mass": float("nan")}
    return {"coverage_gev": float(inw.max() - inw.min()), "min_mass": float(inw.min()), "max_mass": float(inw.max())}


def build_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", type=str, default=None, help="Path to JSON output. Defaults to outputs/<script>.json")
    p.add_argument("--seed", type=int, default=0)
    return p


def finalize_result(script_path: os.PathLike[str] | str, result: dict[str, Any], out_path: str | None = None) -> Path:
    path = Path(out_path) if out_path else get_output_path(script_path)
    write_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return path
