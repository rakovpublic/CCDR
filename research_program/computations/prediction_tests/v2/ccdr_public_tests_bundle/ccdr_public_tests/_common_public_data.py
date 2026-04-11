
#!/usr/bin/env python3
"""
Shared utilities for CCDR public-data tests.

These helpers favor robustness over elegance:
- Every dataset is downloaded from one or more public URLs with retries.
- Heavy public files are cached locally under ./data_cache/.
- The cosmology routines are lightweight and intentionally avoid CAMB/CLASS.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import time
import zipfile
import random
import hashlib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import linalg, optimize, signal, spatial, stats

C_LIGHT = 299792.458  # km/s
T_CMB = 2.7255
H100_SI = 100.0 * 1000.0 / 3.085677581491367e22  # s^-1


DATA_CACHE = Path(__file__).resolve().parent / "data_cache"
DATA_CACHE.mkdir(exist_ok=True)


def _default_headers() -> Dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; CCDRPublicTests/1.0; +https://openai.com)",
        "Accept": "*/*",
    }


def download_bytes(
    urls: Sequence[str],
    cache_name: str,
    timeout: int = 120,
    force: bool = False,
) -> bytes:
    """
    Download a remote resource from the first working URL and cache it locally.
    """
    target = DATA_CACHE / cache_name
    if target.exists() and not force:
        return target.read_bytes()

    last_error: Optional[Exception] = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=_default_headers())
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            if len(payload) == 0:
                raise RuntimeError(f"empty response from {url}")
            target.write_bytes(payload)
            return payload
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            continue

    raise RuntimeError(f"Failed to download {cache_name} from any public URL") from last_error


def download_text(
    urls: Sequence[str],
    cache_name: str,
    timeout: int = 120,
    force: bool = False,
    encoding: str = "utf-8",
) -> str:
    return download_bytes(urls, cache_name=cache_name, timeout=timeout, force=force).decode(
        encoding, errors="replace"
    )


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def positive_float(x: str) -> float:
    val = float(x)
    if not np.isfinite(val) or val <= 0:
        raise ValueError(f"Expected positive float, got {x}")
    return val


# ---------------------------------------------------------------------------
# Public datasets
# ---------------------------------------------------------------------------

PANTHEON_DAT_URLS = [
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat",
]
PANTHEON_COV_URLS = [
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov",
]

DESI_DR2_MEAN_URLS = [
    "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt",
]
DESI_DR2_COV_URLS = [
    "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt",
]

PLANCK_PR3_CHAINS_URLS = [
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip",
    "https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip",
]

SPARC_RAR_URLS = [
    "https://astroweb.case.edu/SPARC/RAR.mrt",
    "http://astroweb.case.edu/SPARC/RAR.mrt",
]
SPARC_ROTMOD_URLS = [
    "https://astroweb.case.edu/SPARC/Rotmod_LTG.zip",
    "http://astroweb.case.edu/SPARC/Rotmod_LTG.zip",
]
SPARC_TABLE1_URLS = [
    "https://astroweb.case.edu/SPARC/Table1.mrt",
    "http://astroweb.case.edu/SPARC/Table1.mrt",
]

PDG_LEPTON_PDF_URLS = [
    "https://pdg.lbl.gov/2024/tables/rpp2024-sum-leptons.pdf",
]
PDG_QUARK_PDF_URLS = [
    "https://pdg.lbl.gov/2024/tables/rpp2024-sum-quarks.pdf",
]

NSA_FITS_URLS = [
    "https://data.sdss.org/sas/dr17/sdss/atlas/v1/nsa_v1_0_1.fits",
]

SDSS_SKYSERVER_SQL_BASES = [
    "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch",
    "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch",
]

# Public direct-detection comparison datasets.
DIRECT_DETECTION_CURVE_URLS = {
    "LZ_4p2tyr_SI": [
        "https://www.hepdata.net/record/155182?format=json",
        "https://www.hepdata.net/record/ins2841863?format=json",
    ],
    "CMS_Fig8_LZ": [
        "https://www.hepdata.net/record/ins3085605?version=2&format=json",
        "https://www.hepdata.net/record/166403?format=json",
    ],
}


# ---------------------------------------------------------------------------
# Lightweight cosmology
# ---------------------------------------------------------------------------

def omega_radiation(h: float) -> float:
    # Includes photons + effectively massless neutrinos in a standard approximation.
    omega_gamma = 2.469e-5 / (h * h) * (T_CMB / 2.7255) ** 4
    return omega_gamma * (1.0 + 0.2271 * 3.046)


def e2_rvm_flat(z: np.ndarray | float, omega_m: float, nu: float, include_radiation: bool = True, h: float = 0.7) -> np.ndarray:
    """
    A compact, late-time flat-RVM-like parameterization.
    This is not the full official RVM implementation; it is a lightweight public-data audit model.
    """
    z = np.asarray(z, dtype=float)
    if not np.all(1.0 + z > 0):
        raise ValueError("z must satisfy 1+z>0")
    om_eff = max(1e-8, min(0.999, omega_m))
    denom = max(1e-6, 1.0 - nu)
    matter = om_eff / denom * ((1.0 + z) ** (3.0 * (1.0 - nu)) - 1.0)
    e2 = 1.0 + matter
    if include_radiation:
        orad = omega_radiation(h)
        e2 = e2 + orad * ((1.0 + z) ** 4 - 1.0)
    return np.clip(e2, 1e-12, None)


def hz_rvm(z: np.ndarray | float, h0: float, omega_m: float, nu: float, include_radiation: bool = True) -> np.ndarray:
    return h0 * np.sqrt(e2_rvm_flat(z, omega_m=omega_m, nu=nu, include_radiation=include_radiation, h=h0 / 100.0))


def comoving_distance_mpc(z: np.ndarray, h0: float, omega_m: float, nu: float, include_radiation: bool = True, nsteps: int = 256) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    zmax = float(np.max(z))
    grid = np.linspace(0.0, zmax, max(nsteps, 8))
    integrand = C_LIGHT / hz_rvm(grid, h0=h0, omega_m=omega_m, nu=nu, include_radiation=include_radiation)
    chi_grid = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))])
    return np.interp(z, grid, chi_grid)


def luminosity_distance_mpc(z: np.ndarray, h0: float, omega_m: float, nu: float, include_radiation: bool = True) -> np.ndarray:
    chi = comoving_distance_mpc(z, h0=h0, omega_m=omega_m, nu=nu, include_radiation=include_radiation)
    return (1.0 + np.asarray(z)) * chi


def distance_modulus(z: np.ndarray, h0: float, omega_m: float, nu: float, include_radiation: bool = True) -> np.ndarray:
    dl = np.clip(luminosity_distance_mpc(z, h0=h0, omega_m=omega_m, nu=nu, include_radiation=include_radiation), 1e-8, None)
    return 5.0 * np.log10(dl) + 25.0


def desi_bao_predictions(
    z: np.ndarray,
    quantity: Sequence[str],
    h0: float,
    omega_m: float,
    nu: float,
    rd_mpc: float,
    include_radiation: bool = True,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    dm = comoving_distance_mpc(z, h0=h0, omega_m=omega_m, nu=nu, include_radiation=include_radiation)
    dh = C_LIGHT / hz_rvm(z, h0=h0, omega_m=omega_m, nu=nu, include_radiation=include_radiation)
    dv = ((z * dm * dm * dh) ** (1.0 / 3.0))
    out = []
    for zi, qi, dmi, dhi, dvi in zip(z, quantity, dm, dh, dv):
        if qi == "DM_over_rs":
            out.append(dmi / rd_mpc)
        elif qi == "DH_over_rs":
            out.append(dhi / rd_mpc)
        elif qi == "DV_over_rs":
            out.append(dvi / rd_mpc)
        else:
            raise ValueError(f"Unsupported BAO quantity: {qi}")
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Pantheon+ / DESI / Planck loaders
# ---------------------------------------------------------------------------

def load_pantheon_plus(use_calibrators: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
    txt = download_text(PANTHEON_DAT_URLS, "Pantheon+SH0ES.dat")
    cov_txt = download_text(PANTHEON_COV_URLS, "Pantheon+SH0ES_STAT+SYS.cov")
    df = pd.read_csv(io.StringIO(txt), delim_whitespace=True, comment="#")
    cov = np.loadtxt(io.StringIO(cov_txt))
    # The covariance is for the whole file ordering.
    if not use_calibrators:
        mask = (df["IS_CALIBRATOR"].astype(int) == 0).to_numpy()
        df = df.loc[mask].reset_index(drop=True)
        cov = cov[np.ix_(mask, mask)]
    return df, cov


def load_desi_dr2(diagonal_only: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
    mean_txt = download_text(DESI_DR2_MEAN_URLS, "desi_gaussian_bao_ALL_GCcomb_mean.txt")
    cov_txt = download_text(DESI_DR2_COV_URLS, "desi_gaussian_bao_ALL_GCcomb_cov.txt")
    rows: List[Tuple[float, float, str]] = []
    for line in mean_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        z, value, quantity = line.split()
        rows.append((float(z), float(value), str(quantity)))
    df = pd.DataFrame(rows, columns=["z", "value", "quantity"])
    cov = np.loadtxt(io.StringIO(cov_txt))
    if diagonal_only:
        cov = np.diag(np.diag(cov))
    return df, cov


def load_planck_rd_prior() -> Dict[str, float]:
    """
    Derive a Gaussian prior for rdrag from the public Planck PR3 chain archive.
    """
    payload = download_bytes(PLANCK_PR3_CHAINS_URLS, "COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip", timeout=300)
    zf = zipfile.ZipFile(io.BytesIO(payload))
    names = zf.namelist()

    param_file = None
    for name in names:
        lower = name.lower()
        if lower.endswith(".paramnames") and "base_plikhm" in lower:
            param_file = name
            break
    if param_file is None:
        raise RuntimeError("Could not find .paramnames in Planck chain zip")

    paramnames = []
    for line in zf.read(param_file).decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paramnames.append(line.split()[0])

    target_name = None
    for candidate in ["rdrag", "r_drag", "rsdrag", "rs_drag"]:
        for p in paramnames:
            if p.lower() == candidate:
                target_name = p
                break
        if target_name is not None:
            break

    if target_name is None:
        # Try a fuzzy fallback.
        for p in paramnames:
            pl = p.lower()
            if "drag" in pl and ("r" in pl or "rs" in pl):
                target_name = p
                break
    if target_name is None:
        raise RuntimeError(f"Planck chain does not expose an r_drag-like parameter. Available: {paramnames[:25]}")

    idx = paramnames.index(target_name)
    chain_arrays = []
    weight_arrays = []
    for name in names:
        lower = name.lower()
        if lower.endswith(".txt") and "base_plikhm" in lower and not lower.endswith("minimum.theory_cl"):
            text = zf.read(name).decode("utf-8", errors="replace")
            if not text.strip():
                continue
            arr = np.loadtxt(io.StringIO(text))
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[1] < idx + 3:
                continue
            weights = arr[:, 0]
            values = arr[:, idx + 2]  # cols: weight, -loglike, params...
            good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            if np.any(good):
                chain_arrays.append(values[good])
                weight_arrays.append(weights[good])

    if not chain_arrays:
        raise RuntimeError("No usable Planck chain tables found inside zip")

    values = np.concatenate(chain_arrays)
    weights = np.concatenate(weight_arrays)
    mean = np.average(values, weights=weights)
    var = np.average((values - mean) ** 2, weights=weights)
    std = float(np.sqrt(max(var, 1e-12)))
    return {
        "parameter": target_name,
        "mean": float(mean),
        "sigma": std,
        "n_samples": int(values.size),
    }


@dataclass
class NuFitResult:
    h0: float
    omega_m: float
    nu: float
    rd_mpc: float
    intercept: float
    chi2: float
    success: bool
    message: str
    n_sne: int
    n_bao: int
    include_sn: bool
    include_bao: bool
    include_planck: bool
    analytic_intercept: bool
    include_radiation: bool
    diagonal_bao: bool


def _analytic_intercept_shift(residual: np.ndarray, inv_cov: np.ndarray) -> Tuple[float, float]:
    """
    For chi2 = (r - a 1)^T C^-1 (r - a 1), return best-fit a and minimized chi2.
    """
    ones = np.ones_like(residual)
    alpha = float(ones @ inv_cov @ ones)
    beta = float(ones @ inv_cov @ residual)
    a = beta / alpha
    rmin = residual - a * ones
    chi2 = float(rmin @ inv_cov @ rmin)
    return a, chi2


def fit_nu_model(
    include_sn: bool = True,
    include_bao: bool = True,
    include_planck: bool = True,
    analytic_intercept: bool = True,
    include_radiation: bool = True,
    diagonal_bao: bool = False,
    use_calibrators: bool = False,
) -> NuFitResult:
    sn_df: Optional[pd.DataFrame] = None
    sn_cov: Optional[np.ndarray] = None
    inv_sn: Optional[np.ndarray] = None
    if include_sn:
        sn_df, sn_cov = load_pantheon_plus(use_calibrators=use_calibrators)
        inv_sn = linalg.inv(sn_cov)

    bao_df: Optional[pd.DataFrame] = None
    bao_cov: Optional[np.ndarray] = None
    inv_bao: Optional[np.ndarray] = None
    if include_bao:
        bao_df, bao_cov = load_desi_dr2(diagonal_only=diagonal_bao)
        inv_bao = linalg.inv(bao_cov)

    planck_prior = load_planck_rd_prior() if include_planck else None

    def chi2(theta: np.ndarray) -> float:
        if analytic_intercept:
            h0, omega_m, nu, rd = theta
            intercept = 0.0
        else:
            h0, omega_m, nu, rd, intercept = theta

        if not (40.0 < h0 < 95.0 and 0.05 < omega_m < 0.6 and -0.05 < nu < 0.05 and 110.0 < rd < 170.0):
            return 1e30

        total = 0.0

        if include_sn and sn_df is not None and inv_sn is not None:
            z = sn_df["zHD"].to_numpy(dtype=float)
            mu_obs = sn_df["MU_SH0ES"].to_numpy(dtype=float)
            mu_th = distance_modulus(z, h0=h0, omega_m=omega_m, nu=nu, include_radiation=include_radiation)
            residual = mu_obs - mu_th
            if analytic_intercept:
                _, chi2_sn = _analytic_intercept_shift(residual, inv_sn)
            else:
                r = residual - intercept
                chi2_sn = float(r @ inv_sn @ r)
            total += chi2_sn

        if include_bao and bao_df is not None and inv_bao is not None:
            pred = desi_bao_predictions(
                bao_df["z"].to_numpy(float),
                bao_df["quantity"].tolist(),
                h0=h0,
                omega_m=omega_m,
                nu=nu,
                rd_mpc=rd,
                include_radiation=include_radiation,
            )
            r = bao_df["value"].to_numpy(float) - pred
            total += float(r @ inv_bao @ r)

        if include_planck and planck_prior is not None:
            total += ((rd - planck_prior["mean"]) / planck_prior["sigma"]) ** 2

        return float(total)

    if analytic_intercept:
        x0 = np.array([67.5, 0.31, 0.001, 147.0], dtype=float)
        bounds = [(50.0, 90.0), (0.1, 0.5), (-0.03, 0.03), (130.0, 160.0)]
    else:
        x0 = np.array([67.5, 0.31, 0.001, 147.0, 0.0], dtype=float)
        bounds = [(50.0, 90.0), (0.1, 0.5), (-0.03, 0.03), (130.0, 160.0), (-1.0, 1.0)]

    best = optimize.minimize(chi2, x0=x0, method="L-BFGS-B", bounds=bounds)
    if analytic_intercept:
        h0, omega_m, nu, rd = best.x
        intercept = 0.0
        if include_sn and sn_df is not None and inv_sn is not None:
            z = sn_df["zHD"].to_numpy(dtype=float)
            mu_obs = sn_df["MU_SH0ES"].to_numpy(dtype=float)
            mu_th = distance_modulus(z, h0=h0, omega_m=omega_m, nu=nu, include_radiation=include_radiation)
            intercept, _ = _analytic_intercept_shift(mu_obs - mu_th, inv_sn)
    else:
        h0, omega_m, nu, rd, intercept = best.x

    return NuFitResult(
        h0=float(h0),
        omega_m=float(omega_m),
        nu=float(nu),
        rd_mpc=float(rd),
        intercept=float(intercept),
        chi2=float(best.fun),
        success=bool(best.success),
        message=str(best.message),
        n_sne=0 if sn_df is None else int(len(sn_df)),
        n_bao=0 if bao_df is None else int(len(bao_df)),
        include_sn=include_sn,
        include_bao=include_bao,
        include_planck=include_planck,
        analytic_intercept=analytic_intercept,
        include_radiation=include_radiation,
        diagonal_bao=diagonal_bao,
    )


def fit_nu_model_fixed_nu(
    fixed_nu: float,
    include_sn: bool = True,
    include_bao: bool = True,
    include_planck: bool = True,
    analytic_intercept: bool = True,
    include_radiation: bool = True,
    diagonal_bao: bool = False,
    use_calibrators: bool = False,
) -> NuFitResult:
    sn_df: Optional[pd.DataFrame] = None
    sn_cov: Optional[np.ndarray] = None
    inv_sn: Optional[np.ndarray] = None
    if include_sn:
        sn_df, sn_cov = load_pantheon_plus(use_calibrators=use_calibrators)
        inv_sn = linalg.inv(sn_cov)

    bao_df: Optional[pd.DataFrame] = None
    bao_cov: Optional[np.ndarray] = None
    inv_bao: Optional[np.ndarray] = None
    if include_bao:
        bao_df, bao_cov = load_desi_dr2(diagonal_only=diagonal_bao)
        inv_bao = linalg.inv(bao_cov)

    planck_prior = load_planck_rd_prior() if include_planck else None

    def chi2(theta: np.ndarray) -> float:
        if analytic_intercept:
            h0, omega_m, rd = theta
            intercept = 0.0
        else:
            h0, omega_m, rd, intercept = theta
        nu = fixed_nu

        if not (40.0 < h0 < 95.0 and 0.05 < omega_m < 0.6 and -0.05 < nu < 0.05 and 110.0 < rd < 170.0):
            return 1e30

        total = 0.0

        if include_sn and sn_df is not None and inv_sn is not None:
            z = sn_df["zHD"].to_numpy(dtype=float)
            mu_obs = sn_df["MU_SH0ES"].to_numpy(dtype=float)
            mu_th = distance_modulus(z, h0=h0, omega_m=omega_m, nu=nu, include_radiation=include_radiation)
            residual = mu_obs - mu_th
            if analytic_intercept:
                _, chi2_sn = _analytic_intercept_shift(residual, inv_sn)
            else:
                r = residual - intercept
                chi2_sn = float(r @ inv_sn @ r)
            total += chi2_sn

        if include_bao and bao_df is not None and inv_bao is not None:
            pred = desi_bao_predictions(
                bao_df["z"].to_numpy(float),
                bao_df["quantity"].tolist(),
                h0=h0,
                omega_m=omega_m,
                nu=nu,
                rd_mpc=rd,
                include_radiation=include_radiation,
            )
            r = bao_df["value"].to_numpy(float) - pred
            total += float(r @ inv_bao @ r)

        if include_planck and planck_prior is not None:
            total += ((rd - planck_prior["mean"]) / planck_prior["sigma"]) ** 2

        return float(total)

    if analytic_intercept:
        x0 = np.array([67.5, 0.31, 147.0], dtype=float)
        bounds = [(50.0, 90.0), (0.1, 0.5), (130.0, 160.0)]
    else:
        x0 = np.array([67.5, 0.31, 147.0, 0.0], dtype=float)
        bounds = [(50.0, 90.0), (0.1, 0.5), (130.0, 160.0), (-1.0, 1.0)]

    best = optimize.minimize(chi2, x0=x0, method="L-BFGS-B", bounds=bounds)
    if analytic_intercept:
        h0, omega_m, rd = best.x
        intercept = 0.0
        if include_sn and sn_df is not None and inv_sn is not None:
            z = sn_df["zHD"].to_numpy(dtype=float)
            mu_obs = sn_df["MU_SH0ES"].to_numpy(dtype=float)
            mu_th = distance_modulus(z, h0=h0, omega_m=omega_m, nu=fixed_nu, include_radiation=include_radiation)
            intercept, _ = _analytic_intercept_shift(mu_obs - mu_th, inv_sn)
    else:
        h0, omega_m, rd, intercept = best.x

    return NuFitResult(
        h0=float(h0),
        omega_m=float(omega_m),
        nu=float(fixed_nu),
        rd_mpc=float(rd),
        intercept=float(intercept),
        chi2=float(best.fun),
        success=bool(best.success),
        message=str(best.message),
        n_sne=0 if sn_df is None else int(len(sn_df)),
        n_bao=0 if bao_df is None else int(len(bao_df)),
        include_sn=include_sn,
        include_bao=include_bao,
        include_planck=include_planck,
        analytic_intercept=analytic_intercept,
        include_radiation=include_radiation,
        diagonal_bao=diagonal_bao,
    )


def nu_significance_from_delta_chi2(best_fit: NuFitResult, null_fit: NuFitResult) -> float:
    return float(np.sqrt(max(0.0, null_fit.chi2 - best_fit.chi2)))


# ---------------------------------------------------------------------------
# SDSS downloader/query for filament tests
# ---------------------------------------------------------------------------

def query_sdss_csv(sql: str, cache_name: str, timeout: int = 300, force: bool = False) -> pd.DataFrame:
    cached = DATA_CACHE / cache_name

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [str(c).strip().lstrip('#').strip().lower() for c in out.columns]
        aliases = {
            'ra': 'ra', 'dec': 'dec', 'z': 'z',
            'p.ra': 'ra', 'p.dec': 'dec', 's.z': 'z',
            'specobj.z': 'z', 'photoobj.ra': 'ra', 'photoobj.dec': 'dec',
        }
        out = out.rename(columns={c: aliases.get(c, c) for c in out.columns})
        # Drop obviously bogus single-column marker tables like '#Table1'.
        if list(out.columns) in [['table1'], ['#table1']] or (len(out.columns) == 1 and str(out.columns[0]).startswith('table')):
            raise RuntimeError(f'SkyServer returned a table marker instead of data: {list(out.columns)}')
        needed = {'ra', 'dec', 'z'}
        if not needed.issubset(set(out.columns)):
            raise RuntimeError(f'SDSS query returned unexpected columns: {list(out.columns)}')
        for c in ['ra', 'dec', 'z']:
            out[c] = pd.to_numeric(out[c], errors='coerce')
        out = out.dropna(subset=['ra', 'dec', 'z'])
        if len(out) == 0:
            raise RuntimeError('SkyServer returned zero usable rows after numeric coercion')
        return out[['ra', 'dec', 'z']]

    if cached.exists() and not force:
        try:
            return _clean(pd.read_csv(cached))
        except Exception:
            try:
                cached.unlink()
            except Exception:
                pass

    last_error: Optional[Exception] = None
    for base in SDSS_SKYSERVER_SQL_BASES:
        try:
            params = {
                "cmd": sql,
                "format": "csv",
            }
            url = base + "?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=_default_headers())
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            if "error" in raw.lower() and "html" in raw.lower():
                raise RuntimeError("SkyServer returned an HTML error page")
            # Some SkyServer responses include comment-prefixed metadata/header markers.
            df = pd.read_csv(io.StringIO(raw), comment='#')
            df = _clean(df)
            df.to_csv(cached, index=False)
            return df
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            continue
    raise RuntimeError(f"SDSS SkyServer query failed for {cache_name}") from last_error


def fetch_sdss_galaxy_sample(
    ra_ranges: Sequence[Tuple[float, float]] = ((120.0, 160.0), (160.0, 200.0), (200.0, 240.0)),
    dec_range: Tuple[float, float] = (0.0, 60.0),
    z_range: Tuple[float, float] = (0.02, 0.14),
    top_per_chunk: int = 15000,
) -> pd.DataFrame:
    """
    Download a moderate public SDSS spectroscopic galaxy sample via SkyServer.
    """
    pieces: List[pd.DataFrame] = []
    for i, (ra_min, ra_max) in enumerate(ra_ranges):
        sql = f"""
        SELECT TOP {int(top_per_chunk)}
            p.ra, p.dec, s.z
        FROM PhotoObj AS p
        JOIN SpecObj AS s ON p.objID = s.bestObjID
        WHERE p.type = 3
          AND p.clean = 1
          AND s.class = 'GALAXY'
          AND s.zWarning = 0
          AND s.z BETWEEN {z_range[0]} AND {z_range[1]}
          AND p.ra BETWEEN {ra_min} AND {ra_max}
          AND p.dec BETWEEN {dec_range[0]} AND {dec_range[1]}
        ORDER BY s.z
        """
        df = query_sdss_csv(sql, cache_name=f"sdss_filament_chunk_{i}.csv")
        pieces.append(df)

    all_df = pd.concat(pieces, ignore_index=True)
    all_df = all_df.drop_duplicates().reset_index(drop=True)
    return all_df


def sky_to_cartesian_mpc(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    z: np.ndarray,
    h0: float = 70.0,
    omega_m: float = 0.3,
) -> np.ndarray:
    chi = comoving_distance_mpc(np.asarray(z, dtype=float), h0=h0, omega_m=omega_m, nu=0.0, include_radiation=False)
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    x = chi * np.cos(dec) * np.cos(ra)
    y = chi * np.cos(dec) * np.sin(ra)
    zc = chi * np.sin(dec)
    return np.column_stack([x, y, zc])


def local_filament_axes(points_xyz: np.ndarray, k: int = 12) -> np.ndarray:
    tree = spatial.cKDTree(points_xyz)
    dists, idx = tree.query(points_xyz, k=min(k + 1, len(points_xyz)))
    axes = np.zeros_like(points_xyz)
    for i in range(len(points_xyz)):
        neigh = points_xyz[idx[i, 1:]]
        if len(neigh) < 3:
            axes[i] = np.array([1.0, 0.0, 0.0])
            continue
        centered = neigh - neigh.mean(axis=0)
        cov = centered.T @ centered / max(1, len(neigh) - 1)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, np.argmax(vals)]
        axes[i] = axis / np.linalg.norm(axis)
    return axes


def filament_orientation_correlation(
    points_xyz: np.ndarray,
    axes: np.ndarray,
    nbins: int = 16,
    r_min: float = 5.0,
    r_max: float = 250.0,
    max_pairs: int = 200_000,
    seed: int = 1234,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(points_xyz)
    if n < 4:
        raise ValueError("Need at least 4 points")
    bins = np.geomspace(r_min, r_max, nbins + 1)
    pair_i = rng.integers(0, n, size=max_pairs)
    pair_j = rng.integers(0, n, size=max_pairs)
    good = pair_i != pair_j
    pair_i = pair_i[good]
    pair_j = pair_j[good]

    sep = np.linalg.norm(points_xyz[pair_i] - points_xyz[pair_j], axis=1)
    dots = np.abs(np.sum(axes[pair_i] * axes[pair_j], axis=1))
    corr = dots**2 - (1.0 / 3.0)

    centers = np.sqrt(bins[:-1] * bins[1:])
    values = np.full(nbins, np.nan)
    errors = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)

    for b in range(nbins):
        m = (sep >= bins[b]) & (sep < bins[b + 1])
        counts[b] = int(np.sum(m))
        if counts[b] >= 12:
            values[b] = float(np.mean(corr[m]))
            errors[b] = float(np.std(corr[m], ddof=1) / np.sqrt(counts[b]))

    return {
        "r_bins": bins.tolist(),
        "r_centers": centers.tolist(),
        "correlation": values.tolist(),
        "correlation_err": errors.tolist(),
        "counts": counts.tolist(),
    }


def fit_exponential_correlation(r: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(r) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    r = np.asarray(r)[mask]
    y = np.asarray(y)[mask]
    yerr = np.asarray(yerr)[mask]
    if len(r) < 4:
        return {"amplitude": float("nan"), "scale_mpc_h": float("nan"), "chi2": float("nan")}

    def model(rr: np.ndarray, amp: float, scale: float) -> np.ndarray:
        return amp * np.exp(-rr / scale)

    def chi2(theta: np.ndarray) -> float:
        amp, scale = theta
        if scale <= 0:
            return 1e30
        res = (y - model(r, amp, scale)) / yerr
        return float(np.sum(res**2))

    x0 = np.array([max(np.nanmax(y), 1e-4), 150.0])
    best = optimize.minimize(chi2, x0=x0, method="Nelder-Mead")
    amp, scale = best.x
    return {"amplitude": float(amp), "scale_mpc_h": float(scale), "chi2": float(best.fun)}


# ---------------------------------------------------------------------------
# SPARC / RAR helpers
# ---------------------------------------------------------------------------

def load_sparc_rar() -> pd.DataFrame:
    """
    Read the public SPARC RAR machine-readable table.
    Requires astropy for CDS/MRT parsing.
    """
    try:
        from astropy.io import ascii
    except Exception as exc:  # pragma: no cover - dependency dependent
        raise RuntimeError("This script needs astropy to parse SPARC .mrt files. Install with: pip install astropy") from exc

    raw = download_text(SPARC_RAR_URLS, "SPARC_RAR.mrt", timeout=120)
    # astropy.ascii.cds expects a filename, a raw string, or an iterable of lines;
    # StringIO can fail on some astropy versions. Pass the raw text directly.
    table = ascii.read(raw, format="cds")
    df = table.to_pandas()
    return df


def rar_relation(gbar: np.ndarray, a0: float) -> np.ndarray:
    x = np.sqrt(np.clip(gbar / a0, 1e-12, None))
    return gbar / (1.0 - np.exp(-x))


# ---------------------------------------------------------------------------
# Direct-detection helpers
# ---------------------------------------------------------------------------

def _hepdata_try_record_json(urls: Sequence[str], cache_name: str) -> Dict[str, Any]:
    text = download_text(urls, cache_name=cache_name, timeout=180)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"HEPData JSON download failed for {cache_name}") from exc


def _extract_hepdata_table_links(record_json: Dict[str, Any]) -> List[str]:
    links: List[str] = []
    if isinstance(record_json, dict):
        for key in ["data_tables", "tables"]:
            if key in record_json and isinstance(record_json[key], list):
                for item in record_json[key]:
                    if isinstance(item, dict):
                        for field in ["data_file", "table", "location", "name", "processed_name", "doi"]:
                            val = item.get(field)
                            if isinstance(val, str):
                                links.append(val)
                        data_block = item.get('data')
                        if isinstance(data_block, dict):
                            for v in data_block.values():
                                if isinstance(v, str):
                                    links.append(v)
                        for res in item.get("resources", []) if isinstance(item.get("resources"), list) else []:
                            if isinstance(res, dict):
                                for v in res.values():
                                    if isinstance(v, str):
                                        links.append(v)
        rec = record_json.get('record')
        if isinstance(rec, dict):
            access = rec.get('access_urls')
            if isinstance(access, dict):
                links_block = access.get('links')
                if isinstance(links_block, dict):
                    for v in links_block.values():
                        if isinstance(v, str):
                            links.append(v)
            for res in rec.get('resources', []) if isinstance(rec.get('resources'), list) else []:
                if isinstance(res, dict):
                    for v in res.values():
                        if isinstance(v, str):
                            links.append(v)
    return list(dict.fromkeys(links))


def smooth_curve_peak_scan(mass: np.ndarray, y: np.ndarray, window: int = 15, polyorder: int = 3) -> Dict[str, Any]:
    mass = np.asarray(mass, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(mass)
    mass = mass[order]
    y = y[order]
    logm = np.log10(mass)
    logy = np.log10(y)

    if len(logy) < window:
        window = max(5, len(logy) // 2 * 2 + 1)
    if window % 2 == 0:
        window += 1
    window = min(window, len(logy) - (1 - len(logy) % 2))
    window = max(window, 5)

    smooth = signal.savgol_filter(logy, window_length=window, polyorder=min(polyorder, window - 2), mode="interp")
    resid = logy - smooth
    peak_idx, props = signal.find_peaks(np.abs(resid), prominence=np.std(resid) * 1.2 if len(resid) > 10 else 0.0)
    peak_masses = mass[peak_idx]
    peak_strengths = np.abs(resid[peak_idx])

    ratios = []
    if len(peak_masses) >= 2:
        ratios = (peak_masses[1:] / peak_masses[:-1]).tolist()

    return {
        "n_peaks": int(len(peak_masses)),
        "peak_masses_gev": [float(x) for x in peak_masses],
        "peak_strengths_log10": [float(x) for x in peak_strengths],
        "consecutive_ratios": [float(x) for x in ratios],
        "residual_std": float(np.std(resid)),
    }


# ---------------------------------------------------------------------------
# PDG / Koide helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - dependency dependent
        raise RuntimeError("This script needs pypdf to parse PDG PDFs. Install with: pip install pypdf") from exc
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        texts.append(txt)
    return "\n".join(texts)


def parse_pdg_lepton_masses() -> Dict[str, float]:
    text = extract_text_from_pdf(download_bytes(PDG_LEPTON_PDF_URLS, "pdg_2024_leptons.pdf", timeout=120))
    patterns = {
        "e": r"Mass m = ([0-9.]+) ± [0-9.]+ MeV",
        "mu": r"Mass m = 105\.6583755 ± [0-9.]+ MeV",
        "tau": r"τ\s*J.*?Mass m = ([0-9.]+) ± ([0-9.]+) MeV",
    }
    m = {}
    first = re.findall(patterns["e"], text)
    if not first:
        raise RuntimeError("Could not parse electron mass from PDG lepton PDF")
    m["e_mev"] = float(first[0])
    mu_match = re.search(patterns["mu"], text)
    if not mu_match:
        # Fallback looser pattern.
        mu_match = re.search(r"Mass m = ([0-9.]+) ± [0-9.]+ MeV", text[text.find("µ"):])
    if not mu_match:
        raise RuntimeError("Could not parse muon mass from PDG lepton PDF")
    m["mu_mev"] = float(mu_match.group(1))
    tau_match = re.search(patterns["tau"], text, flags=re.DOTALL)
    if not tau_match:
        tau_match = re.search(r"τ.*?Mass m = ([0-9.]+) ± ([0-9.]+) MeV", text, flags=re.DOTALL)
    if not tau_match:
        raise RuntimeError("Could not parse tau mass from PDG lepton PDF")
    m["tau_mev"] = float(tau_match.group(1))
    return m


def parse_pdg_quark_masses() -> Dict[str, float]:
    text = extract_text_from_pdf(download_bytes(PDG_QUARK_PDF_URLS, "pdg_2024_quarks.pdf", timeout=120))
    patterns = {
        "u": r"mu = ([0-9.]+) ± [0-9.]+ MeV",
        "d": r"md = ([0-9.]+) ± [0-9.]+ MeV",
        "s": r"ms = ([0-9.]+) ± [0-9.]+ MeV",
        "c": r"mc = ([0-9.]+) ± [0-9.]+ GeV",
        "b": r"mb = ([0-9.]+) ± [0-9.]+ GeV",
        "t": r"Mass \(direct measurements\) m = ([0-9.]+) ± [0-9.]+ GeV",
    }
    out = {}
    for k, pat in patterns.items():
        hit = re.search(pat, text)
        if not hit:
            raise RuntimeError(f"Could not parse {k}-quark mass from PDG quark PDF")
        val = float(hit.group(1))
        if k in {"u", "d", "s"}:
            out[f"{k}_gev"] = val / 1000.0
        else:
            out[f"{k}_gev"] = val
    return out


def koide_q(masses: Sequence[float]) -> float:
    m = np.asarray(masses, dtype=float)
    return float(np.sum(m) / np.sum(np.sqrt(m)) ** 2)


def alpha_s_one_loop(mu_gev: float, nf: int = 5, lambda_qcd_gev: float = 0.2) -> float:
    beta0 = 11.0 - 2.0 * nf / 3.0
    t = np.log(max(mu_gev, lambda_qcd_gev * 1.01) ** 2 / lambda_qcd_gev ** 2)
    return 4.0 * math.pi / (beta0 * t)


def running_mass_one_loop(
    m_ref_gev: float,
    mu_ref_gev: float,
    mu_target_gev: np.ndarray,
    nf: int,
    gamma0: float = 4.0,
) -> np.ndarray:
    mu_target_gev = np.asarray(mu_target_gev, dtype=float)
    beta0 = 11.0 - 2.0 * nf / 3.0
    a_ref = alpha_s_one_loop(mu_ref_gev, nf=nf)
    a_t = np.asarray([alpha_s_one_loop(mu, nf=nf) for mu in mu_target_gev])
    power = gamma0 / beta0
    return m_ref_gev * (a_t / a_ref) ** power
