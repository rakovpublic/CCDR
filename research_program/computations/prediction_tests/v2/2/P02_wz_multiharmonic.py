
#!/usr/bin/env python3
"""
P02_wz_multiharmonic.py

Standalone implementation of CCDR v6 Test 02:
multi-harmonic dark-energy EOS search in Pantheon+/BAO data.

Main features
-------------
- Flexible 8-knot spline model for w(z)
- Joint fit of Omega_m, h, spline nodes, and optional SN absolute-magnitude offset M
- Support for Pantheon+-style distance-modulus or magnitude tables
- Support for BAO tables with common observable conventions
- Lomb-Scargle periodogram on reconstructed delta w(z) = w(z) + 1
- Empirical single-trial and look-elsewhere false-alarm probabilities via permutations
- JSON + CSV + PNG outputs
- Optional synthetic demo mode for quick validation of the pipeline

Notes
-----
This script is designed to be robust and usable without touching any existing codebase.
It is intentionally self-contained and conservative about assumptions.

Expected packages:
    numpy, pandas, scipy, matplotlib

Typical usage
-------------
python P02_wz_multiharmonic.py \\
    --pantheon Pantheon+SH0ES.dat \\
    --pantheon-cov Pantheon+SH0ES_STAT+SYS.cov \\
    --bao desi_dr2.csv \\
    --bao boss_eboss.csv \\
    --outdir out_p02

BAO file format
---------------
Each BAO table should contain, at minimum, columns equivalent to:
    z, observable, value, error

Recognized observables (case/spacing/slash insensitive):
    DM/rd, DA/rd, DH/rd, Hrd/c, DV/rd, rd/DV

You can also pass an optional covariance matrix per BAO table using:
    --bao table.csv|cov.npy

Pantheon+ handling
------------------
Preferred columns:
    zHD and MU_SH0ES (or MU / mu) and optionally zHEL

Fallback mode:
    if only m_b_corr is present, the code fits an additive SN magnitude offset M.
    This is practical for generic usage, though not a full reproduction of every
    Pantheon+SH0ES calibration detail.

Author: OpenAI ChatGPT
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import sys
import urllib.request
import urllib.error

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import linalg, optimize, signal
from scipy.interpolate import CubicSpline, interp1d

C_LIGHT = 299792.458  # km/s


# ----------------------------- Utilities ------------------------------------ #

def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def is_url(path_or_url: str) -> bool:
    return path_or_url.startswith("http://") or path_or_url.startswith("https://")


def fetch_bytes(path_or_url: str) -> bytes:
    if is_url(path_or_url):
        with urllib.request.urlopen(path_or_url) as resp:
            return resp.read()
    return Path(path_or_url).read_bytes()


def smart_open_text(path_or_url: str) -> io.StringIO:
    raw = fetch_bytes(path_or_url)
    if path_or_url.lower().endswith(".gz"):
        raw = gzip.decompress(raw)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return io.StringIO(text)


def first_present(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def infer_sep_from_name(name: str) -> Optional[str]:
    lower = name.lower()
    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        return ","
    if lower.endswith(".tsv") or lower.endswith(".tab") or lower.endswith(".tsv.gz"):
        return "\t"
    return None


def read_table(path_or_url: str) -> pd.DataFrame:
    lower = path_or_url.lower()
    if lower.endswith(".json") or lower.endswith(".json.gz"):
        with smart_open_text(path_or_url) as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("data", "rows", "table", "results"):
                if key in payload and isinstance(payload[key], list):
                    return pd.DataFrame(payload[key])
            # last resort: flatten scalar dict
            return pd.DataFrame([payload])
        raise ValueError(f"Unsupported JSON structure in {path_or_url}")

    sep = infer_sep_from_name(path_or_url)
    with smart_open_text(path_or_url) as f:
        if sep is not None:
            return pd.read_csv(f, sep=sep, comment="#")
        # Let pandas sniff whitespace or delimiter
        return pd.read_csv(f, sep=None, engine="python", comment="#")


def read_covariance(path_or_url: str) -> np.ndarray:
    lower = path_or_url.lower()

    if lower.endswith(".npy"):
        data = np.load(io.BytesIO(fetch_bytes(path_or_url)))
        cov = np.asarray(data, dtype=float)
    elif lower.endswith(".npz"):
        blob = np.load(io.BytesIO(fetch_bytes(path_or_url)))
        if len(blob.files) != 1:
            raise ValueError(f"{path_or_url}: expected exactly one array in .npz")
        cov = np.asarray(blob[blob.files[0]], dtype=float)
    else:
        with smart_open_text(path_or_url) as f:
            text = f.read().strip()
        tokens = np.fromstring(text.replace(",", " "), sep=" ")
        if tokens.size < 4:
            raise ValueError(f"Could not parse covariance matrix from {path_or_url}")

        # Pantheon+ .cov often stores n followed by n*n flat values
        n0 = int(round(tokens[0]))
        if tokens.size == 1 + n0 * n0:
            cov = tokens[1:].reshape(n0, n0)
        else:
            n = int(round(math.sqrt(tokens.size)))
            if n * n != tokens.size:
                raise ValueError(
                    f"Covariance in {path_or_url} does not look square after parsing"
                )
            cov = tokens.reshape(n, n)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{path_or_url}: covariance is not square")
    return cov


def sanitize_observable_name(x: str) -> str:
    x = str(x).strip().lower()
    for old, new in [
        (" ", ""),
        ("_", ""),
        ("-", ""),
        ("\\", "/"),
        ("dmoverrd", "dm/rd"),
        ("daoverrd", "da/rd"),
        ("dhoverrd", "dh/rd"),
        ("dvoverrd", "dv/rd"),
        ("rdoverdv", "rd/dv"),
        ("hzrdc", "hrd/c"),
    ]:
        x = x.replace(old, new)
    return x


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_or_fail(arr: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values found in {label}")


def robust_quantile_bins(z: np.ndarray, max_bins: int) -> np.ndarray:
    """Return irregular support points for the periodogram."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    z = np.unique(np.sort(z))
    if z.size <= max_bins:
        return z
    q = np.linspace(0.0, 1.0, max_bins)
    return np.unique(np.quantile(z, q))


PUBLIC_DATA_URLS = {
    "pantheon_table": "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/refs/heads/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
    "pantheon_cov": "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/refs/heads/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov",
    "desi_dr2_mean": "https://raw.githubusercontent.com/dmitrylife/psi-continuum-v2/main/data/desi/dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt",
    "desi_dr2_cov": "https://raw.githubusercontent.com/dmitrylife/psi-continuum-v2/main/data/desi/dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt",
    "sdss_dr12_mean": "https://raw.githubusercontent.com/dmitrylife/psi-continuum-v2/main/data/bao/sdss_DR12Consensus_bao.dat",
    "sdss_dr12_cov": "https://raw.githubusercontent.com/dmitrylife/psi-continuum-v2/main/data/bao/BAO_consensus_covtot_dM_Hz.txt",
}


def fetch_url_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; P02_wz_multiharmonic/1.1)"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download public dataset from {url}: {exc}") from exc


def download_to_cache(url: str, dest: Path, refresh: bool = False, timeout: int = 120) -> Path:
    ensure_dir(dest.parent)
    if dest.exists() and not refresh:
        return dest
    data = fetch_url_bytes(url, timeout=timeout)
    dest.write_bytes(data)
    return dest


def prepare_default_public_data(
    cache_dir: Path,
    refresh: bool = False,
    include_sdss: bool = True,
    pantheon_url: Optional[str] = None,
    pantheon_cov_url: Optional[str] = None,
    desi_mean_url: Optional[str] = None,
    desi_cov_url: Optional[str] = None,
    sdss_mean_url: Optional[str] = None,
    sdss_cov_url: Optional[str] = None,
) -> Dict[str, Path]:
    cache_dir = Path(cache_dir)
    files = {
        "pantheon_table": cache_dir / "pantheon_plus" / "Pantheon+SH0ES.dat",
        "pantheon_cov": cache_dir / "pantheon_plus" / "Pantheon+SH0ES_STAT+SYS.cov",
        "desi_dr2_mean": cache_dir / "desi" / "dr2" / "desi_gaussian_bao_ALL_GCcomb_mean.txt",
        "desi_dr2_cov": cache_dir / "desi" / "dr2" / "desi_gaussian_bao_ALL_GCcomb_cov.txt",
        "sdss_dr12_mean": cache_dir / "sdss" / "sdss_DR12Consensus_bao.dat",
        "sdss_dr12_cov": cache_dir / "sdss" / "BAO_consensus_covtot_dM_Hz.txt",
    }

    url_map = {
        "pantheon_table": pantheon_url or PUBLIC_DATA_URLS["pantheon_table"],
        "pantheon_cov": pantheon_cov_url or PUBLIC_DATA_URLS["pantheon_cov"],
        "desi_dr2_mean": desi_mean_url or PUBLIC_DATA_URLS["desi_dr2_mean"],
        "desi_dr2_cov": desi_cov_url or PUBLIC_DATA_URLS["desi_dr2_cov"],
        "sdss_dr12_mean": sdss_mean_url or PUBLIC_DATA_URLS["sdss_dr12_mean"],
        "sdss_dr12_cov": sdss_cov_url or PUBLIC_DATA_URLS["sdss_dr12_cov"],
    }

    mandatory = ["pantheon_table", "pantheon_cov", "desi_dr2_mean", "desi_dr2_cov"]
    for key in mandatory:
        download_to_cache(url_map[key], files[key], refresh=refresh)

    if include_sdss:
        for key in ["sdss_dr12_mean", "sdss_dr12_cov"]:
            download_to_cache(url_map[key], files[key], refresh=refresh)

    return files


def _iter_data_lines(path_or_url: str) -> Iterable[str]:
    with smart_open_text(path_or_url) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line


def load_desi_dr2_gaussian(mean_path: str, cov_path: str) -> BAOBlock:
    z_vals: List[float] = []
    obs_vals: List[float] = []
    obs_names: List[str] = []

    mapping = {
        "DV_over_rs": "DV/rd",
        "DM_over_rs": "DM/rd",
        "DH_over_rs": "DH/rd",
    }

    for line in _iter_data_lines(mean_path):
        parts = line.split()
        if len(parts) != 3:
            raise RuntimeError(f"Unexpected DESI DR2 mean-file row: {line}")
        z_str, val_str, name = parts
        if name not in mapping:
            raise RuntimeError(f"Unsupported DESI DR2 quantity '{name}' in {mean_path}")
        z_vals.append(float(z_str))
        obs_vals.append(float(val_str))
        obs_names.append(mapping[name])

    cov = read_covariance(cov_path)
    if cov.shape != (len(z_vals), len(z_vals)):
        raise RuntimeError(
            f"DESI DR2 covariance shape {cov.shape} does not match mean vector length {len(z_vals)}"
        )

    return BAOBlock(
        z=np.asarray(z_vals, dtype=float),
        observable=obs_names,
        observed=np.asarray(obs_vals, dtype=float),
        covariance=cov,
        sigma=None,
        label="DESI_DR2_auto",
    )


def load_sdss_dr12_consensus(mean_path: str, cov_path: str, rd_fid_mpc: float = 147.78) -> BAOBlock:
    raw_z: List[float] = []
    raw_obs: List[float] = []
    raw_names: List[str] = []

    for line in _iter_data_lines(mean_path):
        parts = line.split()
        if len(parts) != 3:
            raise RuntimeError(f"Unexpected SDSS DR12 row: {line}")
        z_str, val_str, name = parts
        raw_z.append(float(z_str))
        raw_obs.append(float(val_str))
        raw_names.append(name)

    cov_y = read_covariance(cov_path)
    n = len(raw_obs)
    if cov_y.shape != (n, n):
        raise RuntimeError(
            f"SDSS DR12 covariance shape {cov_y.shape} does not match mean vector length {n}"
        )

    z_vals: List[float] = []
    obs_vals: List[float] = []
    obs_names: List[str] = []
    jac = np.zeros(n, dtype=float)

    for i, (z, y, name) in enumerate(zip(raw_z, raw_obs, raw_names)):
        if name == "DM_over_rs":
            x = y / rd_fid_mpc
            dx_dy = 1.0 / rd_fid_mpc
            obs = "DM/rd"
        elif name == "bao_Hz_rs":
            x = C_LIGHT / (y * rd_fid_mpc)
            dx_dy = -C_LIGHT / (rd_fid_mpc * y * y)
            obs = "DH/rd"
        else:
            raise RuntimeError(f"Unsupported SDSS DR12 quantity '{name}' in {mean_path}")

        z_vals.append(z)
        obs_vals.append(x)
        obs_names.append(obs)
        jac[i] = dx_dy

    cov_x = cov_y * np.outer(jac, jac)

    return BAOBlock(
        z=np.asarray(z_vals, dtype=float),
        observable=obs_names,
        observed=np.asarray(obs_vals, dtype=float),
        covariance=cov_x,
        sigma=None,
        label="SDSS_DR12_auto",
    )


# ----------------------------- Data containers ------------------------------- #

@dataclass
class PantheonData:
    z_cmb: np.ndarray
    z_hel: np.ndarray
    observed: np.ndarray
    covariance: Optional[np.ndarray]
    sigma: Optional[np.ndarray]
    mode: str  # "mu" or "m"
    source: str

    def __len__(self) -> int:
        return self.observed.size


@dataclass
class BAOBlock:
    z: np.ndarray
    observable: List[str]
    observed: np.ndarray
    covariance: Optional[np.ndarray]
    sigma: Optional[np.ndarray]
    label: str

    def __len__(self) -> int:
        return self.observed.size


@dataclass
class FitResult:
    x: np.ndarray
    fun: float
    success: bool
    message: str
    nfev: int
    nit: int


# ----------------------------- Cosmology model ------------------------------- #

class WzSplineCosmology:
    def __init__(
        self,
        z_knots: np.ndarray,
        z_max: float,
        integration_n: int = 2400,
        rd_mpc: float = 147.09,
    ) -> None:
        self.z_knots = np.asarray(z_knots, dtype=float)
        self.z_max = float(z_max)
        self.integration_n = int(integration_n)
        self.rd_mpc = float(rd_mpc)

    def _background(
        self,
        omega_m: float,
        h: float,
        w_nodes: np.ndarray,
        z_pad_factor: float = 1.1,
    ) -> Dict[str, object]:
        if not (0.0 < omega_m < 1.0):
            raise ValueError("omega_m must be in (0,1)")
        if not (0.2 < h < 1.2):
            raise ValueError("h must be in (0.2,1.2)")

        z_top = max(self.z_max, self.z_knots.max()) * z_pad_factor + 0.05
        z_grid = np.linspace(0.0, z_top, self.integration_n)

        spline = CubicSpline(self.z_knots, w_nodes, bc_type="natural", extrapolate=True)
        w_grid = spline(z_grid)

        integrand = 3.0 * (1.0 + w_grid) / (1.0 + z_grid)
        integral = np.zeros_like(z_grid)
        dz = np.diff(z_grid)
        integral[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dz)
        rho_de = np.exp(integral)

        ez2 = omega_m * (1.0 + z_grid) ** 3 + (1.0 - omega_m) * rho_de
        if np.any(ez2 <= 0.0):
            raise ValueError("Encountered non-positive E(z)^2 during integration")
        e_grid = np.sqrt(ez2)

        chi_grid = np.zeros_like(z_grid)
        chi_grid[1:] = np.cumsum(0.5 * (1.0 / e_grid[1:] + 1.0 / e_grid[:-1]) * dz)

        return {
            "z_grid": z_grid,
            "w_grid": w_grid,
            "e_interp": interp1d(z_grid, e_grid, kind="linear", bounds_error=False, fill_value="extrapolate"),
            "chi_interp": interp1d(z_grid, chi_grid, kind="linear", bounds_error=False, fill_value="extrapolate"),
            "spline": spline,
            "H0": 100.0 * h,
        }

    def e_of_z(self, bg: Dict[str, object], z: np.ndarray) -> np.ndarray:
        return np.asarray(bg["e_interp"](np.asarray(z, dtype=float)), dtype=float)

    def chi_of_z(self, bg: Dict[str, object], z: np.ndarray) -> np.ndarray:
        return np.asarray(bg["chi_interp"](np.asarray(z, dtype=float)), dtype=float)

    def dm_mpc(self, bg: Dict[str, object], z: np.ndarray) -> np.ndarray:
        return (C_LIGHT / bg["H0"]) * self.chi_of_z(bg, z)

    def da_mpc(self, bg: Dict[str, object], z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return self.dm_mpc(bg, z) / (1.0 + z)

    def dh_mpc(self, bg: Dict[str, object], z: np.ndarray) -> np.ndarray:
        return C_LIGHT / (bg["H0"] * self.e_of_z(bg, z))

    def dv_mpc(self, bg: Dict[str, object], z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        dm = self.dm_mpc(bg, z)
        dh = self.dh_mpc(bg, z)
        return np.cbrt(z * dm * dm * dh)

    def distance_modulus(self, bg: Dict[str, object], z_cmb: np.ndarray, z_hel: np.ndarray) -> np.ndarray:
        z_cmb = np.asarray(z_cmb, dtype=float)
        z_hel = np.asarray(z_hel, dtype=float)
        d_l_mpc = (1.0 + z_hel) * self.dm_mpc(bg, z_cmb)
        return 5.0 * np.log10(np.maximum(d_l_mpc, 1e-12)) + 25.0

    def predict_bao(self, bg: Dict[str, object], z: np.ndarray, observable: Sequence[str]) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z, dtype=float)
        for i, obs in enumerate(observable):
            name = sanitize_observable_name(obs)
            if name in ("dm/rd", "dmrd", "dM/rd".lower()):
                out[i] = self.dm_mpc(bg, z[i]) / self.rd_mpc
            elif name in ("da/rd", "dard"):
                out[i] = self.da_mpc(bg, z[i]) / self.rd_mpc
            elif name in ("dh/rd", "dhrd"):
                out[i] = self.dh_mpc(bg, z[i]) / self.rd_mpc
            elif name in ("hrd/c", "h(z)rd/c", "hzrd/c"):
                out[i] = 1.0 / (self.dh_mpc(bg, z[i]) / self.rd_mpc)
            elif name in ("dv/rd", "dvrd"):
                out[i] = self.dv_mpc(bg, z[i]) / self.rd_mpc
            elif name in ("rd/dv", "rddv"):
                out[i] = self.rd_mpc / self.dv_mpc(bg, z[i])
            else:
                raise ValueError(
                    f"Unsupported BAO observable '{obs}'. Supported: DM/rd, DA/rd, DH/rd, Hrd/c, DV/rd, rd/DV"
                )
        return out


# ----------------------------- Data loading ---------------------------------- #

def load_pantheon(path: str, cov_path: Optional[str] = None) -> PantheonData:
    df = read_table(path)

    z_cmb_col = first_present(df, ["zHD", "z_cmb", "zcmb", "z"])
    if z_cmb_col is None:
        raise RuntimeError(f"{path}: could not detect a CMB-frame redshift column")

    z_hel_col = first_present(df, ["zHEL", "zhel"])
    if z_hel_col is None:
        z_hel = df[z_cmb_col].to_numpy(dtype=float)
    else:
        z_hel = df[z_hel_col].to_numpy(dtype=float)

    mu_col = first_present(df, ["MU_SH0ES", "MU", "mu", "distance_modulus", "mu_obs"])
    mag_col = first_present(df, ["m_b_corr", "mb", "mB", "mbcorr"])
    err_col = first_present(df, ["MUERR", "muerr", "dmu", "mu_err", "sigma_mu", "mb_err", "m_b_corr_err"])

    if mu_col is not None:
        observed = df[mu_col].to_numpy(dtype=float)
        mode = "mu"
    elif mag_col is not None:
        observed = df[mag_col].to_numpy(dtype=float)
        mode = "m"
    else:
        raise RuntimeError(
            f"{path}: could not detect MU/MU_SH0ES or m_b_corr-like SN observable columns"
        )

    sigma = None
    cov = None
    if cov_path is not None:
        cov = read_covariance(cov_path)
        if cov.shape != (len(df), len(df)):
            raise RuntimeError(
                f"Pantheon covariance shape {cov.shape} does not match table length {len(df)}"
            )
    elif err_col is not None:
        sigma = df[err_col].to_numpy(dtype=float)
    else:
        raise RuntimeError(
            f"{path}: no covariance supplied and no diagonal error column detected"
        )

    z_cmb = df[z_cmb_col].to_numpy(dtype=float)

    finite_or_fail(z_cmb, "Pantheon z_cmb")
    finite_or_fail(z_hel, "Pantheon z_hel")
    finite_or_fail(observed, "Pantheon observed")
    if sigma is not None:
        finite_or_fail(sigma, "Pantheon sigma")

    return PantheonData(
        z_cmb=z_cmb,
        z_hel=z_hel,
        observed=observed,
        covariance=cov,
        sigma=sigma,
        mode=mode,
        source=path,
    )


def split_bao_arg(arg: str) -> Tuple[str, Optional[str]]:
    if "|" in arg:
        a, b = arg.split("|", 1)
        return a.strip(), b.strip()
    return arg.strip(), None


def load_bao_block(arg: str) -> BAOBlock:
    table_path, cov_path = split_bao_arg(arg)
    df = read_table(table_path)

    z_col = first_present(df, ["z", "redshift", "zeff", "z_eff"])
    obs_col = first_present(df, ["observable", "quantity", "kind", "type", "obs"])
    val_col = first_present(df, ["value", "measurement", "mean", "y"])
    err_col = first_present(df, ["error", "sigma", "err", "unc", "uncertainty", "yerr"])

    if z_col is None or obs_col is None or val_col is None:
        raise RuntimeError(
            f"{table_path}: BAO table must contain z, observable, value columns (or common aliases)"
        )

    z = df[z_col].to_numpy(dtype=float)
    observable = [str(x) for x in df[obs_col].tolist()]
    observed = df[val_col].to_numpy(dtype=float)

    cov = None
    sigma = None
    if cov_path is not None:
        cov = read_covariance(cov_path)
        if cov.shape != (len(df), len(df)):
            raise RuntimeError(
                f"{table_path}: covariance shape {cov.shape} does not match row count {len(df)}"
            )
    elif err_col is not None:
        sigma = df[err_col].to_numpy(dtype=float)
    else:
        raise RuntimeError(
            f"{table_path}: no covariance supplied and no diagonal error column detected"
        )

    finite_or_fail(z, f"BAO z in {table_path}")
    finite_or_fail(observed, f"BAO observed in {table_path}")
    if sigma is not None:
        finite_or_fail(sigma, f"BAO sigma in {table_path}")

    return BAOBlock(
        z=z,
        observable=observable,
        observed=observed,
        covariance=cov,
        sigma=sigma,
        label=Path(table_path).name,
    )


# ----------------------------- Linear algebra -------------------------------- #

def chi2_from_residual(residual: np.ndarray, cov: Optional[np.ndarray], sigma: Optional[np.ndarray]) -> float:
    residual = np.asarray(residual, dtype=float)
    if cov is not None:
        try:
            cfac, lower = linalg.cho_factor(cov, overwrite_a=False, check_finite=False)
            sol = linalg.cho_solve((cfac, lower), residual, check_finite=False)
            return float(residual @ sol)
        except linalg.LinAlgError:
            # fallback: pseudo-inverse
            pinv = np.linalg.pinv(cov)
            return float(residual @ pinv @ residual)
    if sigma is None:
        raise ValueError("Either covariance or sigma must be provided")
    return float(np.sum((residual / sigma) ** 2))


# ----------------------------- Fitter ---------------------------------------- #

class MultiHarmonicWzFitter:
    def __init__(
        self,
        pantheon: PantheonData,
        bao_blocks: Sequence[BAOBlock],
        z_knots: np.ndarray,
        smooth_prior_sigma: float = 0.20,
        rd_mpc: float = 147.09,
        integration_n: int = 2400,
    ) -> None:
        self.pantheon = pantheon
        self.bao_blocks = list(bao_blocks)
        self.z_knots = np.asarray(z_knots, dtype=float)
        self.cosmo = WzSplineCosmology(
            z_knots=self.z_knots,
            z_max=max(
                float(np.max(self.pantheon.z_cmb)),
                max((float(np.max(b.z)) for b in self.bao_blocks), default=0.0),
                float(np.max(self.z_knots)),
            ),
            integration_n=integration_n,
            rd_mpc=rd_mpc,
        )
        self.smooth_prior_sigma = float(smooth_prior_sigma)
        self.has_m_offset = (pantheon.mode == "m")

    def unpack(self, theta: np.ndarray) -> Tuple[float, float, np.ndarray, Optional[float]]:
        omega_m = float(theta[0])
        h = float(theta[1])
        n = self.z_knots.size
        w_nodes = np.asarray(theta[2 : 2 + n], dtype=float)
        M = float(theta[2 + n]) if self.has_m_offset else None
        return omega_m, h, w_nodes, M

    def objective(self, theta: np.ndarray) -> float:
        omega_m, h, w_nodes, M = self.unpack(theta)

        try:
            bg = self.cosmo._background(omega_m=omega_m, h=h, w_nodes=w_nodes)
        except Exception:
            return 1e100

        chi2 = 0.0

        mu = self.cosmo.distance_modulus(bg, self.pantheon.z_cmb, self.pantheon.z_hel)
        if self.pantheon.mode == "m":
            pred = mu + float(M)
        else:
            pred = mu
        r_sn = self.pantheon.observed - pred
        chi2 += chi2_from_residual(r_sn, self.pantheon.covariance, self.pantheon.sigma)

        for block in self.bao_blocks:
            pred_bao = self.cosmo.predict_bao(bg, block.z, block.observable)
            r_bao = block.observed - pred_bao
            chi2 += chi2_from_residual(r_bao, block.covariance, block.sigma)

        # Smoothness penalty on second differences of the spline nodes
        d2 = np.diff(w_nodes, n=2)
        chi2 += float(np.sum((d2 / self.smooth_prior_sigma) ** 2))

        # Soft prior to keep the highest-z node near plausible range and avoid wild extrapolation
        chi2 += float(np.sum(((w_nodes + 1.0) / 1.5) ** 4)) * 0.01

        if not np.isfinite(chi2):
            return 1e100
        return chi2

    def initial_theta(self) -> np.ndarray:
        w0 = np.full(self.z_knots.size, -1.0, dtype=float)
        parts = [np.array([0.30, 0.70], dtype=float), w0]
        if self.has_m_offset:
            parts.append(np.array([-19.3], dtype=float))
        return np.concatenate(parts)

    def bounds(self) -> List[Tuple[float, float]]:
        bnds: List[Tuple[float, float]] = [(0.05, 0.60), (0.50, 0.90)]
        bnds.extend([(-2.5, 0.5)] * self.z_knots.size)
        if self.has_m_offset:
            bnds.append((-20.5, -18.0))
        return bnds

    def fit(self, multistart: int = 8, seed: int = 1234) -> FitResult:
        rng = np.random.default_rng(seed)
        bounds = self.bounds()
        theta0 = self.initial_theta()

        best = None

        for i in range(multistart):
            start = theta0.copy()
            if i > 0:
                start[0] = np.clip(start[0] + rng.normal(0.0, 0.04), *bounds[0])
                start[1] = np.clip(start[1] + rng.normal(0.0, 0.03), *bounds[1])
                start[2 : 2 + self.z_knots.size] += rng.normal(0.0, 0.10, self.z_knots.size)
                for j in range(2, 2 + self.z_knots.size):
                    start[j] = np.clip(start[j], *bounds[j])
                if self.has_m_offset:
                    start[-1] = np.clip(start[-1] + rng.normal(0.0, 0.2), *bounds[-1])

            res = optimize.minimize(
                self.objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=1200, ftol=1e-12, maxls=50),
            )

            candidate = FitResult(
                x=np.asarray(res.x, dtype=float),
                fun=float(res.fun),
                success=bool(res.success),
                message=str(res.message),
                nfev=int(res.nfev),
                nit=int(getattr(res, "nit", -1)),
            )
            if best is None or candidate.fun < best.fun:
                best = candidate

        assert best is not None
        return best


# ----------------------------- Periodogram ----------------------------------- #

def fit_sinusoid_amplitude(z: np.ndarray, y: np.ndarray, freq_cycles_per_z: float) -> float:
    omega = 2.0 * np.pi * freq_cycles_per_z
    X = np.column_stack([np.cos(omega * z), np.sin(omega * z), np.ones_like(z)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta[0], beta[1]
    return float(np.hypot(a, b))


def analyze_periodogram(
    z: np.ndarray,
    y: np.ndarray,
    freq_min: float,
    freq_max: float,
    n_freq: int,
    n_permutations: int,
    random_seed: int,
) -> Dict[str, object]:
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)

    freqs = np.linspace(freq_min, freq_max, n_freq)
    ang = 2.0 * np.pi * freqs
    power = signal.lombscargle(z, y, ang, normalize=True)

    # Empirical null from permutations of y across z
    rng = np.random.default_rng(random_seed)
    null_power = np.empty((n_permutations, n_freq), dtype=float)
    null_max = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        yp = rng.permutation(y)
        p = signal.lombscargle(z, yp, ang, normalize=True)
        null_power[i] = p
        null_max[i] = np.max(p)

    peaks, props = signal.find_peaks(power)
    peak_records: List[Dict[str, float]] = []
    for idx in peaks:
        p_obs = float(power[idx])
        p_single = float(np.mean(null_power[:, idx] >= p_obs))
        p_global = float(np.mean(null_max >= p_obs))
        amp = fit_sinusoid_amplitude(z, y, float(freqs[idx]))
        peak_records.append(
            {
                "index": int(idx),
                "frequency": float(freqs[idx]),
                "period_in_z": float(1.0 / freqs[idx]) if freqs[idx] > 0 else float("inf"),
                "power": p_obs,
                "amplitude": amp,
                "fap_single_trial": p_single,
                "fap_global": p_global,
            }
        )

    peak_records.sort(key=lambda d: d["power"], reverse=True)

    significant = [
        p for p in peak_records
        if p["fap_single_trial"] < 0.01 and p["fap_global"] < 0.10
    ]

    out: Dict[str, object] = {
        "frequencies": freqs,
        "power": power,
        "null_power": null_power,
        "null_max": null_max,
        "peaks": peak_records,
        "significant_peaks": significant,
        "single_trial_1pct_line": np.quantile(null_power, 0.99, axis=0),
        "global_10pct_line": float(np.quantile(null_max, 0.90)),
    }
    return out


# ----------------------------- Demo mode ------------------------------------- #

def generate_demo_data(outdir: Path, rd_mpc: float, seed: int = 7) -> Tuple[PantheonData, List[BAOBlock]]:
    rng = np.random.default_rng(seed)

    z_knots = np.linspace(0.0, 2.5, 8)
    cosmo = WzSplineCosmology(z_knots=z_knots, z_max=2.5, rd_mpc=rd_mpc)

    z_sn = np.sort(np.concatenate([
        rng.uniform(0.01, 0.15, 180),
        rng.uniform(0.15, 0.8, 320),
        rng.uniform(0.8, 2.2, 180),
    ]))
    zhel = z_sn + rng.normal(0.0, 0.0007, size=z_sn.size)

    # Two small harmonics, close to the target idea
    w_nodes = -1.0 + 0.010 * np.cos(2.0 * np.pi * z_knots / 1.1 + 0.4) + 0.007 * np.cos(2.0 * np.pi * z_knots / 0.48 - 0.7)
    bg = cosmo._background(omega_m=0.305, h=0.692, w_nodes=w_nodes)
    mu_true = cosmo.distance_modulus(bg, z_sn, zhel)
    sn_sigma = np.full_like(z_sn, 0.10)
    mu_obs = mu_true + rng.normal(0.0, sn_sigma)

    pantheon = PantheonData(
        z_cmb=z_sn,
        z_hel=zhel,
        observed=mu_obs,
        covariance=None,
        sigma=sn_sigma,
        mode="mu",
        source="demo_synthetic",
    )

    bao_specs = [
        (0.38, "DM/rd", 0.020),
        (0.38, "DH/rd", 0.028),
        (0.51, "DM/rd", 0.018),
        (0.51, "DH/rd", 0.025),
        (0.70, "DM/rd", 0.020),
        (0.70, "DH/rd", 0.030),
        (1.10, "DV/rd", 0.030),
        (1.48, "DH/rd", 0.050),
        (1.48, "DM/rd", 0.040),
        (2.33, "DH/rd", 0.065),
    ]
    z_bao = np.array([x[0] for x in bao_specs], dtype=float)
    obs_kind = [x[1] for x in bao_specs]
    frac_err = np.array([x[2] for x in bao_specs], dtype=float)
    bao_true = cosmo.predict_bao(bg, z_bao, obs_kind)
    bao_sigma = frac_err * bao_true
    bao_obs = bao_true + rng.normal(0.0, bao_sigma)

    bao_block = BAOBlock(
        z=z_bao,
        observable=obs_kind,
        observed=bao_obs,
        covariance=None,
        sigma=bao_sigma,
        label="demo_bao",
    )

    ensure_dir(outdir)
    pd.DataFrame({"zHD": z_sn, "zHEL": zhel, "MU": mu_obs, "MUERR": sn_sigma}).to_csv(outdir / "demo_pantheon.csv", index=False)
    pd.DataFrame({"z": z_bao, "observable": obs_kind, "value": bao_obs, "error": bao_sigma}).to_csv(outdir / "demo_bao.csv", index=False)

    return pantheon, [bao_block]


# ----------------------------- Plotting -------------------------------------- #

def make_fit_plot(
    outpath: Path,
    z_plot: np.ndarray,
    w_plot: np.ndarray,
    z_support: np.ndarray,
    delta_support: np.ndarray,
    z_knots: np.ndarray,
    w_nodes: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(8.6, 7.0))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(z_plot, w_plot, lw=2, label="Best-fit w(z)")
    ax1.axhline(-1.0, ls="--", lw=1, label=r"$w_\Lambda=-1$")
    ax1.scatter(z_knots, w_nodes, s=24, zorder=3, label="Spline knots")
    ax1.set_ylabel("w(z)")
    ax1.set_xlim(z_plot.min(), z_plot.max())
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(z_support, delta_support, lw=2, label=r"$\Delta w(z)=w(z)+1$")
    ax2.axhline(0.0, ls="--", lw=1)
    ax2.set_xlabel("z")
    ax2.set_ylabel(r"$\Delta w$")
    ax2.set_xlim(z_plot.min(), z_plot.max())
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def make_periodogram_plot(
    outpath: Path,
    freqs: np.ndarray,
    power: np.ndarray,
    single_line: np.ndarray,
    global_line: float,
    significant_peaks: Sequence[Dict[str, float]],
) -> None:
    fig = plt.figure(figsize=(8.6, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(freqs, power, lw=2, label="Lomb-Scargle power")
    ax.plot(freqs, single_line, lw=1, ls="--", label="1% single-trial line")
    ax.axhline(global_line, lw=1, ls=":", label="10% global FAP line")

    for peak in significant_peaks:
        ax.axvline(peak["frequency"], ls="--", lw=1)
        ax.annotate(
            f"f={peak['frequency']:.3g}",
            xy=(peak["frequency"], peak["power"]),
            xytext=(4, 8),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Frequency [cycles per unit redshift]")
    ax.set_ylabel("Normalized power")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# ----------------------------- Main workflow --------------------------------- #

def build_support_grid(
    pantheon: PantheonData,
    bao_blocks: Sequence[BAOBlock],
    z_max: float,
    max_points: int = 70,
) -> np.ndarray:
    z_all = [pantheon.z_cmb]
    for block in bao_blocks:
        z_all.append(block.z)
    z = np.concatenate(z_all)
    z = z[(z >= 0.0) & (z <= z_max)]
    support = robust_quantile_bins(z, max_bins=max_points)
    if support[0] > 0.0:
        support = np.insert(support, 0, 0.0)
    if support[-1] < z_max:
        support = np.append(support, z_max)
    return np.unique(support)


def save_result_json(outpath: Path, payload: Dict[str, object]) -> None:
    class Encoder(json.JSONEncoder):
        def default(self, obj):  # type: ignore[override]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return super().default(obj)

    outpath.write_text(json.dumps(payload, indent=2, cls=Encoder))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CCDR v6 Test 02: multi-harmonic w(z) search with Pantheon+/BAO"
    )
    p.add_argument("--pantheon", help="Pantheon+/SN data table path or URL. If omitted together with --bao, the script downloads public defaults automatically.")
    p.add_argument("--pantheon-cov", help="Pantheon covariance path or URL")
    p.add_argument(
        "--bao",
        action="append",
        default=[],
        help="BAO table path or URL, optionally 'table|cov'. Repeatable. If omitted together with --pantheon, public default BAO inputs are downloaded automatically.",
    )
    p.add_argument("--data-cache-dir", default="p02_public_data", help="Local cache directory for auto-downloaded public datasets")
    p.add_argument("--refresh-data", action="store_true", help="Force re-download of public datasets in auto mode")
    p.add_argument("--skip-default-sdss", action="store_true", help="In auto-download mode, do not include the SDSS DR12 consensus BAO supplement")
    p.add_argument("--pantheon-url", help="Override public Pantheon+ table URL used in auto mode")
    p.add_argument("--pantheon-cov-url", help="Override public Pantheon+ covariance URL used in auto mode")
    p.add_argument("--desi-mean-url", help="Override public DESI DR2 mean-file URL used in auto mode")
    p.add_argument("--desi-cov-url", help="Override public DESI DR2 covariance URL used in auto mode")
    p.add_argument("--sdss-mean-url", help="Override public SDSS DR12 mean-file URL used in auto mode")
    p.add_argument("--sdss-cov-url", help="Override public SDSS DR12 covariance URL used in auto mode")
    p.add_argument("--outdir", default="out_p02", help="Output directory")
    p.add_argument("--rd", type=float, default=147.09, help="Sound horizon r_d in Mpc")
    p.add_argument("--n-knots", type=int, default=8, help="Number of spline knots in z")
    p.add_argument("--z-max", type=float, default=2.5, help="Maximum redshift for spline support")
    p.add_argument(
        "--smooth-prior-sigma",
        type=float,
        default=0.20,
        help="Gaussian sigma for second-difference smoothness penalty on w nodes",
    )
    p.add_argument("--multistart", type=int, default=8, help="Number of optimizer restarts")
    p.add_argument("--integration-n", type=int, default=2400, help="Integration grid points")
    p.add_argument("--period-min", type=float, help="Minimum period in redshift units for LS search")
    p.add_argument("--period-max", type=float, help="Maximum period in redshift units for LS search")
    p.add_argument("--n-freq", type=int, default=1000, help="Number of scanned frequencies")
    p.add_argument("--n-permutations", type=int, default=600, help="Permutation count for FAP")
    p.add_argument("--periodogram-points", type=int, default=70, help="Support points for delta w(z)")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run on internally generated synthetic data instead of external files",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    if args.demo:
        pantheon, bao_blocks = generate_demo_data(outdir=outdir, rd_mpc=args.rd, seed=args.seed)
        eprint("[demo] Generated synthetic Pantheon/BAO tables in", outdir)
    else:
        manual_mode = bool(args.pantheon or args.pantheon_cov or args.bao)
        if manual_mode:
            if not args.pantheon:
                raise SystemExit("Manual mode requires --pantheon when any manual dataset arguments are provided")
            if not args.bao:
                raise SystemExit("Manual mode requires at least one --bao when manual dataset arguments are provided")
            pantheon = load_pantheon(args.pantheon, args.pantheon_cov)
            bao_blocks = [load_bao_block(x) for x in args.bao]
            eprint("[data] Using manually supplied Pantheon/BAO inputs")
        else:
            data_files = prepare_default_public_data(
                cache_dir=Path(args.data_cache_dir),
                refresh=args.refresh_data,
                include_sdss=not args.skip_default_sdss,
                pantheon_url=args.pantheon_url,
                pantheon_cov_url=args.pantheon_cov_url,
                desi_mean_url=args.desi_mean_url,
                desi_cov_url=args.desi_cov_url,
                sdss_mean_url=args.sdss_mean_url,
                sdss_cov_url=args.sdss_cov_url,
            )
            pantheon = load_pantheon(str(data_files["pantheon_table"]), str(data_files["pantheon_cov"]))
            bao_blocks = [
                load_desi_dr2_gaussian(
                    str(data_files["desi_dr2_mean"]),
                    str(data_files["desi_dr2_cov"]),
                )
            ]
            if not args.skip_default_sdss:
                bao_blocks.append(
                    load_sdss_dr12_consensus(
                        str(data_files["sdss_dr12_mean"]),
                        str(data_files["sdss_dr12_cov"]),
                    )
                )
            eprint("[data] Downloaded public Pantheon+/DESI inputs into", Path(args.data_cache_dir))
            if not args.skip_default_sdss:
                eprint("[data] Included public SDSS DR12 consensus BAO supplement")

    z_knots = np.linspace(0.0, args.z_max, args.n_knots)
    fitter = MultiHarmonicWzFitter(
        pantheon=pantheon,
        bao_blocks=bao_blocks,
        z_knots=z_knots,
        smooth_prior_sigma=args.smooth_prior_sigma,
        rd_mpc=args.rd,
        integration_n=args.integration_n,
    )

    eprint("[fit] Starting optimization...")
    fit = fitter.fit(multistart=args.multistart, seed=args.seed)
    omega_m, h, w_nodes, M = fitter.unpack(fit.x)
    bg = fitter.cosmo._background(omega_m=omega_m, h=h, w_nodes=w_nodes)

    z_plot = np.linspace(0.0, args.z_max, 400)
    w_plot = np.asarray(bg["spline"](z_plot), dtype=float)

    support_z = build_support_grid(
        pantheon=pantheon,
        bao_blocks=bao_blocks,
        z_max=args.z_max,
        max_points=args.periodogram_points,
    )
    delta_support = np.asarray(bg["spline"](support_z), dtype=float) + 1.0

    baseline = args.z_max
    period_max = args.period_max if args.period_max is not None else baseline
    period_min = args.period_min if args.period_min is not None else baseline / 10.0

    if period_min <= 0 or period_max <= 0:
        raise SystemExit("Periods must be positive")
    if period_min >= period_max:
        raise SystemExit("--period-min must be smaller than --period-max")

    freq_min = 1.0 / period_max
    freq_max = 1.0 / period_min

    eprint("[scan] Running Lomb-Scargle search...")
    pgram = analyze_periodogram(
        z=support_z,
        y=delta_support,
        freq_min=freq_min,
        freq_max=freq_max,
        n_freq=args.n_freq,
        n_permutations=args.n_permutations,
        random_seed=args.seed,
    )

    significant = list(pgram["significant_peaks"])
    n_sig = len(significant)

    inferred_n_minus_4 = int(n_sig)
    pass_v6 = bool(n_sig >= 2)
    pass_v5_only = bool(n_sig == 0)

    # Optional closest pair ratio among significant peaks
    ratio_info = None
    if len(significant) >= 2:
        sig_freqs = np.array([p["frequency"] for p in significant], dtype=float)
        best_pair = None
        best_dist = np.inf
        for i in range(len(sig_freqs)):
            for j in range(i + 1, len(sig_freqs)):
                ratio = max(sig_freqs[i], sig_freqs[j]) / min(sig_freqs[i], sig_freqs[j])
                # nearest simple harmonic-like ratio among 2:1, 3:2, sqrt(2), phi
                targets = np.array([2.0, 1.5, np.sqrt(2.0), (1.0 + np.sqrt(5.0)) / 2.0])
                dist = float(np.min(np.abs(np.log(ratio / targets))))
                if dist < best_dist:
                    best_dist = dist
                    best_pair = {
                        "ratio": float(ratio),
                        "log_distance_to_simple_ratio": dist,
                        "frequencies": [float(sig_freqs[i]), float(sig_freqs[j])],
                    }
        ratio_info = best_pair

    result = {
        "test_name": "CCDR v6 P7-extended multi-harmonic w(z) search",
        "pantheon_source": pantheon.source,
        "bao_sources": [b.label for b in bao_blocks],
        "auto_download_mode": (not args.demo) and (not bool(args.pantheon or args.pantheon_cov or args.bao)),
        "public_data_cache_dir": str(Path(args.data_cache_dir).resolve()) if ((not args.demo) and (not bool(args.pantheon or args.pantheon_cov or args.bao))) else None,
        "fit": {
            "success": fit.success,
            "message": fit.message,
            "chi2_total": fit.fun,
            "nfev": fit.nfev,
            "nit": fit.nit,
            "omega_m": omega_m,
            "h": h,
            "w_nodes": w_nodes.tolist(),
            "z_knots": z_knots.tolist(),
            "M_if_fit": M,
        },
        "periodogram": {
            "support_z": support_z.tolist(),
            "delta_w_support": delta_support.tolist(),
            "frequency_min": freq_min,
            "frequency_max": freq_max,
            "n_freq": args.n_freq,
            "n_permutations": args.n_permutations,
            "single_trial_1pct_global_summary": float(np.quantile(pgram["single_trial_1pct_line"], 0.5)),
            "look_elsewhere_10pct_line": float(pgram["global_10pct_line"]),
            "peaks_all": pgram["peaks"],
            "peaks_significant": significant,
            "closest_frequency_ratio": ratio_info,
        },
        "result": {
            "n_peaks_significant": n_sig,
            "peak_frequencies": [float(p["frequency"]) for p in significant],
            "peak_amplitudes": [float(p["amplitude"]) for p in significant],
            "inferred_N_minus_4": inferred_n_minus_4,
            "pass_v6": pass_v6,
            "pass_v5_only": pass_v5_only,
        },
    }

    save_result_json(outdir / "result.json", result)

    pd.DataFrame(
        {
            "z": support_z,
            "delta_w": delta_support,
            "w": delta_support - 1.0,
        }
    ).to_csv(outdir / "residual_grid.csv", index=False)

    pd.DataFrame(
        {
            "frequency": pgram["frequencies"],
            "power": pgram["power"],
            "single_trial_1pct_line": pgram["single_trial_1pct_line"],
        }
    ).to_csv(outdir / "periodogram.csv", index=False)

    make_fit_plot(
        outpath=outdir / "wz_fit.png",
        z_plot=z_plot,
        w_plot=w_plot,
        z_support=support_z,
        delta_support=delta_support,
        z_knots=z_knots,
        w_nodes=w_nodes,
    )
    make_periodogram_plot(
        outpath=outdir / "periodogram.png",
        freqs=np.asarray(pgram["frequencies"]),
        power=np.asarray(pgram["power"]),
        single_line=np.asarray(pgram["single_trial_1pct_line"]),
        global_line=float(pgram["global_10pct_line"]),
        significant_peaks=significant,
    )

    eprint("[done] Wrote:")
    for name in ["result.json", "residual_grid.csv", "periodogram.csv", "wz_fit.png", "periodogram.png"]:
        eprint("   ", outdir / name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
