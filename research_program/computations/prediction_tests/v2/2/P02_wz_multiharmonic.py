
#!/usr/bin/env python3
"""
P02_wz_multiharmonic.py

Oscillation-only rerun of CCDR Test 02.

Model:
    w(z) = -1 + sum_k [c_k cos(2*pi*f_k*z) + s_k sin(2*pi*f_k*z)]

The script downloads public Pantheon+SH0ES and BAO inputs automatically unless
manual paths are provided. It fits models with 0..K harmonics, compares them by
BIC/AIC, and reports whether the six-harmonic claim survives under this stricter
oscillation-only basis. If not, Test 02 is marked null.

Outputs:
    result.json
    model_selection.csv
    best_harmonics.csv
    six_harmonics.csv
    residual_grid.csv
    wz_fit.png
    model_selection.png
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import linalg, optimize, stats
from scipy.interpolate import interp1d

C_LIGHT = 299792.458  # km/s


# ----------------------------- utilities ------------------------------------ #

def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_url(path_or_url: str) -> bool:
    return path_or_url.startswith("http://") or path_or_url.startswith("https://")


def fetch_bytes(path_or_url: str, timeout: int = 120) -> bytes:
    if is_url(path_or_url):
        req = urllib.request.Request(
            path_or_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; P02_oscillation_only/2.0)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to download public dataset from {path_or_url}: {exc}") from exc
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


def infer_sep(name: str) -> Optional[str]:
    lower = name.lower()
    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        return ","
    if lower.endswith(".tsv") or lower.endswith(".tab") or lower.endswith(".tsv.gz"):
        return "\t"
    return None


def read_table(path_or_url: str) -> pd.DataFrame:
    with smart_open_text(path_or_url) as f:
        sep = infer_sep(path_or_url)
        if sep is not None:
            return pd.read_csv(f, sep=sep, comment="#")
        return pd.read_csv(f, sep=None, engine="python", comment="#")


def read_covariance(path_or_url: str) -> np.ndarray:
    lower = path_or_url.lower()
    if lower.endswith(".npy"):
        return np.asarray(np.load(io.BytesIO(fetch_bytes(path_or_url))), dtype=float)
    if lower.endswith(".npz"):
        blob = np.load(io.BytesIO(fetch_bytes(path_or_url)))
        if len(blob.files) != 1:
            raise ValueError(f"{path_or_url}: expected one array in npz")
        return np.asarray(blob[blob.files[0]], dtype=float)

    with smart_open_text(path_or_url) as f:
        text = f.read().strip()
    tokens = np.fromstring(text.replace(",", " "), sep=" ")
    if tokens.size < 4:
        raise ValueError(f"Could not parse covariance matrix from {path_or_url}")
    n0 = int(round(tokens[0]))
    if tokens.size == 1 + n0 * n0:
        cov = tokens[1:].reshape(n0, n0)
    else:
        n = int(round(math.sqrt(tokens.size)))
        if n * n != tokens.size:
            raise ValueError(f"{path_or_url}: covariance is not square after parsing")
        cov = tokens.reshape(n, n)
    return np.asarray(cov, dtype=float)


def first_present(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    m = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in m:
            return m[name.lower()]
    return None


def finite_or_fail(arr: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values found in {label}")


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


def robust_quantile_bins(z: np.ndarray, max_bins: int) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = np.unique(np.sort(z[np.isfinite(z)]))
    if z.size <= max_bins:
        return z
    return np.unique(np.quantile(z, np.linspace(0.0, 1.0, max_bins)))


# ----------------------------- public data ---------------------------------- #

PUBLIC_DATA_URLS = {
    "pantheon_table": "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/refs/heads/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
    "pantheon_cov": "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/refs/heads/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov",
    "desi_dr2_mean": "https://raw.githubusercontent.com/dmitrylife/psi-continuum-v2/main/data/desi/dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt",
    "desi_dr2_cov": "https://raw.githubusercontent.com/dmitrylife/psi-continuum-v2/main/data/desi/dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt",
    "sdss_dr12_mean": "https://raw.githubusercontent.com/dmitrylife/psi-continuum-v2/main/data/bao/sdss_DR12Consensus_bao.dat",
    "sdss_dr12_cov": "https://raw.githubusercontent.com/dmitrylife/psi-continuum-v2/main/data/bao/BAO_consensus_covtot_dM_Hz.txt",
}


def download_to_cache(url: str, dest: Path, refresh: bool = False) -> Path:
    ensure_dir(dest.parent)
    if dest.exists() and not refresh:
        return dest
    dest.write_bytes(fetch_bytes(url))
    return dest


def prepare_default_public_data(cache_dir: Path, refresh: bool = False, include_sdss: bool = True) -> Dict[str, Path]:
    files = {
        "pantheon_table": cache_dir / "pantheon_plus" / "Pantheon+SH0ES.dat",
        "pantheon_cov": cache_dir / "pantheon_plus" / "Pantheon+SH0ES_STAT+SYS.cov",
        "desi_dr2_mean": cache_dir / "desi" / "dr2" / "desi_gaussian_bao_ALL_GCcomb_mean.txt",
        "desi_dr2_cov": cache_dir / "desi" / "dr2" / "desi_gaussian_bao_ALL_GCcomb_cov.txt",
        "sdss_dr12_mean": cache_dir / "sdss" / "sdss_DR12Consensus_bao.dat",
        "sdss_dr12_cov": cache_dir / "sdss" / "BAO_consensus_covtot_dM_Hz.txt",
    }
    for key in ["pantheon_table", "pantheon_cov", "desi_dr2_mean", "desi_dr2_cov"]:
        download_to_cache(PUBLIC_DATA_URLS[key], files[key], refresh=refresh)
    if include_sdss:
        for key in ["sdss_dr12_mean", "sdss_dr12_cov"]:
            download_to_cache(PUBLIC_DATA_URLS[key], files[key], refresh=refresh)
    return files


def _iter_data_lines(path_or_url: str) -> Iterable[str]:
    with smart_open_text(path_or_url) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


# ----------------------------- data types ----------------------------------- #

@dataclass
class PantheonData:
    z_cmb: np.ndarray
    z_hel: np.ndarray
    observed: np.ndarray
    covariance: Optional[np.ndarray]
    sigma: Optional[np.ndarray]
    mode: str
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
class Metric:
    kind: str
    cho_factor_tuple: Optional[Tuple[np.ndarray, bool]]
    sigma: Optional[np.ndarray]
    pinv: Optional[np.ndarray]


@dataclass
class FitResult:
    x: np.ndarray
    fun: float
    success: bool
    message: str
    nfev: int
    nit: int
    elapsed_sec: float


# ----------------------------- loaders -------------------------------------- #

def load_pantheon(path: str, cov_path: Optional[str] = None) -> PantheonData:
    df = read_table(path)
    z_cmb_col = first_present(df, ["zHD", "z_cmb", "zcmb", "z"])
    if z_cmb_col is None:
        raise RuntimeError(f"{path}: no z column found")
    z_hel_col = first_present(df, ["zHEL", "zhel"])

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
        raise RuntimeError(f"{path}: no MU-like or m_b_corr-like column found")

    cov = None
    sigma = None
    if cov_path is not None:
        cov = read_covariance(cov_path)
        if cov.shape != (len(df), len(df)):
            raise RuntimeError(f"{path}: covariance shape mismatch with table length")
    elif err_col is not None:
        sigma = df[err_col].to_numpy(dtype=float)
    else:
        raise RuntimeError(f"{path}: no covariance supplied and no error column found")

    z_cmb = df[z_cmb_col].to_numpy(dtype=float)
    z_hel = df[z_hel_col].to_numpy(dtype=float) if z_hel_col else z_cmb.copy()

    finite_or_fail(z_cmb, "Pantheon z_cmb")
    finite_or_fail(z_hel, "Pantheon z_hel")
    finite_or_fail(observed, "Pantheon observed")
    return PantheonData(z_cmb, z_hel, observed, cov, sigma, mode, str(path))


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
        raise RuntimeError(f"{table_path}: BAO table must have z, observable, value columns")
    cov = None
    sigma = None
    if cov_path is not None:
        cov = read_covariance(cov_path)
        if cov.shape != (len(df), len(df)):
            raise RuntimeError(f"{table_path}: covariance shape mismatch")
    elif err_col is not None:
        sigma = df[err_col].to_numpy(dtype=float)
    else:
        raise RuntimeError(f"{table_path}: no covariance supplied and no error column found")
    return BAOBlock(
        z=df[z_col].to_numpy(dtype=float),
        observable=[str(x) for x in df[obs_col].tolist()],
        observed=df[val_col].to_numpy(dtype=float),
        covariance=cov,
        sigma=sigma,
        label=Path(table_path).name,
    )


def load_desi_dr2_gaussian(mean_path: str, cov_path: str) -> BAOBlock:
    z_vals: List[float] = []
    obs_vals: List[float] = []
    obs_names: List[str] = []
    mapping = {"DV_over_rs": "DV/rd", "DM_over_rs": "DM/rd", "DH_over_rs": "DH/rd"}
    for line in _iter_data_lines(mean_path):
        z_str, val_str, name = line.split()
        if name not in mapping:
            raise RuntimeError(f"Unsupported DESI quantity {name}")
        z_vals.append(float(z_str))
        obs_vals.append(float(val_str))
        obs_names.append(mapping[name])
    cov = read_covariance(cov_path)
    if cov.shape != (len(z_vals), len(z_vals)):
        raise RuntimeError("DESI covariance shape mismatch")
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
        z_str, val_str, name = line.split()
        raw_z.append(float(z_str))
        raw_obs.append(float(val_str))
        raw_names.append(name)
    cov_y = read_covariance(cov_path)
    n = len(raw_obs)
    if cov_y.shape != (n, n):
        raise RuntimeError("SDSS covariance shape mismatch")

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
            raise RuntimeError(f"Unsupported SDSS quantity {name}")
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


# ----------------------------- residual metrics ------------------------------ #

def prepare_metric(cov: Optional[np.ndarray], sigma: Optional[np.ndarray]) -> Metric:
    if cov is not None:
        try:
            return Metric("chol", linalg.cho_factor(cov, overwrite_a=False, check_finite=False), None, None)
        except linalg.LinAlgError:
            return Metric("pinv", None, None, np.linalg.pinv(cov))
    if sigma is None:
        raise ValueError("Need either covariance or sigma")
    return Metric("diag", None, np.asarray(sigma, dtype=float), None)


def chi2_from_metric(residual: np.ndarray, metric: Metric) -> float:
    r = np.asarray(residual, dtype=float)
    if metric.kind == "chol":
        sol = linalg.cho_solve(metric.cho_factor_tuple, r, check_finite=False)
        return float(r @ sol)
    if metric.kind == "pinv":
        return float(r @ metric.pinv @ r)
    return float(np.sum((r / metric.sigma) ** 2))


# ----------------------------- cosmology ------------------------------------ #

class OscillatoryWzCosmology:
    def __init__(self, z_max: float, integration_n: int = 1000, rd_mpc: float = 147.09) -> None:
        self.z_max = float(z_max)
        self.integration_n = int(integration_n)
        self.rd_mpc = float(rd_mpc)

    def w_of_z(self, z: np.ndarray, freqs: np.ndarray, ccos: np.ndarray, ssin: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        w = -np.ones_like(z, dtype=float)
        for f, c, s in zip(freqs, ccos, ssin):
            ang = 2.0 * np.pi * f * z
            w = w + c * np.cos(ang) + s * np.sin(ang)
        return w

    def background(self, omega_m: float, h: float, freqs: np.ndarray, ccos: np.ndarray, ssin: np.ndarray) -> Dict[str, object]:
        if not (0.0 < omega_m < 1.0):
            raise ValueError("omega_m out of bounds")
        if not (0.2 < h < 1.2):
            raise ValueError("h out of bounds")
        z_top = self.z_max * 1.1 + 0.05
        z_grid = np.linspace(0.0, z_top, self.integration_n)
        w_grid = self.w_of_z(z_grid, freqs, ccos, ssin)
        integrand = 3.0 * (1.0 + w_grid) / (1.0 + z_grid)
        integral = np.zeros_like(z_grid)
        dz = np.diff(z_grid)
        integral[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dz)
        rho_de = np.exp(integral)
        ez2 = omega_m * (1.0 + z_grid) ** 3 + (1.0 - omega_m) * rho_de
        if np.any(ez2 <= 0.0):
            raise ValueError("non-positive E(z)^2")
        e_grid = np.sqrt(ez2)
        chi_grid = np.zeros_like(z_grid)
        chi_grid[1:] = np.cumsum(0.5 * (1.0 / e_grid[1:] + 1.0 / e_grid[:-1]) * dz)
        return {
            "H0": 100.0 * h,
            "e_interp": interp1d(z_grid, e_grid, kind="linear", bounds_error=False, fill_value="extrapolate"),
            "chi_interp": interp1d(z_grid, chi_grid, kind="linear", bounds_error=False, fill_value="extrapolate"),
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
        d_l_mpc = (1.0 + np.asarray(z_hel, dtype=float)) * self.dm_mpc(bg, np.asarray(z_cmb, dtype=float))
        return 5.0 * np.log10(np.maximum(d_l_mpc, 1e-12)) + 25.0

    def predict_bao(self, bg: Dict[str, object], z: np.ndarray, observable: Sequence[str]) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z, dtype=float)
        for i, obs in enumerate(observable):
            name = sanitize_observable_name(obs)
            if name in ("dm/rd", "dmrd"):
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
                raise ValueError(f"Unsupported BAO observable {obs}")
        return out


# ----------------------------- fitter --------------------------------------- #

class OscillationOnlyFitter:
    def __init__(
        self,
        pantheon: PantheonData,
        bao_blocks: Sequence[BAOBlock],
        n_harmonics: int,
        freq_min: float,
        freq_max: float,
        rd_mpc: float = 147.09,
        integration_n: int = 1000,
        coeff_bound: float = 0.20,
        amp_prior_sigma: float = 0.05,
        freq_repulsion_strength: float = 2.0,
        progress_every_sec: float = 15.0,
    ) -> None:
        self.pantheon = pantheon
        self.bao_blocks = list(bao_blocks)
        self.n_harmonics = int(n_harmonics)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.coeff_bound = float(coeff_bound)
        self.amp_prior_sigma = float(amp_prior_sigma)
        self.freq_repulsion_strength = float(freq_repulsion_strength)
        self.progress_every_sec = float(progress_every_sec)
        self.min_freq_separation = (self.freq_max - self.freq_min) / max(8.0, 2.0 * max(1, self.n_harmonics))
        z_max = max(float(np.max(self.pantheon.z_cmb)), max((float(np.max(b.z)) for b in self.bao_blocks), default=0.0), 2.5)
        self.cosmo = OscillatoryWzCosmology(z_max, integration_n=integration_n, rd_mpc=rd_mpc)
        self.has_m_offset = (self.pantheon.mode == "m")
        self.sn_metric = prepare_metric(self.pantheon.covariance, self.pantheon.sigma)
        self.bao_metrics = [prepare_metric(b.covariance, b.sigma) for b in self.bao_blocks]
        self.total_observations = len(self.pantheon) + sum(len(b) for b in self.bao_blocks)
        self.eval_count = 0
        self.last_progress = time.time()

    def param_count(self) -> int:
        return 2 + 3 * self.n_harmonics + (1 if self.has_m_offset else 0)

    def unpack(self, theta: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
        theta = np.asarray(theta, dtype=float)
        k = self.n_harmonics
        om = float(theta[0])
        h = float(theta[1])
        freqs = np.asarray(theta[2:2 + k], dtype=float)
        ccos = np.asarray(theta[2 + k:2 + 2 * k], dtype=float)
        ssin = np.asarray(theta[2 + 2 * k:2 + 3 * k], dtype=float)
        M = float(theta[2 + 3 * k]) if self.has_m_offset else None
        return om, h, freqs, ccos, ssin, M

    def previous_components(self, prev_theta: np.ndarray, prev_k: int) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
        prev_theta = np.asarray(prev_theta, dtype=float)
        om = float(prev_theta[0])
        h = float(prev_theta[1])
        freqs = np.asarray(prev_theta[2:2 + prev_k], dtype=float)
        ccos = np.asarray(prev_theta[2 + prev_k:2 + 2 * prev_k], dtype=float)
        ssin = np.asarray(prev_theta[2 + 2 * prev_k:2 + 3 * prev_k], dtype=float)
        M = float(prev_theta[2 + 3 * prev_k]) if self.has_m_offset and len(prev_theta) > 2 + 3 * prev_k else None
        return om, h, freqs, ccos, ssin, M

    def initial_theta(self, rng: np.random.Generator, prev_theta: Optional[np.ndarray] = None, prev_k: int = 0) -> np.ndarray:
        k = self.n_harmonics
        parts = [np.array([0.30, 0.70], dtype=float)]

        if prev_theta is not None:
            pom, ph, pf, pc, ps, pM = self.previous_components(prev_theta, prev_k)
            parts[0] = np.array([pom, ph], dtype=float)
            if k > 0:
                if prev_k > 0:
                    freqs = np.array(pf, dtype=float)
                    ccos = np.array(pc, dtype=float)
                    ssin = np.array(ps, dtype=float)
                else:
                    freqs = np.empty(0, dtype=float)
                    ccos = np.empty(0, dtype=float)
                    ssin = np.empty(0, dtype=float)
                if k > prev_k:
                    freqs = np.concatenate([freqs, rng.uniform(self.freq_min, self.freq_max, size=k - prev_k)])
                    ccos = np.concatenate([ccos, rng.normal(0.0, 0.002, size=k - prev_k)])
                    ssin = np.concatenate([ssin, rng.normal(0.0, 0.002, size=k - prev_k)])
                parts.extend([freqs[:k], ccos[:k], ssin[:k]])
            if self.has_m_offset:
                parts.append(np.array([pM if pM is not None else -19.3], dtype=float))
        else:
            if k > 0:
                freqs = np.linspace(self.freq_min, self.freq_max, k + 2)[1:-1]
                freqs += rng.normal(0.0, 0.03 * (self.freq_max - self.freq_min), size=k)
                freqs = np.clip(freqs, self.freq_min, self.freq_max)
                parts.extend([freqs, rng.normal(0.0, 0.002, size=k), rng.normal(0.0, 0.002, size=k)])
            if self.has_m_offset:
                parts.append(np.array([-19.3], dtype=float))

        return np.concatenate(parts)

    def bounds(self) -> List[Tuple[float, float]]:
        bounds: List[Tuple[float, float]] = [(0.05, 0.60), (0.50, 0.90)]
        bounds.extend([(self.freq_min, self.freq_max)] * self.n_harmonics)
        bounds.extend([(-self.coeff_bound, self.coeff_bound)] * self.n_harmonics)
        bounds.extend([(-self.coeff_bound, self.coeff_bound)] * self.n_harmonics)
        if self.has_m_offset:
            bounds.append((-20.5, -18.0))
        return bounds

    def repulsion_penalty(self, freqs: np.ndarray) -> float:
        if len(freqs) < 2 or self.freq_repulsion_strength <= 0.0:
            return 0.0
        pen = 0.0
        for i in range(len(freqs)):
            for j in range(i + 1, len(freqs)):
                delta = abs(float(freqs[i] - freqs[j]))
                pen += math.exp(-0.5 * (delta / self.min_freq_separation) ** 2)
        return self.freq_repulsion_strength * pen

    def objective(self, theta: np.ndarray) -> float:
        self.eval_count += 1
        now = time.time()
        if now - self.last_progress >= self.progress_every_sec:
            eprint(f"[fit] k={self.n_harmonics} eval={self.eval_count}")
            self.last_progress = now

        om, h, freqs, ccos, ssin, M = self.unpack(theta)
        try:
            bg = self.cosmo.background(om, h, freqs, ccos, ssin)
        except Exception:
            return 1e100

        chi2 = 0.0
        mu = self.cosmo.distance_modulus(bg, self.pantheon.z_cmb, self.pantheon.z_hel)
        pred_sn = mu + float(M) if self.has_m_offset else mu
        chi2 += chi2_from_metric(self.pantheon.observed - pred_sn, self.sn_metric)

        for block, metric in zip(self.bao_blocks, self.bao_metrics):
            pred = self.cosmo.predict_bao(bg, block.z, block.observable)
            chi2 += chi2_from_metric(block.observed - pred, metric)

        if self.n_harmonics > 0:
            chi2 += float(np.sum((ccos / self.amp_prior_sigma) ** 2 + (ssin / self.amp_prior_sigma) ** 2))
            chi2 += self.repulsion_penalty(freqs)

        return float(chi2) if np.isfinite(chi2) else 1e100

    def fit(self, multistart: int = 2, seed: int = 1234, prev_theta: Optional[np.ndarray] = None, prev_k: int = 0) -> FitResult:
        rng = np.random.default_rng(seed + 7919 * self.n_harmonics)
        bounds = self.bounds()
        best: Optional[FitResult] = None

        for i in range(multistart):
            theta0 = self.initial_theta(rng, prev_theta=prev_theta, prev_k=prev_k)
            if i > 0:
                theta0 = theta0.copy()
                theta0[0] = np.clip(theta0[0] + rng.normal(0.0, 0.04), *bounds[0])
                theta0[1] = np.clip(theta0[1] + rng.normal(0.0, 0.03), *bounds[1])
                if self.n_harmonics > 0:
                    fs = slice(2, 2 + self.n_harmonics)
                    cs = slice(2 + self.n_harmonics, 2 + 2 * self.n_harmonics)
                    ss = slice(2 + 2 * self.n_harmonics, 2 + 3 * self.n_harmonics)
                    theta0[fs] += rng.normal(0.0, 0.08 * (self.freq_max - self.freq_min), size=self.n_harmonics)
                    theta0[fs] = np.clip(theta0[fs], self.freq_min, self.freq_max)
                    theta0[cs] += rng.normal(0.0, 0.01, size=self.n_harmonics)
                    theta0[ss] += rng.normal(0.0, 0.01, size=self.n_harmonics)

            self.eval_count = 0
            self.last_progress = time.time()
            t0 = time.time()
            eprint(f"[fit] starting k={self.n_harmonics} multistart {i+1}/{multistart}")
            res = optimize.minimize(
                self.objective,
                theta0,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=800, ftol=1e-10, maxls=40),
            )
            elapsed = time.time() - t0
            cand = FitResult(
                x=np.asarray(res.x, dtype=float),
                fun=float(res.fun),
                success=bool(res.success),
                message=str(res.message),
                nfev=int(res.nfev),
                nit=int(getattr(res, "nit", -1)),
                elapsed_sec=elapsed,
            )
            eprint(f"[fit] finished k={self.n_harmonics} multistart {i+1}/{multistart} chi2={cand.fun:.3f} elapsed={elapsed:.1f}s")
            if best is None or cand.fun < best.fun:
                best = cand

        assert best is not None
        return best


# ----------------------------- summaries/plots ------------------------------- #

def harmonics_rows(theta: np.ndarray, k: int, has_m_offset: bool) -> List[Dict[str, float]]:
    theta = np.asarray(theta, dtype=float)
    freqs = np.asarray(theta[2:2 + k], dtype=float)
    ccos = np.asarray(theta[2 + k:2 + 2 * k], dtype=float)
    ssin = np.asarray(theta[2 + 2 * k:2 + 3 * k], dtype=float)
    amps = np.hypot(ccos, ssin)
    phase = np.arctan2(-ssin, ccos)
    order = np.argsort(freqs)
    rows: List[Dict[str, float]] = []
    for rank, idx in enumerate(order, start=1):
        rows.append({
            "harmonic_index": rank,
            "frequency": float(freqs[idx]),
            "period_in_z": float(1.0 / freqs[idx]) if freqs[idx] > 0 else float("inf"),
            "cos_coeff": float(ccos[idx]),
            "sin_coeff": float(ssin[idx]),
            "amplitude": float(amps[idx]),
            "phase_rad": float(phase[idx]),
        })
    return rows


def build_support_grid(pantheon: PantheonData, bao_blocks: Sequence[BAOBlock], z_max: float, n_points: int = 120) -> np.ndarray:
    z = np.concatenate([pantheon.z_cmb] + [b.z for b in bao_blocks])
    z = z[(z >= 0.0) & (z <= z_max)]
    support = robust_quantile_bins(z, n_points)
    if support[0] > 0.0:
        support = np.insert(support, 0, 0.0)
    if support[-1] < z_max:
        support = np.append(support, z_max)
    return np.unique(support)


def make_wz_plot(outpath: Path, z_plot: np.ndarray, best_w: np.ndarray, six_w: np.ndarray, z_support: np.ndarray, best_delta: np.ndarray, status_text: str) -> None:
    fig = plt.figure(figsize=(8.6, 7.0))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(z_plot, best_w, lw=2, label="Best BIC model")
    ax1.plot(z_plot, six_w, lw=1.3, ls="--", label="6-harmonic model")
    ax1.axhline(-1.0, lw=1, ls=":", label=r"$w_\Lambda=-1$")
    ax1.set_ylabel("w(z)")
    ax1.legend(loc="best")
    ax1.text(0.02, 0.03, status_text, transform=ax1.transAxes, fontsize=9)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(z_support, best_delta, lw=2, label=r"Best-model $\Delta w(z)$")
    ax2.axhline(0.0, lw=1, ls=":")
    ax2.set_xlabel("z")
    ax2.set_ylabel(r"$\Delta w$")
    ax2.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def make_model_selection_plot(outpath: Path, ks: np.ndarray, bic: np.ndarray, aic: np.ndarray, chi2: np.ndarray) -> None:
    fig = plt.figure(figsize=(8.6, 5.3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ks, bic, marker="o", label="BIC")
    ax.plot(ks, aic, marker="s", label="AIC")
    ax.plot(ks, chi2, marker="^", label=r"$\chi^2$")
    ax.set_xlabel("Number of harmonics")
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def save_result_json(outpath: Path, payload: Dict[str, object]) -> None:
    class Encoder(json.JSONEncoder):
        def default(self, obj):  # type: ignore[override]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            return super().default(obj)
    outpath.write_text(json.dumps(payload, indent=2, cls=Encoder), encoding="utf-8")


# ----------------------------- demo mode ------------------------------------- #

def generate_demo_data(rd_mpc: float, seed: int = 7) -> Tuple[PantheonData, List[BAOBlock]]:
    rng = np.random.default_rng(seed)
    cosmo = OscillatoryWzCosmology(2.5, integration_n=1200, rd_mpc=rd_mpc)

    true_freqs = np.array([0.55, 0.95, 1.45, 1.95, 2.55, 3.25], dtype=float)
    true_amp = np.array([0.0070, 0.0055, 0.0045, 0.0038, 0.0032, 0.0027], dtype=float)
    true_phi = np.array([0.3, -0.9, 1.2, -0.2, 0.8, -1.1], dtype=float)
    true_c = true_amp * np.cos(true_phi)
    true_s = -true_amp * np.sin(true_phi)

    z_sn = np.sort(np.concatenate([
        rng.uniform(0.01, 0.15, 180),
        rng.uniform(0.15, 0.8, 320),
        rng.uniform(0.8, 2.2, 180),
    ]))
    z_hel = z_sn + rng.normal(0.0, 0.0007, size=z_sn.size)
    bg = cosmo.background(0.305, 0.692, true_freqs, true_c, true_s)
    mu_true = cosmo.distance_modulus(bg, z_sn, z_hel)
    sn_sigma = np.full_like(z_sn, 0.10)
    mu_obs = mu_true + rng.normal(0.0, sn_sigma)

    pantheon = PantheonData(z_sn, z_hel, mu_obs, None, sn_sigma, "mu", "demo_synthetic")

    bao_specs = [
        (0.38, "DM/rd", 0.020), (0.38, "DH/rd", 0.028),
        (0.51, "DM/rd", 0.018), (0.51, "DH/rd", 0.025),
        (0.70, "DM/rd", 0.020), (0.70, "DH/rd", 0.030),
        (1.10, "DV/rd", 0.030), (1.48, "DH/rd", 0.050),
        (1.48, "DM/rd", 0.040), (2.33, "DH/rd", 0.065),
    ]
    z_bao = np.array([x[0] for x in bao_specs], dtype=float)
    obs_kind = [x[1] for x in bao_specs]
    frac_err = np.array([x[2] for x in bao_specs], dtype=float)
    bao_true = cosmo.predict_bao(bg, z_bao, obs_kind)
    bao_sigma = frac_err * bao_true
    bao_obs = bao_true + rng.normal(0.0, bao_sigma)

    bao_block = BAOBlock(z_bao, obs_kind, bao_obs, None, bao_sigma, "demo_bao")
    return pantheon, [bao_block]


# ----------------------------- CLI/main -------------------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test 02 oscillation-only rerun")
    p.add_argument("--pantheon", help="Pantheon table path/URL; if omitted, public data are auto-downloaded")
    p.add_argument("--pantheon-cov", help="Pantheon covariance path/URL")
    p.add_argument("--bao", action="append", default=[], help="BAO table path/URL optionally as table|cov")
    p.add_argument("--skip-default-sdss", action="store_true", help="Do not include SDSS DR12 when auto-downloading")
    p.add_argument("--data-cache-dir", default="p02_public_data", help="Cache directory for public data")
    p.add_argument("--refresh-data", action="store_true", help="Force re-download of public data")
    p.add_argument("--outdir", default="out_p02", help="Output directory")
    p.add_argument("--rd", type=float, default=147.09, help="Sound horizon r_d in Mpc")
    p.add_argument("--z-max", type=float, default=2.5, help="Max z for diagnostics and default period range")
    p.add_argument("--period-min", type=float, help="Minimum oscillation period in redshift units")
    p.add_argument("--period-max", type=float, help="Maximum oscillation period in redshift units")
    p.add_argument("--target-harmonics", type=int, default=6, help="Fit 0..K harmonic models up to this value")
    p.add_argument("--headline-threshold", type=int, default=6, help="Number of surviving harmonics required for headline status")
    p.add_argument("--multistart", type=int, default=2, help="Optimizer restarts per harmonic count")
    p.add_argument("--integration-n", type=int, default=1000, help="Integration grid points")
    p.add_argument("--coeff-bound", type=float, default=0.20, help="Hard bound on each sine/cosine coefficient")
    p.add_argument("--amp-prior-sigma", type=float, default=0.05, help="Weak Gaussian prior sigma for sine/cosine coefficients")
    p.add_argument("--freq-repulsion-strength", type=float, default=2.0, help="Penalty strength against duplicate frequencies")
    p.add_argument("--progress-every-sec", type=float, default=15.0, help="Progress heartbeat interval")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument("--demo", action="store_true", help="Run on synthetic data instead of public/manual data")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.headline_threshold > args.target_harmonics:
        raise SystemExit("--headline-threshold cannot exceed --target-harmonics")
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    if args.demo:
        pantheon, bao_blocks = generate_demo_data(args.rd, seed=args.seed)
        eprint("[demo] generated synthetic Pantheon/BAO data")
    else:
        if args.pantheon:
            if not args.bao:
                raise SystemExit("If --pantheon is supplied manually, provide at least one --bao input too")
            pantheon = load_pantheon(args.pantheon, args.pantheon_cov)
            bao_blocks = [load_bao_block(x) for x in args.bao]
        else:
            files = prepare_default_public_data(Path(args.data_cache_dir), refresh=args.refresh_data, include_sdss=not args.skip_default_sdss)
            pantheon = load_pantheon(str(files["pantheon_table"]), str(files["pantheon_cov"]))
            bao_blocks = [load_desi_dr2_gaussian(str(files["desi_dr2_mean"]), str(files["desi_dr2_cov"]))]
            if not args.skip_default_sdss:
                bao_blocks.append(load_sdss_dr12_consensus(str(files["sdss_dr12_mean"]), str(files["sdss_dr12_cov"])))
            eprint(f"[data] Downloaded public Pantheon+/DESI inputs into {args.data_cache_dir}")
            if not args.skip_default_sdss:
                eprint("[data] Included public SDSS DR12 consensus BAO supplement")

    period_max = args.period_max if args.period_max is not None else float(args.z_max)
    period_min = args.period_min if args.period_min is not None else float(args.z_max) / 10.0
    if period_min <= 0.0 or period_max <= 0.0 or period_min >= period_max:
        raise SystemExit("Require positive periods with period_min < period_max")
    freq_min = 1.0 / period_max
    freq_max = 1.0 / period_min

    summaries: List[Dict[str, object]] = []
    prev_theta: Optional[np.ndarray] = None
    prev_k = 0

    for k in range(args.target_harmonics + 1):
        fitter = OscillationOnlyFitter(
            pantheon=pantheon,
            bao_blocks=bao_blocks,
            n_harmonics=k,
            freq_min=freq_min,
            freq_max=freq_max,
            rd_mpc=args.rd,
            integration_n=args.integration_n,
            coeff_bound=args.coeff_bound,
            amp_prior_sigma=args.amp_prior_sigma,
            freq_repulsion_strength=args.freq_repulsion_strength,
            progress_every_sec=args.progress_every_sec,
        )
        fit = fitter.fit(multistart=args.multistart, seed=args.seed, prev_theta=prev_theta, prev_k=prev_k)
        pcount = fitter.param_count()
        nobs = fitter.total_observations
        aic = fit.fun + 2.0 * pcount
        bic = fit.fun + pcount * math.log(nobs)

        delta_chi2_vs_prev = None
        delta_bic_vs_prev = None
        lr_pvalue_vs_prev = None
        if summaries:
            delta_chi2_vs_prev = float(summaries[-1]["chi2_total"] - fit.fun)
            delta_bic_vs_prev = float(bic - float(summaries[-1]["bic"]))
            df_add = pcount - int(summaries[-1]["param_count"])
            lr_pvalue_vs_prev = float(stats.chi2.sf(max(0.0, delta_chi2_vs_prev), df=df_add))

        summary = {
            "n_harmonics": k,
            "chi2_total": float(fit.fun),
            "aic": float(aic),
            "bic": float(bic),
            "param_count": int(pcount),
            "fit_success": bool(fit.success),
            "fit_message": fit.message,
            "nfev": int(fit.nfev),
            "nit": int(fit.nit),
            "elapsed_sec": float(fit.elapsed_sec),
            "delta_chi2_vs_prev": delta_chi2_vs_prev,
            "delta_bic_vs_prev": delta_bic_vs_prev,
            "lr_pvalue_vs_prev": lr_pvalue_vs_prev,
            "harmonics": harmonics_rows(fit.x, k, fitter.has_m_offset) if k > 0 else [],
            "theta": fit.x.tolist(),
        }
        summaries.append(summary)
        prev_theta = fit.x
        prev_k = k

    best_summary = min(summaries, key=lambda d: float(d["bic"]))
    best_k = int(best_summary["n_harmonics"])
    six_summary = summaries[args.headline_threshold]
    min_bic = min(float(s["bic"]) for s in summaries)
    six_survive = bool(best_k >= args.headline_threshold and abs(float(six_summary["bic"]) - min_bic) < 1e-9)
    status = "headline_result" if six_survive else "null_under_oscillation_only"

    best_theta = np.asarray(best_summary["theta"], dtype=float)
    best_fitter = OscillationOnlyFitter(
        pantheon, bao_blocks, best_k, freq_min, freq_max,
        rd_mpc=args.rd, integration_n=args.integration_n,
        coeff_bound=args.coeff_bound, amp_prior_sigma=args.amp_prior_sigma,
        freq_repulsion_strength=args.freq_repulsion_strength, progress_every_sec=1e9,
    )
    om_b, h_b, f_b, c_b, s_b, _ = best_fitter.unpack(best_theta)
    best_w_plot = best_fitter.cosmo.w_of_z(np.linspace(0.0, args.z_max, 400), f_b, c_b, s_b)
    support_z = build_support_grid(pantheon, bao_blocks, args.z_max, 120)
    best_delta_support = best_fitter.cosmo.w_of_z(support_z, f_b, c_b, s_b) + 1.0

    six_theta = np.asarray(six_summary["theta"], dtype=float)
    six_fitter = OscillationOnlyFitter(
        pantheon, bao_blocks, args.headline_threshold, freq_min, freq_max,
        rd_mpc=args.rd, integration_n=args.integration_n,
        coeff_bound=args.coeff_bound, amp_prior_sigma=args.amp_prior_sigma,
        freq_repulsion_strength=args.freq_repulsion_strength, progress_every_sec=1e9,
    )
    _, _, f6, c6, s6, _ = six_fitter.unpack(six_theta)
    z_plot = np.linspace(0.0, args.z_max, 400)
    six_w_plot = six_fitter.cosmo.w_of_z(z_plot, f6, c6, s6)

    model_selection_df = pd.DataFrame([{
        "n_harmonics": s["n_harmonics"],
        "chi2_total": s["chi2_total"],
        "aic": s["aic"],
        "bic": s["bic"],
        "param_count": s["param_count"],
        "delta_chi2_vs_prev": s["delta_chi2_vs_prev"],
        "delta_bic_vs_prev": s["delta_bic_vs_prev"],
        "lr_pvalue_vs_prev": s["lr_pvalue_vs_prev"],
        "fit_success": s["fit_success"],
        "elapsed_sec": s["elapsed_sec"],
    } for s in summaries])
    model_selection_df.to_csv(outdir / "model_selection.csv", index=False)
    pd.DataFrame(best_summary["harmonics"]).to_csv(outdir / "best_harmonics.csv", index=False)
    pd.DataFrame(six_summary["harmonics"]).to_csv(outdir / "six_harmonics.csv", index=False)
    pd.DataFrame({"z": support_z, "delta_w_best": best_delta_support}).to_csv(outdir / "residual_grid.csv", index=False)

    status_text = f"status = {status}\nbest harmonic count by BIC = {best_k}\nsix-harmonic BIC = {float(six_summary['bic']):.2f}"
    make_wz_plot(outdir / "wz_fit.png", z_plot, best_w_plot, six_w_plot, support_z, best_delta_support, status_text)
    make_model_selection_plot(outdir / "model_selection.png", model_selection_df["n_harmonics"].to_numpy(), model_selection_df["bic"].to_numpy(), model_selection_df["aic"].to_numpy(), model_selection_df["chi2_total"].to_numpy())

    result = {
        "test_name": "CCDR Test 02 oscillation-only rerun",
        "basis": "w(z) = -1 + sum_k [c_k cos(2*pi*f_k*z) + s_k sin(2*pi*f_k*z)]",
        "no_drift_term": True,
        "pantheon_source": pantheon.source,
        "bao_sources": [b.label for b in bao_blocks],
        "settings": {
            "target_harmonics": int(args.target_harmonics),
            "headline_threshold": int(args.headline_threshold),
            "period_min": float(period_min),
            "period_max": float(period_max),
            "frequency_min": float(freq_min),
            "frequency_max": float(freq_max),
            "multistart": int(args.multistart),
            "integration_n": int(args.integration_n),
            "coeff_bound": float(args.coeff_bound),
            "amp_prior_sigma": float(args.amp_prior_sigma),
            "freq_repulsion_strength": float(args.freq_repulsion_strength),
            "rd_mpc": float(args.rd),
        },
        "model_selection": summaries,
        "best_model": {
            "n_harmonics": int(best_k),
            "bic": float(best_summary["bic"]),
            "aic": float(best_summary["aic"]),
            "chi2_total": float(best_summary["chi2_total"]),
            "harmonics": best_summary["harmonics"],
        },
        "six_harmonic_model": {
            "bic": float(six_summary["bic"]),
            "aic": float(six_summary["aic"]),
            "chi2_total": float(six_summary["chi2_total"]),
            "delta_bic_vs_prev": six_summary["delta_bic_vs_prev"],
            "delta_chi2_vs_prev": six_summary["delta_chi2_vs_prev"],
            "lr_pvalue_vs_prev": six_summary["lr_pvalue_vs_prev"],
            "harmonics": six_summary["harmonics"],
        },
        "result": {
            "six_harmonics_survive": bool(six_survive),
            "headline_result": bool(six_survive),
            "mark_test02_null": bool(not six_survive),
            "status": status,
            "best_supported_harmonic_count": int(best_k),
        },
    }
    save_result_json(outdir / "result.json", result)
    eprint("[done] wrote result.json, model_selection.csv, best_harmonics.csv, six_harmonics.csv, residual_grid.csv, wz_fit.png, model_selection.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
