#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import emcee
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires emcee. Install it with: pip install emcee"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires matplotlib. Install it with: pip install matplotlib"
    ) from exc

try:
    from scipy import integrate, linalg, optimize, stats
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires scipy. Install it with: pip install scipy"
    ) from exc


C_LIGHT = 299792.458  # km/s
DEFAULT_PANTHEON_URL = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
    "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
)
DEFAULT_PANTHEON_COV_URL = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
    "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"
)
DEFAULT_DESI_MEAN_URL = (
    "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/"
    "desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt"
)
DEFAULT_DESI_COV_URL = (
    "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/"
    "desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt"
)
DEFAULT_PLANCK_ZIP_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/"
    "COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip"
)


@dataclass(slots=True)
class PantheonData:
    z_cmb: np.ndarray
    z_hel: np.ndarray
    mu_obs: np.ndarray
    cov: np.ndarray
    cov_chol: np.ndarray
    names: tuple[str, ...]


@dataclass(slots=True)
class BaoData:
    z: np.ndarray
    observable: list[str]
    value: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray


@dataclass(slots=True)
class PlanckCompressedPrior:
    names: list[str]
    mean: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray
    source_label: str


@dataclass(slots=True)
class FitResult:
    nu_mean: float
    nu_sigma: float
    nu_95ci: tuple[float, float]
    bayes_factor: float | None
    pass_pred: bool
    pass_lcdm_disfavored: bool
    nu_positive_2sigma: bool
    verdict: str
    note: str
    burn_in: int
    thin: int
    tau_max: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "nu_mean": self.nu_mean,
            "nu_sigma": self.nu_sigma,
            "nu_95ci": [self.nu_95ci[0], self.nu_95ci[1]],
            "bayes_factor": self.bayes_factor,
            "pass_pred": self.pass_pred,
            "pass_lcdm_disfavored": self.pass_lcdm_disfavored,
            "nu_positive_2sigma": self.nu_positive_2sigma,
            "verdict": self.verdict,
            "note": self.note,
            "burn_in": self.burn_in,
            "thin": self.thin,
            "tau_max": self.tau_max,
        }


@dataclass(slots=True)
class Context:
    pantheon: PantheonData
    bao: BaoData
    planck: PlanckCompressedPrior
    grid_size: int
    omega_b_h2: float
    tcmb: float
    neff: float
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


OBSERVABLE_ALIASES = {
    "dv/rd": "DV_OVER_RD",
    "dv_rd": "DV_OVER_RD",
    "dv_over_rd": "DV_OVER_RD",
    "dv_over_rs": "DV_OVER_RD",
    "dm/rd": "DM_OVER_RD",
    "dm_rd": "DM_OVER_RD",
    "dm_over_rd": "DM_OVER_RD",
    "dm_over_rs": "DM_OVER_RD",
    "dh/rd": "DH_OVER_RD",
    "dh_rd": "DH_OVER_RD",
    "dh_over_rd": "DH_OVER_RD",
    "dh_over_rs": "DH_OVER_RD",
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


def path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Test 01 fitter for the Running Vacuum Model ν coefficient. "
            "This version downloads Pantheon+, DESI DR2 BAO, and Planck PR3 chain products from public sources."
        )
    )
    parser.add_argument("--out-dir", type=path_arg, default=path_arg("./results_test01_public"))
    parser.add_argument("--cache-dir", type=path_arg, default=path_arg("./data_test01_public"))

    parser.add_argument("--pantheon-url", type=str, default=DEFAULT_PANTHEON_URL)
    parser.add_argument("--pantheon-cov-url", type=str, default=DEFAULT_PANTHEON_COV_URL)
    parser.add_argument("--desi-mean-url", type=str, default=DEFAULT_DESI_MEAN_URL)
    parser.add_argument("--desi-cov-url", type=str, default=DEFAULT_DESI_COV_URL)
    parser.add_argument("--planck-zip-url", type=str, default=DEFAULT_PLANCK_ZIP_URL)
    parser.add_argument("--refresh-downloads", action="store_true")

    parser.add_argument("--sn-z-column", type=str, default=None)
    parser.add_argument("--sn-zhel-column", type=str, default=None)
    parser.add_argument("--sn-mu-column", type=str, default=None)

    parser.add_argument("--walkers", type=positive_int, default=64)
    parser.add_argument("--steps", type=positive_int, default=30000)
    parser.add_argument("--burn-fallback", type=positive_int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--de-maxiter", type=positive_int, default=250)
    parser.add_argument("--distance-grid", type=positive_int, default=5000)
    parser.add_argument("--progress", action="store_true")

    parser.add_argument("--omega-b-h2", type=float, default=0.02237)
    parser.add_argument("--tcmb", type=float, default=2.7255)
    parser.add_argument("--neff", type=float, default=3.046)

    parser.add_argument("--omega-m-min", type=float, default=0.05)
    parser.add_argument("--omega-m-max", type=float, default=0.6)
    parser.add_argument("--h-min", type=float, default=0.55)
    parser.add_argument("--h-max", type=float, default=0.85)
    parser.add_argument("--nu-min", type=float, default=-0.01)
    parser.add_argument("--nu-max", type=float, default=0.01)

    parser.add_argument("--nu-pred", type=float, default=1.0e-3)
    parser.add_argument("--nu-pred-err", type=float, default=1.0e-4)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def ensure_downloaded(url: str, dst: Path, refresh: bool = False) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0 and not refresh:
        return dst
    with urllib.request.urlopen(url) as response, dst.open("wb") as fout:
        shutil.copyfileobj(response, fout)
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"Download failed or produced an empty file: {url}")
    return dst


def detect_column(names: tuple[str, ...], explicit: str | None, candidates: list[str], label: str) -> str:
    if explicit is not None:
        if explicit not in names:
            raise RuntimeError(f"Requested {label} column '{explicit}' not found. Available columns: {list(names)}")
        return explicit
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise RuntimeError(f"Could not detect {label} column. Available columns: {list(names)}")


def load_pantheon_data(
    data_path: Path,
    cov_path: Path,
    z_column: str | None,
    zhel_column: str | None,
    mu_column: str | None,
) -> PantheonData:
    try:
        table = np.genfromtxt(data_path, names=True, dtype=None, encoding=None)
    except Exception as exc:
        raise RuntimeError(f"Failed to read Pantheon+ table: {data_path}") from exc

    if table.dtype.names is None:
        raise RuntimeError(f"Pantheon+ table at {data_path} does not expose named columns.")

    names = tuple(table.dtype.names)
    z_cmb_name = detect_column(names, z_column, ["zHD", "zCMB", "zcmb", "z"], "SN redshift")
    z_hel_name = detect_column(names, zhel_column, ["zHEL", "zhel", "z_hel", z_cmb_name], "SN heliocentric redshift")
    mu_name = detect_column(names, mu_column, ["MU_SH0ES", "MU", "mu", "mu_shoes"], "SN distance-modulus")

    z_cmb = np.asarray(table[z_cmb_name], dtype=float)
    z_hel = np.asarray(table[z_hel_name], dtype=float)
    mu_obs = np.asarray(table[mu_name], dtype=float)

    if not (len(z_cmb) == len(z_hel) == len(mu_obs)):
        raise RuntimeError("Pantheon+ columns have inconsistent lengths.")

    raw = np.loadtxt(cov_path)
    flat = np.asarray(raw, dtype=float).ravel()
    n = len(mu_obs)
    if flat.size == n * n + 1 and int(round(flat[0])) == n:
        flat = flat[1:]
    if flat.size != n * n:
        raise RuntimeError(
            f"Pantheon covariance size mismatch: expected {n*n} elements (or {n*n+1} with a leading size), got {flat.size}."
        )
    cov = flat.reshape(n, n)
    cov_chol = linalg.cholesky(cov, lower=True, check_finite=False)
    return PantheonData(z_cmb=z_cmb, z_hel=z_hel, mu_obs=mu_obs, cov=cov, cov_chol=cov_chol, names=names)


def normalize_observable(value: str) -> str:
    key = value.strip().lower()
    if key not in OBSERVABLE_ALIASES:
        raise RuntimeError(
            f"Unsupported BAO observable '{value}'. Supported forms: DV/rd, DM/rd, DH/rd, and rs aliases."
        )
    return OBSERVABLE_ALIASES[key]


def load_matrix_file(path: Path, expected_size: int) -> np.ndarray:
    raw = np.loadtxt(path)
    flat = np.asarray(raw, dtype=float).ravel()
    if flat.size == expected_size * expected_size + 1 and int(round(flat[0])) == expected_size:
        flat = flat[1:]
    if flat.size != expected_size * expected_size:
        raise RuntimeError(
            f"Covariance matrix at {path} has {flat.size} elements; expected {expected_size * expected_size}."
        )
    return flat.reshape(expected_size, expected_size)


def load_public_desi_bao(mean_path: Path, cov_path: Path) -> BaoData:
    rows: list[tuple[float, str, float]] = []
    with mean_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                raise RuntimeError(f"Unexpected DESI mean-file row: {line!r}")
            z = float(parts[0])
            value = float(parts[1])
            observable = normalize_observable(parts[2])
            rows.append((z, observable, value))

    if not rows:
        raise RuntimeError(f"DESI mean file appears empty: {mean_path}")

    z = np.array([r[0] for r in rows], dtype=float)
    observable = [r[1] for r in rows]
    value = np.array([r[2] for r in rows], dtype=float)
    cov = load_matrix_file(cov_path, len(rows))
    inv_cov = linalg.inv(cov, check_finite=False)
    return BaoData(z=z, observable=observable, value=value, cov=cov, inv_cov=inv_cov)


def _clean_param_name(name: str) -> str:
    return name.strip().replace("*", "").replace("\\", "").strip().lower()


def _find_planck_param_indices(param_names: list[str]) -> tuple[int, int, str]:
    omegam_aliases = {"omegam", "omega_m", "omega_matter", "omm"}
    h_aliases = {"h"}
    h0_aliases = {"h0", "hubble", "hubble0"}

    omegam_idx = None
    h_idx = None
    h_mode = None
    for idx, name in enumerate(param_names):
        cleaned = _clean_param_name(name)
        if omegam_idx is None and cleaned in omegam_aliases:
            omegam_idx = idx
        if h_idx is None and cleaned in h_aliases:
            h_idx = idx
            h_mode = "h"
        if h_idx is None and cleaned in h0_aliases:
            h_idx = idx
            h_mode = "H0"

    if omegam_idx is None or h_idx is None or h_mode is None:
        raise RuntimeError(
            "Could not locate omegam and h/H0 columns in the Planck parameter names file. "
            f"Found names: {param_names[:20]}{'...' if len(param_names) > 20 else ''}"
        )
    return omegam_idx, h_idx, h_mode


def _weighted_mean_and_cov_from_planck_zip(zip_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        paramname_files = [name for name in names if name.endswith(".paramnames")]
        if not paramname_files:
            raise RuntimeError("No .paramnames file found inside the Planck cosmology zip archive.")
        paramname_file = sorted(paramname_files)[0]
        prefix = paramname_file[: -len(".paramnames")]

        chain_files = [
            name for name in names
            if name.startswith(prefix) and name.endswith(".txt") and "minimum" not in name.lower()
        ]
        if not chain_files:
            chain_files = [name for name in names if name.endswith(".txt") and "minimum" not in name.lower()]
        if not chain_files:
            raise RuntimeError("No chain .txt files found inside the Planck cosmology zip archive.")

        with zf.open(paramname_file, "r") as f:
            raw_text = f.read().decode("utf-8", errors="replace")
        param_names: list[str] = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            token = stripped.split()[0]
            param_names.append(token)

        omegam_idx, h_idx, h_mode = _find_planck_param_indices(param_names)
        data_omegam_col = 2 + omegam_idx
        data_h_col = 2 + h_idx

        sum_w = 0.0
        sum_wx = np.zeros(2, dtype=float)
        sum_wxx = np.zeros((2, 2), dtype=float)
        total_rows = 0

        for chain_name in sorted(chain_files):
            with zf.open(chain_name, "r") as f:
                for raw_line in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                    stripped = raw_line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    parts = stripped.split()
                    if len(parts) <= max(data_omegam_col, data_h_col):
                        continue
                    w = float(parts[0])
                    if not np.isfinite(w) or w <= 0.0:
                        continue
                    omegam = float(parts[data_omegam_col])
                    h_val = float(parts[data_h_col])
                    if h_mode == "H0":
                        h_val /= 100.0
                    x = np.array([omegam, h_val], dtype=float)
                    sum_w += w
                    sum_wx += w * x
                    sum_wxx += w * np.outer(x, x)
                    total_rows += 1

        if total_rows == 0 or sum_w <= 0.0:
            raise RuntimeError("No usable rows were read from the Planck chain files.")

        mean = sum_wx / sum_w
        cov = sum_wxx / sum_w - np.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        return mean, cov, paramname_file


def load_planck_prior_from_public_zip(zip_path: Path) -> PlanckCompressedPrior:
    mean, cov, source_label = _weighted_mean_and_cov_from_planck_zip(zip_path)
    inv_cov = linalg.inv(cov, check_finite=False)
    return PlanckCompressedPrior(
        names=["omega_m", "h"],
        mean=mean,
        cov=cov,
        inv_cov=inv_cov,
        source_label=source_label,
    )


def omega_radiation(h: float, tcmb: float, neff: float) -> float:
    theta = tcmb / 2.7255
    omega_gamma = 2.469e-5 * theta ** 4 / (h ** 2)
    return omega_gamma * (1.0 + 0.22710731766 * neff)


def rvm_e2(z: np.ndarray | float, omega_m: float, h: float, nu: float, tcmb: float, neff: float) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    om_r = omega_radiation(h, tcmb, neff)
    one_plus_z = 1.0 + z_arr
    e2 = 1.0 + (omega_m / (1.0 - nu)) * (one_plus_z ** (3.0 * (1.0 - nu)) - 1.0) + om_r * (one_plus_z ** 4.0 - 1.0)
    return e2


def hubble_rate(z: np.ndarray | float, omega_m: float, h: float, nu: float, tcmb: float, neff: float) -> np.ndarray:
    h0 = 100.0 * h
    e2 = rvm_e2(z, omega_m, h, nu, tcmb, neff)
    if np.any(e2 <= 0.0):
        raise FloatingPointError("Encountered non-positive E(z)^2 in RVM background.")
    return h0 * np.sqrt(e2)


def comoving_distance_mpc(z: np.ndarray, omega_m: float, h: float, nu: float, tcmb: float, neff: float, grid_size: int) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if np.any(z < 0.0):
        raise FloatingPointError("Negative redshift is not supported.")
    z_max = float(np.max(z)) if len(z) else 0.0
    if z_max == 0.0:
        return np.zeros_like(z)
    n_grid = max(grid_size, int(1000 * z_max) + 32)
    z_grid = np.linspace(0.0, z_max, n_grid)
    e2_grid = rvm_e2(z_grid, omega_m, h, nu, tcmb, neff)
    if np.any(e2_grid <= 0.0):
        raise FloatingPointError("Encountered non-positive E(z)^2 while integrating distances.")
    inv_e = 1.0 / np.sqrt(e2_grid)
    chi = integrate.cumulative_trapezoid(inv_e, z_grid, initial=0.0)
    return (C_LIGHT / (100.0 * h)) * np.interp(z, z_grid, chi)


def luminosity_distance_mpc(
    z_cmb: np.ndarray,
    z_hel: np.ndarray,
    omega_m: float,
    h: float,
    nu: float,
    tcmb: float,
    neff: float,
    grid_size: int,
) -> np.ndarray:
    d_c = comoving_distance_mpc(z_cmb, omega_m, h, nu, tcmb, neff, grid_size)
    return (1.0 + z_hel) * d_c


def distance_modulus_from_dl(dl_mpc: np.ndarray) -> np.ndarray:
    if np.any(dl_mpc <= 0.0):
        raise FloatingPointError("Luminosity distance must stay positive.")
    return 5.0 * np.log10(dl_mpc) + 25.0


def eisenstein_hu_rd_mpc(omega_m: float, h: float, omega_b_h2: float, tcmb: float) -> float:
    omega_m_h2 = omega_m * h * h
    theta = tcmb / 2.7
    z_eq = 2.50e4 * omega_m_h2 / (theta ** 4)
    k_eq = 0.0746 * omega_m_h2 / (theta ** 2)
    b1 = 0.313 * omega_m_h2 ** (-0.419) * (1.0 + 0.607 * omega_m_h2 ** 0.674)
    b2 = 0.238 * omega_m_h2 ** 0.223
    z_d = (
        1291.0 * omega_m_h2 ** 0.251 / (1.0 + 0.659 * omega_m_h2 ** 0.828)
        * (1.0 + b1 * omega_b_h2 ** b2)
    )
    r_eq = 31.5 * omega_b_h2 / (theta ** 4) * (1000.0 / z_eq)
    r_d = 31.5 * omega_b_h2 / (theta ** 4) * (1000.0 / z_d)
    numerator = math.sqrt(1.0 + r_d) + math.sqrt(r_d + r_eq)
    denominator = 1.0 + math.sqrt(r_eq)
    rd = (2.0 / (3.0 * k_eq)) * math.sqrt(6.0 / r_eq) * math.log(numerator / denominator)
    return rd


def bao_model_vector(
    bao: BaoData,
    omega_m: float,
    h: float,
    nu: float,
    omega_b_h2: float,
    tcmb: float,
    neff: float,
    grid_size: int,
) -> np.ndarray:
    rd = eisenstein_hu_rd_mpc(omega_m, h, omega_b_h2, tcmb)
    d_m = comoving_distance_mpc(bao.z, omega_m, h, nu, tcmb, neff, grid_size)
    h_z = hubble_rate(bao.z, omega_m, h, nu, tcmb, neff)
    d_h = C_LIGHT / h_z
    out = np.empty(len(bao.z), dtype=float)
    for i, obs in enumerate(bao.observable):
        if obs == "DM_OVER_RD":
            out[i] = d_m[i] / rd
        elif obs == "DH_OVER_RD":
            out[i] = d_h[i] / rd
        elif obs == "DV_OVER_RD":
            d_v = (bao.z[i] * d_h[i] * d_m[i] ** 2) ** (1.0 / 3.0)
            out[i] = d_v / rd
        else:  # pragma: no cover
            raise RuntimeError(f"Unexpected BAO observable: {obs}")
    return out


def within_bounds(theta: np.ndarray, bounds: tuple[tuple[float, float], ...]) -> bool:
    return all(lo <= x <= hi for x, (lo, hi) in zip(theta, bounds))


def log_prior(theta: np.ndarray, bounds: tuple[tuple[float, float], ...]) -> float:
    if not within_bounds(theta, bounds):
        return -np.inf
    return 0.0


def loglike_sn(theta: np.ndarray, ctx: Context) -> float:
    omega_m, h, nu = map(float, theta)
    dl = luminosity_distance_mpc(
        ctx.pantheon.z_cmb,
        ctx.pantheon.z_hel,
        omega_m,
        h,
        nu,
        ctx.tcmb,
        ctx.neff,
        ctx.grid_size,
    )
    mu_model = distance_modulus_from_dl(dl)
    resid = ctx.pantheon.mu_obs - mu_model
    whitened = linalg.solve_triangular(ctx.pantheon.cov_chol, resid, lower=True, check_finite=False)
    return -0.5 * float(np.dot(whitened, whitened))


def loglike_bao(theta: np.ndarray, ctx: Context) -> float:
    omega_m, h, nu = map(float, theta)
    model = bao_model_vector(
        ctx.bao,
        omega_m,
        h,
        nu,
        ctx.omega_b_h2,
        ctx.tcmb,
        ctx.neff,
        ctx.grid_size,
    )
    resid = ctx.bao.value - model
    chi2 = float(resid @ ctx.bao.inv_cov @ resid)
    return -0.5 * chi2


def loglike_planck(theta: np.ndarray, ctx: Context) -> float:
    mapping = {"omega_m": float(theta[0]), "h": float(theta[1]), "nu": float(theta[2])}
    vec = np.array([mapping[name] for name in ctx.planck.names], dtype=float)
    resid = vec - ctx.planck.mean
    chi2 = float(resid @ ctx.planck.inv_cov @ resid)
    return -0.5 * chi2


def log_probability(theta: np.ndarray, ctx: Context) -> float:
    lp = log_prior(theta, ctx.bounds)
    if not np.isfinite(lp):
        return -np.inf
    try:
        total = lp + loglike_planck(theta, ctx) + loglike_sn(theta, ctx) + loglike_bao(theta, ctx)
    except (FloatingPointError, ValueError, ZeroDivisionError, OverflowError):
        return -np.inf
    return total if np.isfinite(total) else -np.inf


def find_best_fit(ctx: Context, seed: int, de_maxiter: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    def objective(x: np.ndarray) -> float:
        lp = log_probability(x, ctx)
        return 1.0e100 if not np.isfinite(lp) else -lp

    result = optimize.differential_evolution(
        objective,
        bounds=list(ctx.bounds),
        strategy="best1bin",
        maxiter=de_maxiter,
        popsize=18,
        polish=True,
        seed=seed,
        updating="deferred",
        workers=1,
    )
    best = np.asarray(result.x, dtype=float)
    if np.isfinite(log_probability(best, ctx)):
        return best

    for _ in range(5000):
        trial = np.array([rng.uniform(lo, hi) for lo, hi in ctx.bounds], dtype=float)
        if np.isfinite(log_probability(trial, ctx)):
            return trial
    raise RuntimeError("Failed to find a valid starting point for the MCMC walkers.")


def initialize_walkers(best: np.ndarray, ctx: Context, walkers: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ndim = len(best)
    positions = np.empty((walkers, ndim), dtype=float)
    for i in range(walkers):
        accepted = False
        for _ in range(1000):
            scale = np.array([0.005, 0.003, 2.0e-4], dtype=float)
            trial = best + scale * rng.normal(size=ndim)
            for j, (lo, hi) in enumerate(ctx.bounds):
                trial[j] = np.clip(trial[j], lo + 1.0e-10, hi - 1.0e-10)
            if np.isfinite(log_probability(trial, ctx)):
                positions[i] = trial
                accepted = True
                break
        if not accepted:
            raise RuntimeError(f"Could not initialize walker {i} near the best-fit point.")
    return positions


def estimate_burn_and_thin(sampler: emcee.EnsembleSampler, steps: int, burn_fallback: int) -> tuple[int, int, float | None]:
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_max = float(np.max(tau))
        burn = int(min(steps - 2, max(1, math.ceil(5.0 * tau_max))))
        thin = max(1, int(math.ceil(0.5 * tau_max)))
        return burn, thin, tau_max
    except Exception:
        burn = min(steps - 2, burn_fallback)
        return burn, 1, None


def savage_dickey_bayes_factor(nu_samples: np.ndarray, nu_min: float, nu_max: float) -> float | None:
    if not (nu_min < 0.0 < nu_max):
        return None
    if len(nu_samples) < 32:
        return None
    prior_density = 1.0 / (nu_max - nu_min)
    kde = stats.gaussian_kde(nu_samples)
    posterior_density_at_zero = float(kde.evaluate([0.0])[0])
    if posterior_density_at_zero <= 0.0 or not np.isfinite(posterior_density_at_zero):
        return None
    return prior_density / posterior_density_at_zero


def summarize_result(nu_samples: np.ndarray, bayes_factor: float | None, nu_pred: float) -> FitResult:
    nu_mean = float(np.mean(nu_samples))
    nu_sigma = float(np.std(nu_samples, ddof=1))
    nu_95ci = (float(np.percentile(nu_samples, 2.5)), float(np.percentile(nu_samples, 97.5)))
    pass_pred = abs(nu_mean - nu_pred) <= 2.0 * nu_sigma
    nu_positive_2sigma = nu_mean >= 2.0 * nu_sigma
    pass_lcdm_disfavored = nu_positive_2sigma and (bayes_factor is None or bayes_factor < 3.0)

    if pass_pred and not pass_lcdm_disfavored:
        verdict = "consistent_but_not_confirmatory"
        note = (
            "The fitted ν is compatible with the 10^-3 target, but the test does not robustly disfavour ν = 0 "
            "under the stated criteria."
        )
    elif pass_pred and pass_lcdm_disfavored:
        verdict = "confirmatory_pass"
        note = "The fitted ν matches the target and the result disfavors ν = 0 strongly enough to count as a pass."
    elif (not pass_pred) and pass_lcdm_disfavored:
        verdict = "lcdm_disfavored_but_prediction_missed"
        note = "ν = 0 is disfavored, but the fitted ν misses the 10^-3 prediction band by more than 2σ."
    else:
        verdict = "not_supported"
        note = "The current fit does not satisfy the full Test 01 pass conditions."

    if bayes_factor is not None and bayes_factor >= 10.0:
        note += " Bayes factor is decisively against the model by the test's own threshold."

    return FitResult(
        nu_mean=nu_mean,
        nu_sigma=nu_sigma,
        nu_95ci=nu_95ci,
        bayes_factor=bayes_factor,
        pass_pred=pass_pred,
        pass_lcdm_disfavored=pass_lcdm_disfavored,
        nu_positive_2sigma=nu_positive_2sigma,
        verdict=verdict,
        note=note,
        burn_in=0,
        thin=1,
        tau_max=None,
    )


def write_plot(path: Path, nu_samples: np.ndarray, nu_pred: float, nu_pred_err: float) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(nu_samples, bins=60, density=True, alpha=0.65)
    ax.axvspan(nu_pred - nu_pred_err, nu_pred + nu_pred_err, alpha=0.15, label="prediction band")
    ax.axvline(0.0, linestyle="--", linewidth=1.5, label="ν = 0")
    ax.axvline(nu_pred, linestyle=":", linewidth=1.5, label="ν = 10⁻³")
    ax.set_xlabel("ν")
    ax.set_ylabel("Posterior density")
    ax.set_title("Test 01: posterior on ν")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(
    path: Path,
    result: FitResult,
    args: argparse.Namespace,
    pantheon: PantheonData,
    bao: BaoData,
    planck: PlanckCompressedPrior,
) -> None:
    lines = [
        "# Test 01 — RVM ν Coefficient from Joint Cosmological Fit",
        "",
        "## Summary",
        f"- nu_mean: {result.nu_mean:.8g}",
        f"- nu_sigma: {result.nu_sigma:.8g}",
        f"- nu_95ci: [{result.nu_95ci[0]:.8g}, {result.nu_95ci[1]:.8g}]",
        f"- bayes_factor: {result.bayes_factor if result.bayes_factor is not None else 'null'}",
        f"- pass_pred: {str(result.pass_pred).lower()}",
        f"- pass_lcdm_disfavored: {str(result.pass_lcdm_disfavored).lower()}",
        f"- verdict: {result.verdict}",
        "",
        "## Interpretation",
        result.note,
        "",
        "## Run configuration",
        f"- walkers: {args.walkers}",
        f"- steps: {args.steps}",
        f"- burn_in: {result.burn_in}",
        f"- thin: {result.thin}",
        f"- tau_max: {result.tau_max if result.tau_max is not None else 'unavailable'}",
        f"- Pantheon+ rows: {len(pantheon.mu_obs)}",
        f"- BAO rows: {len(bao.value)}",
        f"- Planck prior parameters: {planck.names}",
        f"- Planck source entry: {planck.source_label}",
        "",
        "## Data sources",
        f"- Pantheon+ table URL: {args.pantheon_url}",
        f"- Pantheon+ covariance URL: {args.pantheon_cov_url}",
        f"- DESI DR2 BAO mean URL: {args.desi_mean_url}",
        f"- DESI DR2 BAO covariance URL: {args.desi_cov_url}",
        f"- Planck PR3 cosmology zip URL: {args.planck_zip_url}",
        "",
        "## Notes",
        "This standalone script does not import or modify P01_rvm_fit_v2_fixed.py.",
        "The Planck term is derived on the fly as a compressed Gaussian prior in (omega_m, h) from the public Planck PR3 cosmology chains.",
        "The BAO sound horizon is computed with the Eisenstein-Hu approximation using fixed ω_b h² unless you edit the script.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.walkers < 2 * 3:
        parser.error("--walkers must be at least 6 for a 3-parameter ensemble sampler.")
    if args.steps <= 10:
        parser.error("--steps must be comfortably larger than 10.")
    if not (args.omega_m_min < args.omega_m_max and args.h_min < args.h_max and args.nu_min < args.nu_max):
        parser.error("Each min bound must be smaller than the corresponding max bound.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    pantheon_data_path = ensure_downloaded(
        args.pantheon_url, args.cache_dir / "Pantheon+SH0ES.dat", refresh=args.refresh_downloads
    )
    pantheon_cov_path = ensure_downloaded(
        args.pantheon_cov_url, args.cache_dir / "Pantheon+SH0ES_STAT+SYS.cov", refresh=args.refresh_downloads
    )
    desi_mean_path = ensure_downloaded(
        args.desi_mean_url, args.cache_dir / "desi_gaussian_bao_ALL_GCcomb_mean.txt", refresh=args.refresh_downloads
    )
    desi_cov_path = ensure_downloaded(
        args.desi_cov_url, args.cache_dir / "desi_gaussian_bao_ALL_GCcomb_cov.txt", refresh=args.refresh_downloads
    )
    planck_zip_path = ensure_downloaded(
        args.planck_zip_url,
        args.cache_dir / "COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip",
        refresh=args.refresh_downloads,
    )

    pantheon = load_pantheon_data(
        pantheon_data_path,
        pantheon_cov_path,
        args.sn_z_column,
        args.sn_zhel_column,
        args.sn_mu_column,
    )
    bao = load_public_desi_bao(desi_mean_path, desi_cov_path)
    planck = load_planck_prior_from_public_zip(planck_zip_path)

    ctx = Context(
        pantheon=pantheon,
        bao=bao,
        planck=planck,
        grid_size=args.distance_grid,
        omega_b_h2=args.omega_b_h2,
        tcmb=args.tcmb,
        neff=args.neff,
        bounds=((args.omega_m_min, args.omega_m_max), (args.h_min, args.h_max), (args.nu_min, args.nu_max)),
    )

    best = find_best_fit(ctx, seed=args.seed, de_maxiter=args.de_maxiter)
    start = initialize_walkers(best, ctx, walkers=args.walkers, seed=args.seed + 1)

    sampler = emcee.EnsembleSampler(args.walkers, 3, log_probability, args=[ctx])
    sampler.run_mcmc(start, args.steps, progress=args.progress)

    burn, thin, tau_max = estimate_burn_and_thin(sampler, args.steps, args.burn_fallback)
    flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    if flat_samples.shape[0] < 100:
        raise RuntimeError(
            f"Too few post-burn-in samples remain ({flat_samples.shape[0]}). Increase --steps or lower --burn-fallback."
        )

    nu_samples = flat_samples[:, 2]
    bayes_factor = savage_dickey_bayes_factor(nu_samples, args.nu_min, args.nu_max)
    result = summarize_result(nu_samples, bayes_factor, nu_pred=args.nu_pred)
    result.burn_in = burn
    result.thin = thin
    result.tau_max = tau_max

    result_payload = result.as_dict()
    result_payload["best_fit"] = {
        "omega_m": float(np.mean(flat_samples[:, 0])),
        "h": float(np.mean(flat_samples[:, 1])),
        "nu": float(np.mean(flat_samples[:, 2])),
    }
    result_payload["posterior_95ci"] = {
        "omega_m": [float(np.percentile(flat_samples[:, 0], 2.5)), float(np.percentile(flat_samples[:, 0], 97.5))],
        "h": [float(np.percentile(flat_samples[:, 1], 2.5)), float(np.percentile(flat_samples[:, 1], 97.5))],
        "nu": [float(np.percentile(flat_samples[:, 2], 2.5)), float(np.percentile(flat_samples[:, 2], 97.5))],
    }
    result_payload["config"] = {
        "walkers": args.walkers,
        "steps": args.steps,
        "nu_prediction": args.nu_pred,
        "nu_prediction_error": args.nu_pred_err,
        "planck_parameters": planck.names,
        "planck_source_entry": planck.source_label,
        "omega_b_h2": args.omega_b_h2,
        "tcmb": args.tcmb,
        "neff": args.neff,
        "data_urls": {
            "pantheon_data": args.pantheon_url,
            "pantheon_cov": args.pantheon_cov_url,
            "desi_mean": args.desi_mean_url,
            "desi_cov": args.desi_cov_url,
            "planck_zip": args.planck_zip_url,
        },
    }

    (args.out_dir / "result.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    np.savez_compressed(args.out_dir / "chain.npz", flat_samples=flat_samples)
    write_report(args.out_dir / "report.md", result, args, pantheon, bao, planck)
    if not args.no_plot:
        write_plot(args.out_dir / "nu_posterior.png", nu_samples, args.nu_pred, args.nu_pred_err)

    print(json.dumps(result_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
