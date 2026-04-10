#!/usr/bin/env python3
"""
P01_rvm_fit_v2.py

Running-vacuum-model (RVM) fit with:
  1. Planck 2018 CMB distance-prior covariance
  2. Pantheon+ full covariance (statistical + systematic)
  3. Combined BAO data with overlap handling baked into the dataset
  4. MCMC with emcee
  5. Pre-computed covariance factorizations and cached distance grids

This script is based on the P01 v2 specification supplied by the user.
It intentionally follows the simplified late-time flat RVM setup from that
specification:
  * fixed z_* and fixed r_d
  * fixed omega_b in the CMB prior vector
  * flat universe only
  * no radiation sector in E(z)

Those approximations are useful for a fast public-data consistency test, but
should not be treated as a full Boltzmann-code-level cosmological analysis.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Iterable

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import differential_evolution

try:
    import emcee
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'emcee'. Install with: pip install emcee"
    ) from exc


# ============================================================
# CONSTANTS
# ============================================================
C_KMS = 2.99792458e5  # km/s
Z_STAR = 1089.92
R_D_FIDUCIAL = 147.09  # Mpc
DEFAULT_SEED = 42


# ============================================================
# PLANCK 2018 CMB DISTANCE PRIORS (from the provided spec)
# ============================================================
CMB_MEAN = np.array([1.7502, 301.471, 0.02236], dtype=float)
CMB_SIGMA = np.array([0.0046, 0.090, 0.00015], dtype=float)
CMB_CORR = np.array(
    [
        [1.0000, 0.4597, -0.4832],
        [0.4597, 1.0000, -0.5765],
        [-0.4832, -0.5765, 1.0000],
    ],
    dtype=float,
)
CMB_COV = np.diag(CMB_SIGMA) @ CMB_CORR @ np.diag(CMB_SIGMA)
CMB_CHO = cho_factor(CMB_COV, lower=True, check_finite=False)


# ============================================================
# BAO DATA (from the provided spec, with overlap already resolved)
# ============================================================
# Format: (z, observable_type, value, error)
# observable_type in {"DV_rd", "DM_rd", "DH_rd", "rd_DV"}
BAO_DATA: list[tuple[float, str, float, float]] = [
    (0.106, "rd_DV", 0.336, 0.015),
    (0.150, "DV_rd", 4.466, 0.168),
    (0.610, "DM_rd", 15.15, 0.23),
    (0.610, "DH_rd", 20.68, 0.52),
    (0.300, "DV_rd", 7.93, 0.15),
    (0.510, "DV_rd", 13.62, 0.25),
    (0.706, "DM_rd", 16.85, 0.32),
    (0.706, "DH_rd", 20.09, 0.49),
    (0.930, "DM_rd", 21.71, 0.28),
    (0.930, "DH_rd", 17.88, 0.35),
    (1.317, "DM_rd", 27.79, 0.69),
    (1.317, "DH_rd", 13.82, 0.42),
    (1.491, "DM_rd", 30.69, 1.01),
    (1.491, "DH_rd", 13.16, 0.54),
    (2.330, "DM_rd", 39.71, 0.94),
    (2.330, "DH_rd", 8.52, 0.17),
]


# ============================================================
# DATA URLS
# ============================================================
PANTHEON_URL = (
    "https://github.com/PantheonPlusSH0ES/DataRelease/raw/main/"
    "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
)
PANTHEON_COV_URL = (
    "https://github.com/PantheonPlusSH0ES/DataRelease/raw/main/"
    "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"
)


@dataclass(slots=True)
class PantheonData:
    z_cmb: np.ndarray
    mu_obs: np.ndarray
    mu_err: np.ndarray
    cov: np.ndarray
    cho: tuple[np.ndarray, bool]

    @property
    def n(self) -> int:
        return int(self.z_cmb.size)


@dataclass(slots=True)
class DistanceCache:
    z_grid: np.ndarray
    dc_grid: np.ndarray

    def dc(self, z: np.ndarray | float) -> np.ndarray | float:
        return np.interp(z, self.z_grid, self.dc_grid)


class RVMModel:
    """Flat late-time RVM model with cached distance grids."""

    def __init__(self, interp_points: int = 5000) -> None:
        if interp_points < 100:
            raise ValueError("interp_points must be >= 100")
        self.interp_points = int(interp_points)
        self._cache_key: tuple[float, float, float] | None = None
        self._cache_val: DistanceCache | None = None

    @staticmethod
    def e2(z: np.ndarray | float, omega_m: float, nu: float) -> np.ndarray | float:
        if abs(nu) >= 0.999:
            if np.isscalar(z):
                return 1e30
            return np.full_like(np.asarray(z, dtype=float), 1e30)
        om_eff = omega_m / (1.0 - nu)
        return (1.0 - om_eff) + om_eff * np.power(1.0 + np.asarray(z), 3.0 * (1.0 - nu))

    @classmethod
    def e(cls, z: np.ndarray | float, omega_m: float, nu: float) -> np.ndarray | float:
        e2 = cls.e2(z, omega_m, nu)
        if np.isscalar(e2):
            return math.sqrt(e2) if e2 > 0.0 else 1e15
        bad = e2 <= 0.0
        out = np.sqrt(np.where(bad, 1.0, e2))
        out[bad] = 1e15
        return out

    def _build_cache(self, omega_m: float, h: float, nu: float, z_max: float = 2.5) -> DistanceCache:
        # Linear grid is sufficient for SN/BAO redshifts and dramatically faster
        # than integrating each SN point independently.
        z_grid = np.linspace(0.0, z_max, self.interp_points)
        inv_e = 1.0 / self.e(z_grid, omega_m, nu)
        integ = cumulative_trapezoid(inv_e, z_grid, initial=0.0)
        dc_grid = (C_KMS / (100.0 * h)) * integ
        return DistanceCache(z_grid=z_grid, dc_grid=dc_grid)

    def _get_cache(self, omega_m: float, h: float, nu: float) -> DistanceCache:
        key = (round(omega_m, 12), round(h, 12), round(nu, 12))
        if self._cache_key != key or self._cache_val is None:
            self._cache_key = key
            self._cache_val = self._build_cache(omega_m, h, nu)
        return self._cache_val

    def comoving_distance(self, z: np.ndarray | float, omega_m: float, h: float, nu: float) -> np.ndarray | float:
        cache = self._get_cache(omega_m, h, nu)
        z_arr = np.asarray(z)
        if np.any(z_arr > cache.z_grid[-1]):
            raise ValueError(f"Requested z={float(np.max(z_arr)):.3f} exceeds interpolation grid")
        return cache.dc(z)

    def comoving_distance_highz(self, z: float, omega_m: float, h: float, nu: float) -> float:
        # Keep the CMB-distance-prior computation accurate enough by direct integration.
        integrand = lambda zp: 1.0 / float(self.e(zp, omega_m, nu))
        result, _ = quad(integrand, 0.0, float(z), epsabs=1e-8, epsrel=1e-8, limit=500)
        return (C_KMS / (100.0 * h)) * result

    def luminosity_distance(self, z: np.ndarray | float, omega_m: float, h: float, nu: float) -> np.ndarray | float:
        dc = self.comoving_distance(z, omega_m, h, nu)
        return (1.0 + np.asarray(z)) * dc

    def distance_modulus(self, z: np.ndarray, omega_m: float, h: float, nu: float) -> np.ndarray:
        dl = np.asarray(self.luminosity_distance(z, omega_m, h, nu), dtype=float)
        if np.any(dl <= 0.0):
            raise ValueError("Encountered non-positive luminosity distance")
        return 5.0 * np.log10(dl) + 25.0


# ============================================================
# DATA LOADING
# ============================================================
def download_file(url: str, filepath: Path, *, timeout: int = 90) -> None:
    if filepath.exists() and filepath.stat().st_size > 1000:
        print(f"  [cache] {filepath.name}")
        return
    print(f"  [download] {filepath.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (P01-v2)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    filepath.write_bytes(data)
    print(f"  [saved] {filepath.name} ({len(data) / 1e6:.2f} MB)")


def ensure_data(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    pantheon_dat = data_dir / "Pantheon+SH0ES.dat"
    pantheon_cov = data_dir / "Pantheon+SH0ES_STAT+SYS.cov"
    download_file(PANTHEON_URL, pantheon_dat)
    download_file(PANTHEON_COV_URL, pantheon_cov)
    return pantheon_dat, pantheon_cov


def _detect_columns(header_line: str) -> dict[str, int]:
    header = header_line.lstrip("#").strip()
    if not header:
        return {}
    columns = header.replace(",", " ").split()
    idx = {name: i for i, name in enumerate(columns)}
    aliases = {}
    for name in ("zHD", "zCMB", "zHEL", "MU", "MUERR"):
        if name in idx:
            aliases[name] = idx[name]
    return aliases


def load_pantheon_data(dat_path: Path, cov_path: Path, *, jitter: float = 1e-10) -> PantheonData:
    z_list: list[float] = []
    mu_list: list[float] = []
    mu_err_list: list[float] = []

    colmap: dict[str, int] = {}
    with dat_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if not colmap:
                    maybe = _detect_columns(stripped)
                    if maybe:
                        colmap = maybe
                continue

            parts = stripped.replace(",", " ").split()
            try:
                if {"zHD", "MU", "MUERR"}.issubset(colmap):
                    z = float(parts[colmap["zHD"]])
                    mu = float(parts[colmap["MU"]])
                    mu_err = float(parts[colmap["MUERR"]])
                else:
                    # Fall back to the column convention noted in the supplied spec.
                    z = float(parts[1])
                    mu = float(parts[3])
                    mu_err = float(parts[4])
            except (IndexError, ValueError):
                continue

            if 0.0 < z < 3.0 and 5.0 < mu < 60.0 and 0.0 < mu_err < 10.0:
                z_list.append(z)
                mu_list.append(mu)
                mu_err_list.append(mu_err)

    if not z_list:
        raise RuntimeError("Could not parse any Pantheon+ rows from the data table")

    z_arr = np.asarray(z_list, dtype=float)
    mu_arr = np.asarray(mu_list, dtype=float)
    mu_err_arr = np.asarray(mu_err_list, dtype=float)
    n = z_arr.size
    print(f"  Loaded {n} Pantheon+ supernovae")

    with cov_path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
        try:
            n_cov = int(first)
        except ValueError as exc:
            raise RuntimeError(f"First line of covariance file is not an integer: {first!r}") from exc
        cov_vals: list[float] = []
        for line in f:
            cov_vals.extend(float(x) for x in line.split())

    cov_flat = np.asarray(cov_vals, dtype=float)
    expected = n_cov * n_cov
    if cov_flat.size != expected:
        raise RuntimeError(
            f"Pantheon+ covariance has {cov_flat.size} numbers but expected {expected}"
        )

    cov = cov_flat.reshape((n_cov, n_cov))
    if n_cov != n:
        if n_cov < n:
            raise RuntimeError(f"Covariance dimension {n_cov} is smaller than data length {n}")
        print(f"  Warning: trimming covariance {n_cov}x{n_cov} to {n}x{n}")
        cov = cov[:n, :n]

    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    cov += np.eye(n, dtype=float) * jitter
    cho = cho_factor(cov, lower=True, check_finite=False)
    print(f"  Cholesky factorization ready for {n}x{n} covariance")

    return PantheonData(z_cmb=z_arr, mu_obs=mu_arr, mu_err=mu_err_arr, cov=cov, cho=cho)


# ============================================================
# OPTIONAL BINNING FOR SPEED
# ============================================================
def bin_pantheon(data: PantheonData, n_bins: int) -> PantheonData:
    if n_bins <= 1 or n_bins >= data.n:
        return data

    order = np.argsort(data.z_cmb)
    z = data.z_cmb[order]
    mu = data.mu_obs[order]
    cov = data.cov[np.ix_(order, order)]

    edges = np.linspace(0, data.n, n_bins + 1, dtype=int)
    groups = [np.arange(edges[i], edges[i + 1]) for i in range(n_bins) if edges[i + 1] > edges[i]]
    B = np.zeros((len(groups), data.n), dtype=float)
    for i, g in enumerate(groups):
        B[i, g] = 1.0

    cov_bin = B @ cov @ B.T
    cov_bin = 0.5 * (cov_bin + cov_bin.T) + np.eye(len(groups)) * 1e-12

    z_bin = np.array([np.mean(z[g]) for g in groups], dtype=float)
    mu_bin = np.array([np.mean(mu[g]) for g in groups], dtype=float)
    mu_err_bin = np.sqrt(np.diag(cov_bin))
    cho_bin = cho_factor(cov_bin, lower=True, check_finite=False)

    print(f"  Binned Pantheon+: {data.n} -> {len(groups)} redshift bins")
    return PantheonData(z_cmb=z_bin, mu_obs=mu_bin, mu_err=mu_err_bin, cov=cov_bin, cho=cho_bin)


# ============================================================
# CHI-SQUARED TERMS
# ============================================================
def chi2_cmb(model: RVMModel, omega_m: float, h: float, nu: float) -> float:
    dc_star = model.comoving_distance_highz(Z_STAR, omega_m, h, nu)
    r_model = math.sqrt(omega_m) * 100.0 * h * dc_star / C_KMS
    l_a_model = math.pi * dc_star / R_D_FIDUCIAL
    omega_b_model = 0.02236

    delta = np.array([r_model, l_a_model, omega_b_model], dtype=float) - CMB_MEAN
    x = cho_solve(CMB_CHO, delta, check_finite=False)
    return float(delta @ x)


def chi2_bao(model: RVMModel, omega_m: float, h: float, nu: float) -> float:
    chi2 = 0.0
    z_values = np.array([row[0] for row in BAO_DATA], dtype=float)
    dc_values = np.asarray(model.comoving_distance(z_values, omega_m, h, nu), dtype=float)
    hz_values = 100.0 * h * np.asarray(model.e(z_values, omega_m, nu), dtype=float)

    for (z, obs_type, value, error), dc, hz in zip(BAO_DATA, dc_values, hz_values, strict=True):
        dh = C_KMS / hz
        dv = np.power(dc * dc * z * C_KMS / hz, 1.0 / 3.0)
        if obs_type == "DV_rd":
            pred = dv / R_D_FIDUCIAL
        elif obs_type == "DM_rd":
            pred = dc / R_D_FIDUCIAL
        elif obs_type == "DH_rd":
            pred = dh / R_D_FIDUCIAL
        elif obs_type == "rd_DV":
            pred = R_D_FIDUCIAL / dv
        else:  # pragma: no cover
            raise ValueError(f"Unknown BAO observable type: {obs_type}")
        chi2 += ((pred - value) / error) ** 2
    return float(chi2)


def chi2_sn(model: RVMModel, sn: PantheonData, omega_m: float, h: float, nu: float) -> float:
    mu_model = model.distance_modulus(sn.z_cmb, omega_m, h, nu)
    delta = sn.mu_obs - mu_model
    ones = np.ones_like(delta)

    c_inv_delta = cho_solve(sn.cho, delta, check_finite=False)
    c_inv_ones = cho_solve(sn.cho, ones, check_finite=False)

    a = float(delta @ c_inv_delta)
    b = float(delta @ c_inv_ones)
    d = float(ones @ c_inv_ones)
    return a - (b * b) / d


# ============================================================
# POSTERIOR
# ============================================================
def log_prior(theta: Iterable[float]) -> float:
    omega_m, h, nu = theta
    if not (0.1 < omega_m < 0.5):
        return -np.inf
    if not (0.55 < h < 0.85):
        return -np.inf
    if not (-0.05 < nu < 0.05):
        return -np.inf
    return 0.0


def log_likelihood(theta: np.ndarray, model: RVMModel, sn: PantheonData) -> float:
    omega_m, h, nu = map(float, theta)
    try:
        total = (
            chi2_cmb(model, omega_m, h, nu)
            + chi2_bao(model, omega_m, h, nu)
            + chi2_sn(model, sn, omega_m, h, nu)
        )
    except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
        return -np.inf
    if not np.isfinite(total):
        return -np.inf
    return -0.5 * total


def log_posterior(theta: np.ndarray, model: RVMModel, sn: PantheonData) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model, sn)


# ============================================================
# UTILITIES
# ============================================================
def summarize_samples(samples: np.ndarray) -> dict[str, float]:
    median = float(np.median(samples))
    lo, hi = np.percentile(samples, [16.0, 84.0])
    return {
        "median": median,
        "lo": float(lo),
        "hi": float(hi),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def try_make_plots(out_dir: Path, sampler: emcee.EnsembleSampler, flat_samples: np.ndarray, nu_pred: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    chain = sampler.get_chain()
    nu_chain = chain[:, :, 2]
    om_samples = flat_samples[:, 0]
    h_samples = flat_samples[:, 1]
    nu_samples = flat_samples[:, 2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(nu_chain, alpha=0.08)
    axes[0, 0].axhline(0.0, ls="--", alpha=0.6)
    axes[0, 0].axhline(nu_pred, ls=":", alpha=0.8)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel(r"$\nu$")
    axes[0, 0].set_title("Trace for $\\nu$")

    axes[0, 1].hist(nu_samples, bins=60, density=True, alpha=0.75)
    axes[0, 1].axvline(0.0, ls="--", alpha=0.7)
    axes[0, 1].axvline(nu_pred, ls=":", alpha=0.9)
    axes[0, 1].axvline(np.mean(nu_samples), alpha=0.9)
    axes[0, 1].set_xlabel(r"$\nu$")
    axes[0, 1].set_ylabel("Posterior density")
    axes[0, 1].set_title(r"Posterior of $\nu$")

    axes[1, 0].scatter(om_samples[::10], nu_samples[::10], s=1, alpha=0.05)
    axes[1, 0].axhline(0.0, ls="--", alpha=0.6)
    axes[1, 0].axhline(nu_pred, ls=":", alpha=0.8)
    axes[1, 0].set_xlabel(r"$\Omega_m$")
    axes[1, 0].set_ylabel(r"$\nu$")
    axes[1, 0].set_title(r"$\Omega_m$ vs $\nu$")

    axes[1, 1].scatter(h_samples[::10], nu_samples[::10], s=1, alpha=0.05)
    axes[1, 1].axhline(0.0, ls="--", alpha=0.6)
    axes[1, 1].axhline(nu_pred, ls=":", alpha=0.8)
    axes[1, 1].set_xlabel("h")
    axes[1, 1].set_ylabel(r"$\nu$")
    axes[1, 1].set_title("h vs $\\nu$")

    plt.tight_layout()
    fig.savefig(out_dir / "P01_rvm_v2.png", dpi=150)
    plt.close(fig)
    print(f"[plot] Saved {(out_dir / 'P01_rvm_v2.png').name}")

    try:
        import corner
    except ImportError:
        print("[plot] corner not installed; skipping corner plot")
        return

    fig = corner.corner(
        flat_samples,
        labels=[r"$\Omega_m$", "h", r"$\nu$"],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".5f",
    )
    fig.savefig(out_dir / "P01_rvm_v2_corner.png", dpi=150)
    plt.close(fig)
    print(f"[plot] Saved {(out_dir / 'P01_rvm_v2_corner.png').name}")


# ============================================================
# MAIN
# ============================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RVM nu extraction with full Pantheon+ covariance and CMB distance priors"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/p01_rvm_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/p01_rvm_v2"))
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--burn", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--interp-points", type=int, default=5000)
    parser.add_argument("--de-maxiter", type=int, default=200)
    parser.add_argument("--nu-pred", type=float, default=1.0e-3)
    parser.add_argument("--nu-pred-err", type=float, default=0.3e-3)
    parser.add_argument(
        "--bin-pantheon",
        type=int,
        default=0,
        help="Optionally compress Pantheon+ into this many redshift bins for speed",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG outputs even if matplotlib/corner are installed",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("P01 v2: RVM ν Extraction")
    print("  Full Planck prior covariance + full Pantheon+ covariance + BAO")
    print("=" * 72)

    print("\n[data] Ensuring Pantheon+ files are present...")
    pantheon_dat, pantheon_cov = ensure_data(args.data_dir)

    print("\n[data] Loading Pantheon+...")
    sn = load_pantheon_data(pantheon_dat, pantheon_cov)
    if args.bin_pantheon:
        sn = bin_pantheon(sn, args.bin_pantheon)

    model = RVMModel(interp_points=args.interp_points)

    def neg_loglike(theta: np.ndarray) -> float:
        val = log_likelihood(theta, model, sn)
        return np.inf if not np.isfinite(val) else -val

    print("\n[optimize] Differential-evolution search for a starting point...")
    bounds = [(0.2, 0.4), (0.60, 0.80), (-0.03, 0.03)]
    t0 = time()
    opt = differential_evolution(
        neg_loglike,
        bounds=bounds,
        seed=args.seed,
        maxiter=args.de_maxiter,
        polish=True,
        tol=1e-6,
        updating="deferred",
        workers=1,
    )
    opt_elapsed = time() - t0
    p_best = np.asarray(opt.x, dtype=float)
    chi2_best = 2.0 * float(opt.fun)
    print(
        f"  Best fit: Ω_m={p_best[0]:.5f}, h={p_best[1]:.5f}, ν={p_best[2]:.6f}"
    )
    print(f"  χ²_min = {chi2_best:.3f}")
    print(f"  Optimizer wall time: {opt_elapsed / 60.0:.2f} min")

    ndim = 3
    walkers = int(args.walkers)
    if walkers < 2 * ndim:
        raise SystemExit("--walkers must be at least 2 * ndim = 6")
    if args.burn >= args.steps:
        raise SystemExit("--burn must be smaller than --steps")

    p0 = p_best + 1e-4 * rng.standard_normal((walkers, ndim))

    print(f"\n[mcmc] Running {walkers} walkers × {args.steps} steps")
    sampler = emcee.EnsembleSampler(walkers, ndim, log_posterior, args=(model, sn))
    t1 = time()
    sampler.run_mcmc(p0, args.steps, progress=True)
    elapsed = time() - t1
    print(f"  MCMC wall time: {elapsed / 3600.0:.2f} h")

    print("\n[diagnostics]")
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        tau_max = float(np.max(tau))
        thin = max(1, int(round(tau_max / 2.0)))
        n_eff = int(walkers * args.steps / tau_max)
        print(
            "  Autocorr times: "
            f"Ω_m={tau[0]:.1f}, h={tau[1]:.1f}, ν={tau[2]:.1f}"
        )
    except Exception as exc:
        print(f"  Autocorr estimation warning: {exc}")
        tau = np.array([args.steps / 10.0] * ndim, dtype=float)
        tau_max = float(np.max(tau))
        thin = max(1, int(round(tau_max / 2.0)))
        n_eff = int(walkers * args.steps / max(tau_max, 1.0))

    af = sampler.acceptance_fraction
    print(f"  Mean acceptance fraction: {np.mean(af):.3f}")
    print(f"  Suggested thinning: {thin}")
    print(f"  Rough effective sample count: {n_eff}")

    flat_samples = sampler.get_chain(discard=args.burn, thin=thin, flat=True)
    if flat_samples.size == 0:
        raise RuntimeError("No posterior samples remain after burn-in/thinning")
    print(f"  Retained posterior samples: {flat_samples.shape[0]}")

    om_samples = flat_samples[:, 0]
    h_samples = flat_samples[:, 1]
    nu_samples = flat_samples[:, 2]

    om_sum = summarize_samples(om_samples)
    h_sum = summarize_samples(h_samples)
    nu_sum = summarize_samples(nu_samples)
    nu_95 = np.percentile(nu_samples, [2.5, 97.5])
    z_from_zero = abs(nu_sum["mean"]) / nu_sum["std"] if nu_sum["std"] > 0 else np.inf
    z_from_pred = abs(nu_sum["mean"] - args.nu_pred) / nu_sum["std"] if nu_sum["std"] > 0 else np.inf

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(
        f"Ω_m = {om_sum['median']:.5f} +{om_sum['hi'] - om_sum['median']:.5f} "
        f"-{om_sum['median'] - om_sum['lo']:.5f}"
    )
    print(
        f"h   = {h_sum['median']:.5f} +{h_sum['hi'] - h_sum['median']:.5f} "
        f"-{h_sum['median'] - h_sum['lo']:.5f}"
    )
    print(
        f"ν   = {nu_sum['median']:.6f} +{nu_sum['hi'] - nu_sum['median']:.6f} "
        f"-{nu_sum['median'] - nu_sum['lo']:.6f}"
    )
    print(f"\nν = {nu_sum['mean']:.6f} ± {nu_sum['std']:.6f}")
    print(f"ν / 10⁻³ = {nu_sum['mean'] * 1e3:.3f} ± {nu_sum['std'] * 1e3:.3f}")

    print("\n" + "=" * 72)
    print("PREDICTION TEST")
    print("=" * 72)
    print(f"ν from zero:       {z_from_zero:.2f}σ")
    print(f"ν from prediction: {z_from_pred:.2f}σ")
    print(f"95% CI for ν:      [{nu_95[0]:.6f}, {nu_95[1]:.6f}]")
    print(f"Prediction ν in 95% CI: {'YES' if nu_95[0] <= args.nu_pred <= nu_95[1] else 'NO'}")
    print(f"ΛCDM ν=0 in 95% CI:     {'YES' if nu_95[0] <= 0.0 <= nu_95[1] else 'NO'}")

    bayes_factor = math.nan
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(nu_samples)
        posterior_density_at_zero = float(kde(0.0))
        prior_density_at_zero = 1.0 / 0.10
        bayes_factor = posterior_density_at_zero / prior_density_at_zero
        print(f"Bayes factor (ΛCDM vs RVM): {bayes_factor:.3f}")
    except Exception as exc:
        print(f"Bayes factor computation skipped: {exc}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "config": {
            "walkers": walkers,
            "steps": int(args.steps),
            "burn": int(args.burn),
            "thin": int(thin),
            "seed": int(args.seed),
            "interp_points": int(args.interp_points),
            "bin_pantheon": int(args.bin_pantheon),
            "nu_pred": float(args.nu_pred),
            "nu_pred_err": float(args.nu_pred_err),
        },
        "omega_m": om_sum,
        "h": h_sum,
        "nu": {
            **nu_sum,
            "ci_95": [float(nu_95[0]), float(nu_95[1])],
        },
        "chi2_min": float(chi2_best),
        "n_sn": int(sn.n),
        "n_bao": int(len(BAO_DATA)),
        "diagnostics": {
            "acceptance_fraction_mean": float(np.mean(af)),
            "acceptance_fraction_min": float(np.min(af)),
            "acceptance_fraction_max": float(np.max(af)),
            "autocorr_time": [float(x) for x in tau],
            "n_effective_rough": int(n_eff),
            "optimizer_minutes": float(opt_elapsed / 60.0),
            "mcmc_hours": float(elapsed / 3600.0),
        },
        "tests": {
            "z_from_zero": float(z_from_zero),
            "z_from_prediction": float(z_from_pred),
            "prediction_in_95CI": bool(nu_95[0] <= args.nu_pred <= nu_95[1]),
            "lcdm_in_95CI": bool(nu_95[0] <= 0.0 <= nu_95[1]),
            "bayes_factor_lcdm_vs_rvm": None if not np.isfinite(bayes_factor) else float(bayes_factor),
        },
        "notes": {
            "approximation": (
                "Late-time flat RVM with fixed z_star, fixed r_d, fixed omega_b, "
                "and no explicit radiation component, matching the supplied spec."
            )
        },
    }

    save_json(out_dir / "result_v2.json", result)
    np.savez_compressed(out_dir / "chain_v2.npz", flat_samples=flat_samples, chain=sampler.get_chain())
    print(f"\n[save] Wrote {(out_dir / 'result_v2.json')} and {(out_dir / 'chain_v2.npz')}")

    if not args.no_plots:
        try_make_plots(out_dir, sampler, flat_samples, args.nu_pred)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
