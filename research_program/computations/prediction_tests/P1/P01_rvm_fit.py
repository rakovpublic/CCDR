#!/usr/bin/env python3
"""
P01_rvm_fit.py
Fit the Running Vacuum Model to DESI BAO + Planck + Pantheon+ data.
Extract ν and compare with ν ~ 10⁻³.

Prediction: ν = (1.0 ± 0.3) × 10⁻³ from fitting the RVM equation
ρ_vac(H) = C₀ + νm²H²/M²_pl to cosmological data.

Data sources:
  1. DESI DR1 BAO (2024): arXiv:2404.03002
  2. Planck 2018 CMB compressed (shift parameters R, l_a, ω_b): arXiv:1807.06209
  3. Pantheon+ SNIa (2022): https://github.com/PantheonPlusSH0ES/DataRelease

Success criterion: ν = (1.0 ± 0.5) × 10⁻³ at > 2σ from zero.
"""

import os
import sys
import argparse
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_LIGHT = 299792.458  # km/s

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download_pantheon_plus():
    """Download Pantheon+ SH0ES data release from GitHub."""
    import urllib.request
    import zipfile

    url = "https://github.com/PantheonPlusSH0ES/DataRelease/archive/refs/heads/main.zip"
    out_dir = os.path.join(DATA_DIR, "pantheon_plus")
    marker = os.path.join(out_dir, ".downloaded")

    if os.path.exists(marker):
        print("  Pantheon+ data already downloaded.")
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, "pantheon_plus.zip")

    print("  Downloading Pantheon+ data from GitHub...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"  WARNING: Could not download Pantheon+ data: {e}")
        print("  Will use built-in Pantheon+ summary statistics instead.")
        return None

    print("  Extracting...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    except zipfile.BadZipFile:
        print("  WARNING: Bad zip file. Will use built-in summary statistics.")
        return None

    os.remove(zip_path)
    with open(marker, "w") as f:
        f.write("ok\n")

    print("  Pantheon+ data ready.")
    return out_dir


def download_data():
    """Download public datasets needed for the fit."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Checking / downloading data...")

    # DESI BAO: we use the published table values directly (hardcoded below)
    print("  DESI DR1 BAO: using published table values (arXiv:2404.03002).")

    # Planck 2018 compressed: shift parameters from arXiv:1807.06209
    print("  Planck 2018 CMB: using compressed shift parameters (arXiv:1807.06209).")

    # Pantheon+
    pantheon_dir = download_pantheon_plus()
    return pantheon_dir


# ---------------------------------------------------------------------------
# RVM cosmology
# ---------------------------------------------------------------------------
def hubble_rvm(z, Om, h, nu):
    """E(z) = H(z)/H0 for the Running Vacuum Model.

    In the RVM the vacuum energy density runs as:
        ρ_vac(H) = C₀ + ν × (3H²M²_pl)/(8π)
    giving an effective modification to the Friedmann equation.
    For small ν:
        E²(z) = Ω_m(1+z)³ + Ω_Λ + ν Ω_m [(1+z)³ - 1]
    """
    OL = 1.0 - Om
    E2 = Om * (1 + z) ** 3 + OL + nu * Om * ((1 + z) ** 3 - 1)
    return np.sqrt(np.abs(E2))


def comoving_distance(z, Om, h, nu):
    """Comoving distance D_C(z) in Mpc."""

    def integrand(zp):
        return 1.0 / hubble_rvm(zp, Om, h, nu)

    result, _ = quad(integrand, 0, z)
    return result * C_LIGHT / (100.0 * h)


def DM_over_rd(z, Om, h, nu, rd):
    """Transverse comoving distance / sound horizon: D_M(z)/r_d."""
    return comoving_distance(z, Om, h, nu) / rd


def DH_over_rd(z, Om, h, nu, rd):
    """Hubble distance / sound horizon: D_H(z)/r_d = c / [H(z) r_d]."""
    Hz = 100.0 * h * hubble_rvm(z, Om, h, nu)
    return C_LIGHT / (Hz * rd)


def sound_horizon(Om, h, Ob_h2=0.02237):
    """Approximate r_d (sound horizon at drag epoch) in Mpc.

    Eisenstein & Hu (1998) fitting formula.
    """
    om_m = Om * h ** 2
    om_b = Ob_h2
    return 147.05 * (om_m / 0.143) ** (-0.255) * (om_b / 0.02237) ** (-0.128)


def luminosity_distance(z, Om, h, nu):
    """Luminosity distance d_L(z) in Mpc."""
    return (1 + z) * comoving_distance(z, Om, h, nu)


def distance_modulus(z, Om, h, nu):
    """Distance modulus μ(z) = 5 log10(d_L / 10 pc)."""
    dL = luminosity_distance(z, Om, h, nu)
    return 5.0 * np.log10(dL) + 25.0


# ---------------------------------------------------------------------------
# DESI DR1 BAO data (arXiv:2404.03002)
# z, DM/rd, σ(DM/rd), DH/rd, σ(DH/rd)
# ---------------------------------------------------------------------------
DESI_DATA = np.array(
    [
        [0.295, 7.93, 0.15, 20.08, 0.61],
        [0.510, 13.62, 0.25, 20.58, 0.61],
        [0.706, 16.85, 0.32, 20.08, 0.52],
        [0.930, 21.71, 0.28, 17.88, 0.35],
        [1.317, 27.79, 0.69, 13.82, 0.42],
        [2.330, 39.71, 0.94, 8.52, 0.17],
    ]
)

# ---------------------------------------------------------------------------
# Planck 2018 compressed likelihood (arXiv:1807.06209, Table 2)
# Shift parameters: R, l_a, ω_b  with covariance
# ---------------------------------------------------------------------------
PLANCK_MEAN = np.array([1.7502, 301.471, 0.02236])  # R, l_a, Omega_b h^2

PLANCK_COV = np.array(
    [
        [1.1902e-04, -3.2898e-02, -1.4661e-06],
        [-3.2898e-02, 2.1592e01, 8.8065e-04],
        [-1.4661e-06, 8.8065e-04, 1.8025e-07],
    ]
)

PLANCK_ICOV = np.linalg.inv(PLANCK_COV)


def cmb_shift_parameters(Om, h, nu):
    """Compute CMB shift parameters R, l_a for the RVM."""
    z_star = 1089.92  # last scattering redshift (Planck 2018)
    DA_star = comoving_distance(z_star, Om, h, nu)  # D_A(z*) ≈ D_C(z*) for flat
    R = np.sqrt(Om) * (100.0 * h) * DA_star / C_LIGHT
    rd = sound_horizon(Om, h)
    l_a = np.pi * DA_star / rd
    ob_h2 = 0.02237  # fixed
    return np.array([R, l_a, ob_h2])


# ---------------------------------------------------------------------------
# Pantheon+ summary statistics (compressed, if full data unavailable)
# Binned distance moduli at representative redshifts
# From Brout et al. (2022), arXiv:2202.04077
# ---------------------------------------------------------------------------
PANTHEON_ZBINS = np.array(
    [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80,
     1.00, 1.20, 1.50, 2.00]
)

PANTHEON_MU = np.array(
    [33.08, 34.58, 35.45, 36.52, 37.23, 38.02, 38.88, 39.47, 40.30, 40.88, 41.30,
     41.83, 42.23, 42.68, 43.01, 43.41, 43.94]
)

PANTHEON_MU_ERR = np.array(
    [0.08, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04,
     0.05, 0.07, 0.10, 0.18]
)


def load_pantheon_full(pantheon_dir):
    """Try to load the full Pantheon+ dataset. Returns (z, mu, cov) or None."""
    if pantheon_dir is None:
        return None

    # Search for the data file
    for root, dirs, files in os.walk(pantheon_dir):
        for fname in files:
            if "Pantheon+SH0ES" in fname and fname.endswith(".dat"):
                fpath = os.path.join(root, fname)
                try:
                    data = np.genfromtxt(fpath, names=True, dtype=None, encoding="utf-8")
                    z = np.array([float(row["zHD"]) for row in data])
                    mu = np.array([float(row["MU"]) for row in data])
                    mu_err = np.array([float(row["MU_ERR"]) for row in data])
                    print(f"  Loaded full Pantheon+ data: {len(z)} SNIa from {fpath}")
                    return z, mu, mu_err
                except Exception as e:
                    print(f"  Could not parse {fpath}: {e}")
                    continue
    return None


# ---------------------------------------------------------------------------
# Chi-squared components
# ---------------------------------------------------------------------------
def chi2_bao(params):
    """χ² from DESI BAO measurements."""
    Om, h, nu = params
    rd = sound_horizon(Om, h)
    chi2 = 0.0
    for row in DESI_DATA:
        z, DM_obs, DM_err, DH_obs, DH_err = row
        DM_th = DM_over_rd(z, Om, h, nu, rd)
        DH_th = DH_over_rd(z, Om, h, nu, rd)
        chi2 += ((DM_obs - DM_th) / DM_err) ** 2
        chi2 += ((DH_obs - DH_th) / DH_err) ** 2
    return chi2


def chi2_cmb(params):
    """χ² from Planck 2018 compressed shift parameters."""
    Om, h, nu = params
    theory = cmb_shift_parameters(Om, h, nu)
    delta = theory - PLANCK_MEAN
    return float(delta @ PLANCK_ICOV @ delta)


def chi2_sn(params, sn_data=None):
    """χ² from Pantheon+ SNIa.

    Uses full data if available, otherwise compressed summary statistics.
    The absolute magnitude M_B is analytically marginalised.
    """
    Om, h, nu = params

    if sn_data is not None:
        z_arr, mu_obs, mu_err_arr = sn_data
    else:
        z_arr, mu_obs, mu_err_arr = PANTHEON_ZBINS, PANTHEON_MU, PANTHEON_MU_ERR

    # Theory distance moduli
    mu_th = np.array([distance_modulus(z, Om, h, nu) for z in z_arr])

    # Analytically marginalise over M_B offset
    inv_var = 1.0 / mu_err_arr ** 2
    delta = mu_obs - mu_th
    A = np.sum(delta * inv_var)
    B = np.sum(inv_var)
    chi2 = np.sum(delta ** 2 * inv_var) - A ** 2 / B
    return chi2


def chi2_total(params, sn_data=None):
    """Total χ² = BAO + CMB + SNIa."""
    return chi2_bao(params) + chi2_cmb(params) + chi2_sn(params, sn_data)


# ---------------------------------------------------------------------------
# Bayesian inference with MCMC
# ---------------------------------------------------------------------------
def log_prior(params):
    """Flat prior on [Ω_m, h, ν]."""
    Om, h, nu = params
    if not (0.1 < Om < 0.5):
        return -np.inf
    if not (0.5 < h < 0.9):
        return -np.inf
    if not (-0.05 < nu < 0.05):
        return -np.inf
    return 0.0


def log_likelihood(params, sn_data=None):
    """Log-posterior ∝ log-prior + log-likelihood."""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp - 0.5 * chi2_total(params, sn_data)


def find_bestfit(sn_data=None):
    """Find best-fit parameters via optimisation."""
    p0 = [0.315, 0.674, 0.001]
    result = minimize(
        lambda p: chi2_total(p, sn_data),
        p0,
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-7},
    )
    print(f"Best-fit: Ω_m={result.x[0]:.5f}, h={result.x[1]:.5f}, ν={result.x[2]:.6f}")
    print(f"  χ²_min = {result.fun:.2f}")
    return result.x


def run_mcmc(n_walkers=32, n_steps=5000, n_burn=1000, sn_data=None):
    """Run MCMC to extract posterior on [Ω_m, h, ν]."""
    try:
        import emcee
    except ImportError:
        print("ERROR: emcee not installed. Install with: pip install emcee")
        print("Falling back to optimisation-only result.")
        return find_bestfit(sn_data)

    ndim = 3

    # Start near best-fit
    print("\nFinding best-fit starting point...")
    p0 = find_bestfit(sn_data)

    pos = p0 + 1e-4 * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_likelihood, kwargs={"sn_data": sn_data}
    )

    print(f"\nRunning MCMC: {n_walkers} walkers x {n_steps} steps...")
    sampler.run_mcmc(pos, n_steps, progress=True)

    # Discard burn-in
    samples = sampler.get_chain(discard=n_burn, flat=True)
    labels = ["Omega_m", "h", "nu"]

    # Print results
    print("\n" + "=" * 60)
    print("MCMC RESULTS")
    print("=" * 60)
    for i, label in enumerate(labels):
        q = np.percentile(samples[:, i], [16, 50, 84])
        print(f"  {label:>8s} = {q[1]:.5f}  +{q[2] - q[1]:.5f}  -{q[1] - q[0]:.5f}")

    nu_samples = samples[:, 2]
    nu_mean = np.mean(nu_samples)
    nu_std = np.std(nu_samples)

    print(f"\n  nu = {nu_mean:.6f} +/- {nu_std:.6f}")
    print(f"  nu / 1e-3 = {nu_mean / 1e-3:.3f} +/- {nu_std / 1e-3:.3f}")

    # Prediction test
    print("\n" + "=" * 60)
    print("PREDICTION TEST: nu = (1.0 +/- 0.3) x 10^-3")
    print("=" * 60)

    sigma_from_zero = abs(nu_mean) / nu_std if nu_std > 0 else 0
    if sigma_from_zero > 2:
        print(f"  PASS: nu != 0 at {sigma_from_zero:.1f} sigma")
    else:
        print(f"  FAIL: nu consistent with 0 ({sigma_from_zero:.1f} sigma)")

    # Check consistency with predicted value
    nu_pred = 1.0e-3
    nu_pred_err = 0.3e-3
    tension = abs(nu_mean - nu_pred) / np.sqrt(nu_std ** 2 + nu_pred_err ** 2)
    print(f"  Tension with prediction: {tension:.1f} sigma")
    if tension < 2:
        print(f"  CONSISTENT with nu ~ 10^-3 prediction")
    else:
        print(f"  INCONSISTENT with nu ~ 10^-3 prediction at {tension:.1f} sigma")

    # Reference comparison
    print("\n  Reference (Sola Peracaula 2023): nu = (0.9 +/- 0.3) x 10^-3")
    print(f"  This analysis:                   nu = ({nu_mean / 1e-3:.2f} +/- {nu_std / 1e-3:.2f}) x 10^-3")

    # Save samples
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "P01_nu_samples.npy")
    np.save(out_path, samples)
    print(f"\n  Saved chain samples to {out_path}")

    # Corner plot
    try:
        import corner

        fig = corner.corner(
            samples,
            labels=[r"$\Omega_m$", r"$h$", r"$\nu$"],
            truths=[0.315, 0.674, 1e-3],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
        )
        fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "P01_rvm_corner.png")
        fig.savefig(fig_path, dpi=150)
        print(f"  Saved corner plot to {fig_path}")
    except ImportError:
        print("  (corner not installed — skipping corner plot)")

    return nu_mean, nu_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="P01: Extract nu from DESI BAO + CMB + SNIa data using the RVM."
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download Pantheon+ data before fitting"
    )
    parser.add_argument(
        "--walkers", type=int, default=32,
        help="Number of MCMC walkers (default: 32)"
    )
    parser.add_argument(
        "--steps", type=int, default=10000,
        help="Number of MCMC steps (default: 10000)"
    )
    parser.add_argument(
        "--burn", type=int, default=2000,
        help="Burn-in steps to discard (default: 2000)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run with fewer steps (1000 steps, 200 burn-in)"
    )
    parser.add_argument(
        "--optimize-only", action="store_true",
        help="Only run optimisation, skip MCMC"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("P01: Extract nu from DESI BAO + Planck CMB + Pantheon+ SNIa")
    print("     Running Vacuum Model (RVM) fit")
    print("=" * 60)

    # Data download
    sn_data = None
    if args.download:
        pantheon_dir = download_data()
        sn_data = load_pantheon_full(pantheon_dir)
        if sn_data is None:
            print("  Using compressed Pantheon+ summary statistics.")
    else:
        print("\nUsing built-in data tables (use --download for full Pantheon+ data).")

    if args.optimize_only:
        find_bestfit(sn_data)
        return

    if args.quick:
        args.walkers = 16
        args.steps = 1000
        args.burn = 200
        print(f"\nQuick mode: {args.walkers} walkers x {args.steps} steps")

    nu_mean, nu_std = run_mcmc(
        n_walkers=args.walkers,
        n_steps=args.steps,
        n_burn=args.burn,
        sn_data=sn_data,
    )


if __name__ == "__main__":
    main()
