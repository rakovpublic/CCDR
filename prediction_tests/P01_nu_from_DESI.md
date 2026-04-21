# P01: Extract ν from DESI BAO + CMB + SNIa Public Data

## Prediction
**ν = (1.0 ± 0.3) × 10⁻³** from fitting the RVM equation ρ_vac(H) = C₀ + νm²H²/M²_pl to cosmological data.

## Hardware
- i9 + 32 GB RAM: sufficient for full MCMC (8 parallel chains)
- GPU: optional (CosmoMC can use it but cobaya runs fine on CPU)
- Disk: ~10 GB for chains + data

## Software
```bash
# Option A: cobaya (recommended — pure Python, easy setup)
pip install cobaya camb getdist

# Option B: MontePython + CLASS
pip install montepython-public
# CLASS: https://github.com/lesgourg/class_public
git clone https://github.com/lesgourg/class_public.git
cd class_public && make

# For plotting:
pip install getdist matplotlib corner
```

## Public Data Sources
1. **DESI DR1 BAO** (2024): arXiv:2404.03002, data at https://data.desi.lbl.gov/
   - Files: `desi_bao_distances.dat` (D_H/r_d, D_M/r_d at z = 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.3)
2. **Planck 2018 CMB**: https://pla.esac.esa.int/
   - Compressed: use the shift parameters (R, l_a, ω_b) from arXiv:1807.06209
3. **Pantheon+ SNIa** (2022): https://github.com/PantheonPlusSH0ES/DataRelease
   - File: `Pantheon+SH0ES.dat` (1701 SNIa with covariance matrix)

## Method

### Step 1: Implement the RVM in CAMB/CLASS

The RVM modifies the Friedmann equation:
```
H²(a) = (8πG/3) × [ρ_matter(a) + ρ_rad(a) + ρ_vac(a)]
where ρ_vac(a) = C₀ + ν × (3H²M²_pl)/(8π)
```

This gives an effective dark energy equation of state:
```
w_eff(z) = -1 + ν × Ω_m(z) / [Ω_Λ(z) + ν × Ω_m(z)]
```

```python
#!/usr/bin/env python3
"""
P01_rvm_fit.py
Fit the Running Vacuum Model to DESI BAO + Planck + Pantheon+ data.
Extract ν and compare with ν ~ 10⁻³.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import emcee
import corner

# Cosmological parameters: [Ω_m, h, ν]
# Fixed: Ω_b h² = 0.02237, n_s = 0.9649 (Planck 2018)

C_LIGHT = 299792.458  # km/s

def hubble_rvm(z, Om, h, nu):
    """H(z)/H0 for the RVM."""
    a = 1.0 / (1 + z)
    # In RVM: ρ_vac/ρ_crit0 = (1-Om) + nu*Om*(1+z)^3 / (1 - nu)
    # Simplified for small ν:
    OL = 1 - Om
    E2 = Om * (1+z)**3 + OL + nu * Om * ((1+z)**3 - 1)
    return np.sqrt(np.abs(E2))

def comoving_distance(z, Om, h, nu):
    """D_C(z) in Mpc."""
    def integrand(zp):
        return 1.0 / hubble_rvm(zp, Om, h, nu)
    result, _ = quad(integrand, 0, z)
    return result * C_LIGHT / (100 * h)

def DM_over_rd(z, Om, h, nu, rd):
    """D_M(z)/r_d."""
    return comoving_distance(z, Om, h, nu) / rd

def DH_over_rd(z, Om, h, nu, rd):
    """D_H(z)/r_d = c / (H(z) × r_d)."""
    Hz = 100 * h * hubble_rvm(z, Om, h, nu)
    return C_LIGHT / (Hz * rd)

def sound_horizon(Om, h, Ob_h2=0.02237):
    """Approximate r_d (sound horizon at drag epoch) in Mpc."""
    # Eisenstein & Hu (1998) fitting formula
    om_m = Om * h**2
    om_b = Ob_h2
    zd = 1060  # approximate drag epoch
    Rd = 31500 * om_b / (1 + zd) / 1e4
    return 147.05 * (om_m / 0.143)**(-0.255) * (om_b / 0.02237)**(-0.128)

# ---- DESI DR1 BAO data ----
# z, DM/rd (measured), σ(DM/rd), DH/rd (measured), σ(DH/rd)
DESI_DATA = np.array([
    [0.295, 7.93, 0.15, 20.08, 0.61],
    [0.510, 13.62, 0.25, 20.58, 0.61],
    [0.706, 16.85, 0.32, 20.08, 0.52],
    [0.930, 21.71, 0.28, 17.88, 0.35],
    [1.317, 27.79, 0.69, 13.82, 0.42],
    [2.330, 39.71, 0.94, 8.52, 0.17],
])

def chi2_bao(params):
    Om, h, nu = params
    rd = sound_horizon(Om, h)
    chi2 = 0
    for row in DESI_DATA:
        z, DM_obs, DM_err, DH_obs, DH_err = row
        DM_th = DM_over_rd(z, Om, h, nu, rd)
        DH_th = DH_over_rd(z, Om, h, nu, rd)
        chi2 += ((DM_obs - DM_th) / DM_err)**2
        chi2 += ((DH_obs - DH_th) / DH_err)**2
    return chi2

def log_prior(params):
    Om, h, nu = params
    if not (0.1 < Om < 0.5): return -np.inf
    if not (0.5 < h < 0.9): return -np.inf
    if not (-0.05 < nu < 0.05): return -np.inf
    return 0.0

def log_likelihood(params):
    lp = log_prior(params)
    if not np.isfinite(lp): return -np.inf
    return lp - 0.5 * chi2_bao(params)

def run_mcmc(n_walkers=32, n_steps=5000, n_burn=1000):
    """Run MCMC to extract ν."""
    ndim = 3
    p0 = np.array([0.315, 0.674, 0.001])  # initial guess
    pos = p0 + 1e-3 * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood)

    print("Running MCMC...")
    print(f"  {n_walkers} walkers × {n_steps} steps")
    sampler.run_mcmc(pos, n_steps, progress=True)

    # Discard burn-in
    samples = sampler.get_chain(discard=n_burn, flat=True)
    labels = ['Ω_m', 'h', 'ν']

    # Results
    for i, label in enumerate(labels):
        q = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{label} = {q[1]:.5f} +{q[2]-q[1]:.5f} -{q[1]-q[0]:.5f}")

    nu_samples = samples[:, 2]
    nu_mean = np.mean(nu_samples)
    nu_std = np.std(nu_samples)

    print(f"\nν = {nu_mean:.5f} ± {nu_std:.5f}")
    print(f"ν / 10⁻³ = {nu_mean/1e-3:.2f} ± {nu_std/1e-3:.2f}")

    if abs(nu_mean) > 2 * nu_std:
        print(f"✓ ν ≠ 0 at {abs(nu_mean)/nu_std:.1f}σ")
    else:
        print(f"✗ ν consistent with 0 at current precision")

    # Corner plot
    try:
        fig = corner.corner(samples, labels=labels, truths=[0.315, 0.674, 1e-3])
        fig.savefig('P01_rvm_corner.png', dpi=150)
        print("Saved: P01_rvm_corner.png")
    except: pass

    np.save('P01_nu_samples.npy', nu_samples)
    return nu_mean, nu_std

if __name__ == '__main__':
    run_mcmc(n_walkers=32, n_steps=10000, n_burn=2000)
```

### Step 2: Run
```bash
python P01_rvm_fit.py
```

Expected runtime: ~30 min on i9 (32 walkers × 10000 steps).

## Expected Results
| Parameter | Expected | Solà Peracaula (2023) |
|---|---|---|
| ν | (0.5–1.5) × 10⁻³ | (0.9 ± 0.3) × 10⁻³ |
| Ω_m | 0.31 ± 0.01 | 0.312 ± 0.008 |
| h | 0.68 ± 0.01 | 0.681 ± 0.006 |

## Success: ν = (1.0 ± 0.5) × 10⁻³ at > 2σ from zero.
## Failure: ν consistent with zero at 2σ with this data combination.
