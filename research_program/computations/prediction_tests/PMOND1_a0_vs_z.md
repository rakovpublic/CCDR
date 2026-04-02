# P-MOND-1: a₀(z) = cH(z) vs Constant a₀ from Galaxy Rotation Curves

## Prediction
CCDR: a₀(z) = cH(z) — acceleration scale varies with redshift.
Standard MOND: a₀ = 1.2 × 10⁻¹⁰ m/s² = constant.
At z=1: H(1) ≈ 1.5 H₀, so a₀(z=1) should be ~50% larger.

## Software
```bash
pip install numpy scipy matplotlib astropy pandas emcee
```

## Data (all public)
1. **SPARC** (z ≈ 0): http://astroweb.cwru.edu/SPARC/
   - 175 galaxies with measured rotation curves + photometry
   - Download: `SPARC_Lelli2016c.mrt` and individual rotation curves
2. **High-z rotation curves** (z ~ 1–2):
   - Genzel et al. (2017): 6 galaxies at z ~ 0.9–2.4 (Table 2, ApJ 840, 92)
   - Übler et al. (2017): 6 galaxies at z ~ 0.6–2.6 (ApJ 842, 121)
   - Rizzo et al. (2020): SPT0418-47 at z = 4.2 (Nature 584, 201)
   - JWST rotation curves from Tsukui et al. (2023), Nelson et al. (2023)

## Code

```python
#!/usr/bin/env python3
"""
PMOND1_a0_vs_z.py
Test whether a₀ varies with redshift as a₀(z) = cH(z) (CCDR)
or remains constant (standard MOND).
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

C_LIGHT = 2.998e8  # m/s
H0 = 67.4  # km/s/Mpc
H0_SI = H0 * 1e3 / 3.086e22  # convert to 1/s
A0_STANDARD = 1.2e-10  # m/s² (Milgrom's value)

def H_z(z, Om=0.315):
    """Hubble parameter at redshift z (ΛCDM)."""
    return H0_SI * np.sqrt(Om * (1+z)**3 + (1 - Om))

def a0_ccdr(z):
    """CCDR prediction: a₀(z) = c × H(z)."""
    return C_LIGHT * H_z(z)

def mond_rotation(r_kpc, M_bary, a0):
    """MOND predicted rotation velocity (simple interpolation).
    v⁴ = G M_bary a₀  (deep MOND regime, BTFR)
    Full: v² = v²_N × [1 + √(1 + (2r_a₀/(v²_N))²)] / 2
    """
    G = 6.674e-11
    M_sun = 1.989e30
    r_m = r_kpc * 3.086e19  # kpc to m
    M_kg = M_bary * M_sun

    v2_N = G * M_kg / r_m  # Newtonian
    x = v2_N / (a0 * r_m)  # a_N / a₀

    # MOND interpolation: μ(x) = x / √(1 + x²)
    v2_mond = v2_N / (0.5 * (1 + np.sqrt(1 + 4/x**2))) if x > 0 else v2_N
    return np.sqrt(abs(v2_mond)) / 1e3  # m/s to km/s

def fit_a0_to_galaxy(r_data, v_data, v_err, M_bary_solar):
    """Fit a₀ to a single galaxy's rotation curve."""
    def chi2(log_a0):
        a0 = 10**log_a0
        v_model = np.array([mond_rotation(r, M_bary_solar, a0)
                           for r in r_data])
        return np.sum(((v_data - v_model) / v_err)**2)

    result = minimize(chi2, x0=np.log10(A0_STANDARD),
                     bounds=[(-12, -8)], method='L-BFGS-B')
    a0_best = 10**result.x[0]

    # Error from Δχ² = 1
    chi2_min = result.fun
    def delta_chi2(log_a0):
        return abs(chi2(log_a0) - chi2_min - 1)
    from scipy.optimize import minimize_scalar
    res_hi = minimize_scalar(delta_chi2,
                            bounds=(result.x[0], result.x[0]+1),
                            method='bounded')
    a0_err = 10**res_hi.x - a0_best

    return a0_best, a0_err

def analyse_sparc():
    """Fit a₀ for each SPARC galaxy (z ≈ 0)."""
    print("SPARC analysis (z ≈ 0)")
    print("Download data from http://astroweb.cwru.edu/SPARC/")
    print("Place rotation curve files in ./SPARC_rotcurves/")
    print()

    # Example with a well-measured galaxy (NGC 2403)
    # Replace with actual SPARC data
    r = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20])  # kpc
    v = np.array([40, 70, 100, 115, 125, 130, 132, 133, 134])  # km/s
    v_err = np.array([5, 5, 5, 5, 5, 5, 5, 7, 10])
    M_bary = 3e10  # solar masses

    a0, a0_err = fit_a0_to_galaxy(r, v, v_err, M_bary)
    print(f"NGC 2403 (example): a₀ = {a0:.2e} ± {a0_err:.2e} m/s²")
    print(f"Standard a₀ = {A0_STANDARD:.2e}")
    print(f"CCDR a₀(z=0) = c×H₀ = {C_LIGHT * H0_SI:.2e}")
    return a0, a0_err

def analyse_high_z():
    """Compare a₀ at different redshifts."""
    print("\nHIGH-z ANALYSIS")
    print("=" * 60)

    # Published best-fit a₀ values from high-z studies
    # (extract from papers or fit yourself)
    measurements = [
        # (z, a0_measured, a0_error, source)
        (0.0, 1.2e-10, 0.2e-10, "SPARC mean"),
        (0.9, None, None, "Genzel+2017 — fit from their data"),
        (2.0, None, None, "Übler+2017 — fit from their data"),
        (4.2, None, None, "Rizzo+2020 — fit from their data"),
    ]

    print("Predictions:")
    for z in [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.2]:
        a0_pred = a0_ccdr(z)
        ratio = a0_pred / A0_STANDARD
        print(f"  z = {z:.1f}: a₀(CCDR) = {a0_pred:.2e} m/s² "
              f"({ratio:.2f} × a₀_standard)")

    print()
    print("KEY TEST: At z = 1, CCDR predicts a₀ ~50% larger than z = 0.")
    print("At z = 4.2, CCDR predicts a₀ ~5× larger.")
    print("Standard MOND predicts the SAME a₀ at all z.")
    print()
    print("TO DO: Download Genzel+2017, Übler+2017, Rizzo+2020 data.")
    print("Fit MOND to each galaxy. Extract a₀. Plot a₀ vs z.")
    print("Compare with a₀(z) = cH(z) and a₀ = constant.")

if __name__ == '__main__':
    analyse_sparc()
    analyse_high_z()
```

## Expected Results
| z | a₀ (CCDR) | a₀ (standard MOND) | Ratio |
|---|---|---|---|
| 0.0 | 6.8 × 10⁻¹⁰ | 1.2 × 10⁻¹⁰ | 1.00 |
| 1.0 | 1.0 × 10⁻⁹ | 1.2 × 10⁻¹⁰ | 1.50 |
| 2.0 | 1.5 × 10⁻⁹ | 1.2 × 10⁻¹⁰ | 2.15 |
| 4.2 | 3.1 × 10⁻⁹ | 1.2 × 10⁻¹⁰ | 4.57 |

Note: CCDR's a₀ = cH₀ ≈ 6.8×10⁻¹⁰ at z=0 differs from Milgrom's 1.2×10⁻¹⁰ by a factor of ~5.7. The prediction is the SCALING with z, not the absolute value — or the absolute value needs a correction factor from the crystal grain structure.

## Timeline: 1–2 weeks (data download + fitting)
## Smoking gun: a₀ increasing with z, measurably different from constant
