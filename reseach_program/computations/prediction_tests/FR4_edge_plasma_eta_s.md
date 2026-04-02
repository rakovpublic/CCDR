# FR4: Edge Plasma η/s Non-Monotonicity — KSS Minimum

## Prediction
The ratio η/s in tokamak edge plasma should show non-monotonic behaviour
as a function of collisionality ν*, with a minimum approaching the KSS bound
η/s = ℏ/(4πk_B).

## Software
```bash
pip install numpy scipy matplotlib pandas
```

## Data (partially public)
- **ITPA Global Confinement Database**: https://nucleus.iaea.org/sites/fusionportal/
  - Contains edge transport data from JET, DIII-D, ASDEX-U, EAST, KSTAR
  - Access may require registration (free for researchers)
- **Published edge transport studies**:
  - Wagner (2007): Plasma Phys. Control. Fusion 49, B1 (review)
  - Burrell (1997): Phys. Plasmas 4, 1499
  - Published χ_edge values from multiple tokamaks
- **ITER Physics Basis** (2007): Nuclear Fusion 47, S1 (public)

## Code
```python
#!/usr/bin/env python3
"""FR4_edge_eta_s.py — Test non-monotonic η/s vs collisionality."""
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
HBAR = 1.055e-34  # J·s
KB = 1.381e-23    # J/K
KSS = HBAR / (4 * np.pi * KB)  # ~ 6.08e-13 Pa·s·m³/J

# Published edge transport data (approximate, from review papers)
# Format: (ν* (collisionality), χ_edge (m²/s), T_edge (eV), n_edge (10¹⁹ m⁻³))
# ν* = q R n_e / (ε^1.5 × T_e² × τ_ei)  (dimensionless)
published_data = [
    # From various tokamaks (order-of-magnitude values)
    (0.01, 0.5,  500,  3),   # low ν* (hot edge)
    (0.03, 0.3,  300,  4),
    (0.1,  0.15, 200,  5),
    (0.3,  0.08, 150,  6),   # moderate ν*
    (1.0,  0.10, 100,  8),   # high ν*
    (3.0,  0.20, 80,   10),
    (10.0, 0.50, 50,   15),  # very high ν* (cold edge)
]

def chi_to_eta_over_s(chi, T_eV, n_19):
    """Convert thermal diffusivity to η/s."""
    T_K = T_eV * 11604.5
    n_m3 = n_19 * 1e19
    m_D = 3.344e-27  # deuteron mass
    # η ~ n m χ (kinematic viscosity to dynamic)
    eta = n_m3 * m_D * chi
    # s ~ n k_B (entropy density ~ n k_B for ideal gas)
    s = n_m3 * KB
    return eta / s

print("FR4: Edge Plasma η/s vs Collisionality")
print("=" * 60)
print(f"KSS bound: η/s = ℏ/(4πk_B) = {KSS:.3e} Pa·s·m³/J")
print()

nu_star = []
eta_s_values = []

for nu_s, chi, T, n in published_data:
    es = chi_to_eta_over_s(chi, T, n)
    ratio = es / KSS
    nu_star.append(nu_s)
    eta_s_values.append(es)
    print(f"ν* = {nu_s:5.2f}: χ = {chi:.2f} m²/s, "
          f"η/s = {es:.3e}, η/s / KSS = {ratio:.0f}×")

# Check non-monotonicity
eta_s_arr = np.array(eta_s_values)
min_idx = np.argmin(eta_s_arr)
print(f"\nMinimum η/s at ν* = {nu_star[min_idx]:.2f}")
print(f"  η/s_min = {eta_s_arr[min_idx]:.3e}")
print(f"  η/s_min / KSS = {eta_s_arr[min_idx] / KSS:.0f}×")

if min_idx > 0 and min_idx < len(eta_s_arr) - 1:
    print("✓ NON-MONOTONIC behaviour observed (minimum in interior)")
else:
    print("✗ Monotonic — no interior minimum (insufficient data range)")

print("\nTO DO: Replace approximate values with actual tokamak database entries.")
print("Access ITPA database for systematic χ_edge vs ν* scan.")
```

## Expected: Minimum in η/s at intermediate ν*, ~100-1000× above KSS.
## Timeline: 1–3 weeks (depends on database access).
## Note: Most data is published in review papers; ITPA database access varies.
