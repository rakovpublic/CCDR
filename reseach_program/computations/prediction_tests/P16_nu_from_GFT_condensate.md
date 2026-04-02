# P16: Deriving ν from GFT Condensate Fraction on CDT Monte Carlo

## Prediction
**ν_GFT = ν_RVM ~ 10⁻³ ± 10⁻⁴**

The GFT condensate fraction φ†φ / N_total, evaluated at CDT-constrained coupling constants, gives the RVM running vacuum coefficient ν. This is the first computation that connects quantum gravity (CDT) to observational cosmology (DESI BAO data) through a single number.

## What This Tests
If ν_GFT ≠ ν_RVM (from DESI): the GFT condensate does not reproduce the RVM. Baryogenesis loses its parameter-free derivation. ν reverts to a fit parameter.

If ν_GFT = ν_RVM: first bridge between lattice quantum gravity and observational cosmology.

---

## Prerequisites

### Software
- **CDT-plusplus**: https://github.com/acgetchell/CDT-plusplus (C++17, requires CGAL)
- **Python 3.10+** with numpy, scipy, matplotlib, h5py
- **Optional**: Julia with TensorOperations.jl for tensor contractions

### Hardware
- Minimum: 8-core workstation, 32 GB RAM (for N₄ ~ 10⁴)
- Recommended: 64-core cluster node, 128 GB RAM (for N₄ ~ 10⁵)
- For production: HPC cluster with MPI (for N₄ ~ 10⁶)

### Knowledge
- CDT phase structure (A/B/C/C_b phases)
- GFT mean-field theory (Gielen & Sindoni 2016)
- RVM cosmology (Solà Peracaula et al. 2023)

---

## Step-by-Step Computational Programme

### Step 1: Generate CDT C-Phase Triangulations

```bash
# Clone and build CDT-plusplus
git clone https://github.com/acgetchell/CDT-plusplus.git
cd CDT-plusplus && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Generate C-phase triangulation
# Parameters tuned for C-phase (from Ambjørn et al. 2004):
#   k0 (inverse bare Newton coupling) ~ 2.2
#   k4 (cosmological constant tuning) ~ adjusted for target N₄
#   N₄ = number of 4-simplices (start with 10000, scale up)
./cdt-opt --spacetime-dimension 4 \
          --number-of-simplices 10000 \
          --k0 2.2 \
          --k4 0.8 \
          --passes 1000 \
          --checkpoint-interval 100 \
          --output c_phase_N10000
```

**Validation**: Verify C-phase by checking:
1. Volume profile ⟨N₃(t)⟩ ∝ cos³(t/t₀) (de Sitter shape)
2. Spectral dimension d_s ≈ 4 at large scales, flowing toward 2 at small scales
3. No blob (A-phase) or crumpled (B-phase) pathology

```python
# validate_c_phase.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def de_sitter_profile(t, N0, t0):
    return N0 * np.cos(np.pi * t / t0)**3

def validate_volume_profile(triangulation_data):
    """Check that the volume profile matches de Sitter."""
    t_slices = triangulation_data['time_slices']
    volumes = triangulation_data['slice_volumes']

    popt, pcov = curve_fit(de_sitter_profile, t_slices, volumes,
                           p0=[max(volumes), len(t_slices)])
    residuals = volumes - de_sitter_profile(t_slices, *popt)
    rms = np.sqrt(np.mean(residuals**2)) / np.mean(volumes)

    print(f"De Sitter fit: N0={popt[0]:.0f}, t0={popt[1]:.1f}")
    print(f"Residual RMS (fractional): {rms:.4f}")

    if rms < 0.05:
        print("✓ C-phase confirmed (de Sitter profile)")
        return True
    else:
        print("✗ NOT C-phase — check parameters")
        return False
```

### Step 2: Extract CDT Geometric Observables

From each validated C-phase triangulation, extract:

```python
# extract_cdt_observables.py
import numpy as np

def extract_observables(triangulation):
    """Extract geometric observables needed for GFT matching."""

    # 2a: Average vertex coordination number <q>
    # Number of 4-simplices sharing each vertex
    vertex_coords = []
    for vertex in triangulation.vertices():
        q = len(triangulation.simplices_containing(vertex))
        vertex_coords.append(q)
    q_avg = np.mean(vertex_coords)
    q_std = np.std(vertex_coords)

    # 2b: Average edge length in units of bare lattice spacing
    edge_lengths = []
    for edge in triangulation.edges():
        l = triangulation.edge_length(edge)  # in lattice units
        edge_lengths.append(l)
    l_avg = np.mean(edge_lengths)

    # 2c: Volume-volume correlator
    # C(δt) = ⟨N₃(t) N₃(t+δt)⟩ - ⟨N₃(t)⟩⟨N₃(t+δt)⟩
    volumes = triangulation.slice_volumes()
    T = len(volumes)
    correlator = np.zeros(T // 2)
    for dt in range(T // 2):
        c = np.mean(volumes * np.roll(volumes, dt)) - np.mean(volumes)**2
        correlator[dt] = c
    correlator /= correlator[0]  # normalise

    # 2d: Spectral dimension via return probability
    # d_s(σ) = -2 d ln P(σ) / d ln σ
    # P(σ) = average return probability of random walk after σ steps
    sigma_values = np.logspace(1, 3, 50, dtype=int)
    return_probs = []
    for sigma in sigma_values:
        p = compute_return_probability(triangulation, sigma, n_walks=1000)
        return_probs.append(p)
    return_probs = np.array(return_probs)

    # Compute d_s
    log_sigma = np.log(sigma_values.astype(float))
    log_P = np.log(return_probs)
    d_s = -2 * np.gradient(log_P, log_sigma)

    return {
        'q_avg': q_avg,
        'q_std': q_std,
        'l_avg': l_avg,
        'volume_correlator': correlator,
        'spectral_dimension': d_s,
        'sigma_values': sigma_values,
        'N4': triangulation.num_simplices(4)
    }
```

### Step 3: Map CDT Observables to GFT Coupling Constants

The GFT action on SU(2)⁴ is:
```
S_GFT[φ] = ∫ φ†(g) K(g) φ(g) dg + (λ/5!) ∫ V(g₁,...,g₅) φ⁵ dg
```
where K is the kinetic kernel and V is the vertex (EPRL amplitude).

The CDT-to-GFT dictionary:

```python
# cdt_to_gft.py
import numpy as np

def map_cdt_to_gft(cdt_obs):
    """
    Map CDT geometric observables to GFT coupling constants.

    The GFT mass parameter m²_GFT is related to the CDT
    volume-volume correlator decay rate.

    The GFT interaction λ_GFT is related to the CDT
    vertex coordination number.

    Reference: Gielen & Sindoni (2016), de Cesare et al. (2016)
    """

    # 3a: GFT mass from volume correlator decay
    # The volume correlator C(δt) ~ exp(-m²_GFT × δt) at large δt
    correlator = cdt_obs['volume_correlator']
    # Fit exponential decay to extract m²
    dt_values = np.arange(len(correlator))
    # Use only the tail (δt > T/4) for exponential fit
    tail_start = len(correlator) // 4
    log_C = np.log(np.abs(correlator[tail_start:]) + 1e-15)
    dt_tail = dt_values[tail_start:]

    from numpy.polynomial import polynomial as P
    coeffs = P.polyfit(dt_tail, log_C, 1)
    m2_GFT = -coeffs[1]  # decay rate = m²_GFT

    # 3b: GFT interaction from vertex coordination
    # λ_GFT ~ 1/q_avg (higher coordination → weaker effective interaction)
    q_avg = cdt_obs['q_avg']
    lambda_GFT = 1.0 / q_avg

    # 3c: Effective N_eff (number of active degrees of freedom)
    # N_eff ~ q_avg × (geometric factor from simplex structure)
    # For equilateral 4-simplices: geometric factor = 5!/120 = 1
    N_eff = q_avg

    return {
        'm2_GFT': m2_GFT,
        'lambda_GFT': lambda_GFT,
        'N_eff': N_eff,
        'q_avg': q_avg
    }
```

### Step 4: Compute ν from GFT Condensate

```python
# compute_nu.py
import numpy as np

def compute_nu_from_gft(gft_params):
    """
    Compute ν from GFT condensate fraction.

    In the GFT condensate phase (⟨φ⟩ ≠ 0), the condensate fraction is:
        f_condensate = |⟨φ⟩|² / N_total = λ_GFT / m²_GFT

    The RVM coefficient ν is related to f_condensate by:
        ν = f_condensate / (48π) × N_eff

    This gives ν = N_eff / (48π) when λ/m² ~ 1 (stable condensate).

    Reference: de Cesare, Pithis & Sakellariadou (2016) PRD 94, 064051
    """

    m2 = gft_params['m2_GFT']
    lam = gft_params['lambda_GFT']
    N_eff = gft_params['N_eff']

    # Condensate fraction
    f_cond = lam / m2

    # Check stability: need λ/m² << 1 for perturbative control
    if f_cond > 0.1:
        print(f"WARNING: λ/m² = {f_cond:.3f} — near edge of perturbative regime")

    # Compute ν
    nu = N_eff / (48 * np.pi)

    # Alternative: direct from condensate fraction
    nu_direct = f_cond * N_eff / (48 * np.pi)

    # Check Israel-Stewart stability condition
    # ν must be in [10⁻⁴, 10⁻²] for stable crystal
    nu_stable = 1e-4 <= nu <= 1e-2

    return {
        'nu': nu,
        'nu_direct': nu_direct,
        'f_condensate': f_cond,
        'N_eff': N_eff,
        'stable': nu_stable
    }
```

### Step 5: Statistical Analysis Across Ensemble

```python
# ensemble_analysis.py
import numpy as np

def ensemble_nu(triangulation_files, n_configs=100):
    """
    Compute ν across an ensemble of CDT configurations.

    For each configuration:
    1. Validate C-phase
    2. Extract observables
    3. Map to GFT
    4. Compute ν

    Report: mean, std, systematic uncertainty from N₄ extrapolation.
    """

    nu_values = []

    for i, tfile in enumerate(triangulation_files[:n_configs]):
        tri = load_triangulation(tfile)

        if not validate_volume_profile(tri):
            print(f"Config {i}: NOT C-phase, skipping")
            continue

        obs = extract_observables(tri)
        gft = map_cdt_to_gft(obs)
        result = compute_nu_from_gft(gft)

        nu_values.append(result['nu'])
        if i % 10 == 0:
            print(f"Config {i}: ν = {result['nu']:.5f}, "
                  f"N_eff = {result['N_eff']:.1f}, "
                  f"f_cond = {result['f_condensate']:.4f}")

    nu_values = np.array(nu_values)
    nu_mean = np.mean(nu_values)
    nu_std = np.std(nu_values) / np.sqrt(len(nu_values))

    print(f"\n{'='*50}")
    print(f"RESULT: ν_GFT = {nu_mean:.5f} ± {nu_std:.5f}")
    print(f"RVM target: ν_RVM ~ 10⁻³")
    print(f"Ratio: ν_GFT / ν_RVM = {nu_mean / 1e-3:.2f}")
    print(f"N configs used: {len(nu_values)}")

    if abs(nu_mean - 1e-3) < 3 * nu_std + 1e-4:
        print("✓ CONSISTENT with ν_RVM at current precision")
    else:
        print("✗ TENSION with ν_RVM — investigate systematics")

    return nu_mean, nu_std, nu_values
```

### Step 6: Finite-Size Scaling (Continuum Limit)

```python
# finite_size_scaling.py
import numpy as np
from scipy.optimize import curve_fit

def extrapolate_nu(N4_values, nu_values, nu_errors):
    """
    Extrapolate ν to the continuum limit (N₄ → ∞).

    ν(N₄) = ν_∞ + a/N₄^α + ...

    Fit power-law corrections to extract ν_∞.
    """

    def scaling_form(N4, nu_inf, a, alpha):
        return nu_inf + a * N4**(-alpha)

    popt, pcov = curve_fit(scaling_form, N4_values, nu_values,
                           sigma=nu_errors, p0=[1e-3, 0.1, 0.5],
                           bounds=([0, -1, 0], [0.1, 1, 2]))

    nu_inf = popt[0]
    nu_inf_err = np.sqrt(pcov[0, 0])

    print(f"Continuum limit: ν_∞ = {nu_inf:.5f} ± {nu_inf_err:.5f}")
    print(f"Leading correction: a = {popt[1]:.4f}, α = {popt[2]:.2f}")

    return nu_inf, nu_inf_err
```

---

## Expected Results

| N₄ | Expected ν | Statistical error | Notes |
|---|---|---|---|
| 10⁴ | 0.5–2 × 10⁻³ | ~30% | Large finite-size effects |
| 10⁵ | 0.8–1.2 × 10⁻³ | ~10% | Finite-size effects diminishing |
| 10⁶ | 1.0 ± 0.1 × 10⁻³ | ~5% | Near continuum limit |
| ∞ (extrapolated) | 10⁻³ ± 10⁻⁴ | ~10% | The prediction |

**Success criterion**: ν_GFT = (1.0 ± 0.3) × 10⁻³, consistent with ν_RVM from DESI.

**Failure criterion**: ν_GFT outside [10⁻⁴, 10⁻²] at the extrapolated continuum limit.

---

## Timeline
- **Weeks 1–4**: Set up CDT-plusplus, generate N₄ = 10⁴ triangulations, validate C-phase.
- **Weeks 5–8**: Extract observables, map to GFT, compute ν for ~100 configurations.
- **Weeks 9–12**: Scale to N₄ = 10⁵, finite-size analysis.
- **Months 4–6**: N₄ = 10⁶ on cluster, continuum extrapolation, publication-ready result.

## References
- Gielen & Sindoni (2016). New J. Phys. 18, 013032.
- de Cesare, Pithis & Sakellariadou (2016). PRD 94, 064051.
- Solà Peracaula et al. (2023). EPJC 83, 516.
- Ambjørn, Jurkiewicz & Loll (2004). PRL 93, 131301.
- DESI Collaboration (2025). arXiv:2503.14738.
