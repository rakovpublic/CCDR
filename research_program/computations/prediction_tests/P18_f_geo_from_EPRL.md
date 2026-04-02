# P18: Computing f_geo from the EPRL Chiral Amplitude

## Prediction
**f_geo = |sin(γ_AS × S_Regge / ℏ)| ~ 10⁻⁴**

The baryogenesis geometric factor f_geo measures the fractional chiral imbalance per crystallisation event, computed from the oriented/anti-oriented ratio of the EPRL spin foam vertex amplitude evaluated at the CDT C-phase saddle point.

## What This Tests
Combined with P16 (ν ~ 10⁻³) and δ_CP ~ 10⁻³ from CDT chirality:

**η = ν × δ_CP × f_geo ~ 10⁻³ × 10⁻³ × 10⁻⁴ = 10⁻¹⁰**

If f_geo ~ 10⁻⁴: baryogenesis is parameter-free. η matches observation.

If f_geo is significantly different: either the baryogenesis formula is wrong, or γ_AS is not the correct value at the QCD scale.

---

## Prerequisites

### Software
- **sl2cfoam-next**: https://github.com/qg-cpt-marseille/sl2cfoam-next
  - Computes Lorentzian EPRL spin foam amplitudes numerically
  - Requires: C compiler, GSL, OpenMP (optional for parallelism)
- **Python 3.10+** with numpy, scipy, mpmath
- **Mathematica 13+** (optional, for symbolic checks)

### Key Papers
- Engle, Pereira, Rovelli & Livine (2008). NPB 799, 136. [EPRL definition]
- Dona, Gozzini & Sarno (2019). PRD 100, 106003. [sl2cfoam-next]
- Barrett et al. (2009). CQG 27, 165009. [Large-spin asymptotics]

---

## Background: The EPRL Vertex Amplitude

The EPRL vertex amplitude for a single 4-simplex with ten boundary spins {j_f} and five intertwiners {i_e} is:

```
A_EPRL(v) = ∫_{SL(2,C)^5} ∏_f K^γ_{j_f}(g_{s(f)} g⁻¹_{t(f)}) ∏_e dg_e
```

where K^γ_j is the EPRL propagator kernel implementing the simplicity constraint with Barbero–Immirzi parameter γ:

```
K^γ_j(g) = ∑_p d_p × D^{(p,(1+γ)j/2,|1-γ|j/2)}(g)
```

In the large-spin (stationary phase) limit, the amplitude becomes:

```
A_EPRL(v) → N_v × [exp(i γ S_Regge) + exp(-i γ S_Regge)] + corrections
```

The two terms correspond to **oriented** and **anti-oriented** 4-simplices.

---

## Step-by-Step Computational Programme

### Step 1: Set Up sl2cfoam-next

```bash
# Clone and build sl2cfoam-next
git clone https://github.com/qg-cpt-marseille/sl2cfoam-next.git
cd sl2cfoam-next
make

# Verify installation with the test suite
make test

# The main executable computes vertex amplitudes:
# ./sl2cfoam <gamma> <2j1> <2j2> ... <2j10> <dl_min> <dl_max>
# where 2j are twice the boundary spins (integers)
# and dl_min, dl_max are the SL(2,C) representation truncation
```

### Step 2: Compute the EPRL Amplitude for the Equilateral 4-Simplex

The CDT C-phase saddle point is an equilateral 4-simplex: all 10 boundary spins equal, all 5 intertwiners equal. The simplest non-trivial case is j = 1/2 (2j = 1 for all faces).

```bash
# Compute amplitude for equilateral 4-simplex at j=1/2
# gamma = 0.274 (AS fixed point value)
# All 10 spins = 1/2 (2j = 1)
# SL(2,C) truncation: dl_min = 0, dl_max = 30

./sl2cfoam 0.274  1 1 1 1 1 1 1 1 1 1  0 30 > amplitude_j05_g0274.dat

# Repeat for several gamma values to study dependence
for gamma in 0.20 0.22 0.2375 0.25 0.274 0.30; do
    ./sl2cfoam $gamma  1 1 1 1 1 1 1 1 1 1  0 30 > amplitude_j05_g${gamma}.dat
done

# Higher spins for convergence check
for j2 in 1 2 3 4 5 6; do
    ./sl2cfoam 0.274  $j2 $j2 $j2 $j2 $j2 $j2 $j2 $j2 $j2 $j2  0 40 > amplitude_j${j2}_g0274.dat
done
```

### Step 3: Extract the Oriented/Anti-Oriented Ratio

```python
# extract_chiral_ratio.py
import numpy as np
import os

def parse_sl2cfoam_output(filename):
    """
    Parse sl2cfoam-next output to extract the complex amplitude.

    The output format is typically:
    real_part  imaginary_part  [convergence info]
    """
    data = np.loadtxt(filename, comments='#')
    if data.ndim == 1:
        return complex(data[0], data[1])
    else:
        # Multiple dl values — take the last (most converged)
        return complex(data[-1, 0], data[-1, 1])


def compute_chiral_ratio(gamma, amplitude):
    """
    Extract the oriented/anti-oriented ratio from the EPRL amplitude.

    In the large-spin limit:
        A_EPRL = N × [exp(iγS) + exp(-iγS)] = 2N cos(γS)  (real part)
                                                + i × 0      (imaginary from cosine)

    But the full amplitude has corrections:
        A_EPRL = A_oriented + A_anti-oriented

    where:
        A_oriented     ~ exp(+i γ S_Regge) × (1 + corrections)
        A_anti-oriented ~ exp(-i γ S_Regge) × (1 + corrections')

    The chiral asymmetry is:
        R = |A_oriented|² - |A_anti-oriented|²
          / |A_oriented|² + |A_anti-oriented|²

    For the exact amplitude (not just large-spin limit):
        R = 2 Im(A) × Re(A) / |A|²  (at leading order in the chiral asymmetry)

    This is nonzero when the amplitude has both real and imaginary parts
    that are not in the trivial ratio Im/Re = tan(γS).
    """

    S_Regge = 5 * np.arccos(0.25)  # equilateral 4-simplex

    Re_A = amplitude.real
    Im_A = amplitude.imag

    # The chiral ratio from the full amplitude
    if abs(amplitude) < 1e-30:
        return 0.0

    # Method 1: Direct from Re/Im structure
    R_direct = 2 * Re_A * Im_A / (Re_A**2 + Im_A**2)

    # Method 2: Compare with the symmetric (achiral) prediction
    # The achiral amplitude would have Im(A) / Re(A) = tan(γ × S_Regge)
    tan_gS = np.tan(gamma * S_Regge)
    expected_ratio = Im_A / Re_A if abs(Re_A) > 1e-30 else float('inf')
    chiral_deviation = (expected_ratio - tan_gS) / (1 + abs(tan_gS))

    # Method 3: The f_geo formula from the synthesis
    # f_geo = |sin(γ × S_Regge)|  at the stationary phase
    f_geo_analytic = abs(np.sin(gamma * S_Regge))

    return {
        'R_direct': R_direct,
        'chiral_deviation': chiral_deviation,
        'f_geo_analytic': f_geo_analytic,
        'Re_A': Re_A,
        'Im_A': Im_A,
        'phase': np.angle(amplitude),
        'gamma': gamma,
        'S_Regge': S_Regge
    }


def scan_gamma_values():
    """
    Compute f_geo as a function of γ.
    """
    gamma_values = [0.20, 0.22, 0.2375, 0.25, 0.274, 0.30]
    results = []

    for gamma in gamma_values:
        filename = f"amplitude_j05_g{gamma}.dat"
        if os.path.exists(filename):
            amplitude = parse_sl2cfoam_output(filename)
            result = compute_chiral_ratio(gamma, amplitude)
            results.append(result)

            print(f"γ = {gamma:.4f}: f_geo(analytic) = {result['f_geo_analytic']:.6f}, "
                  f"R_direct = {result['R_direct']:.6f}, "
                  f"phase = {result['phase']:.4f}")
        else:
            print(f"γ = {gamma:.4f}: file not found — run sl2cfoam first")

    return results

results = scan_gamma_values()
```

### Step 4: Compute f_geo at the Synthesis Prediction Point

```python
# compute_f_geo.py
import numpy as np

def compute_f_geo_prediction(gamma_AS=0.274):
    """
    Compute the predicted f_geo from the synthesis.

    f_geo = |sin(γ_AS × S_Regge / ℏ)|

    where S_Regge is the Regge action of the equilateral CDT C-phase
    4-simplex, evaluated in Planck units (ℏ = 1).

    For the equilateral 4-simplex with edge length l_CDT:
        S_Regge = (l²_CDT / l²_Pl) × Σ_hinges [A_hinge × δ_hinge]

    For l_CDT = l_Pl (Planck-scale simplex):
        S_Regge = Σ_hinges [A_hinge × δ_hinge]

    A 4-simplex has 10 hinges (triangular faces).
    For equilateral: A_hinge = √3/4 × l² and δ_hinge = arccos(1/4).
    Total: S_Regge = 10 × (√3/4) × arccos(1/4) ≈ 5.707

    But the standard formula uses 5 tetrahedra, each with
    deficit angle at the opposite edge:
        S_Regge = 5 × A_triangle × (2π - n_sharing × θ_dihedral)

    For a SINGLE equilateral 4-simplex:
        S_Regge = 10 × A_triangle × (π - θ_dihedral)
        θ_dihedral = arccos(1/4) ≈ 1.3181 rad

    With A_triangle = √3/4 (in l_Pl units):
        S_Regge = 10 × (√3/4) × (π - arccos(1/4))
                = 10 × 0.4330 × 1.8235
                ≈ 7.896
    """

    # Dihedral angle of equilateral 4-simplex
    theta_dihedral = np.arccos(1/4)

    # Triangle area (equilateral, l = 1 in Planck units)
    A_triangle = np.sqrt(3) / 4

    # Regge action: sum over all 10 hinges (triangular faces)
    # Deficit angle for a single simplex: δ = π - θ_dihedral
    # (for a single simplex in flat space, the deficit is 2π minus
    #  the dihedral angle times the number of simplices sharing the hinge)
    # For an isolated equilateral 4-simplex:
    deficit = np.pi - theta_dihedral
    S_Regge = 10 * A_triangle * deficit

    # Alternative: the compact formula often used
    S_Regge_compact = 5 * np.arccos(0.25)

    print(f"θ_dihedral = {theta_dihedral:.4f} rad = {np.degrees(theta_dihedral):.1f}°")
    print(f"A_triangle = {A_triangle:.4f} (Planck units)")
    print(f"S_Regge (10 hinges) = {S_Regge:.4f}")
    print(f"S_Regge (compact)   = {S_Regge_compact:.4f}")

    # Use the compact formula (standard in the literature)
    S = S_Regge_compact

    # f_geo = |sin(γ × S)|
    f_geo = abs(np.sin(gamma_AS * S))

    print(f"\nγ_AS = {gamma_AS:.4f}")
    print(f"γ × S = {gamma_AS * S:.4f}")
    print(f"sin(γ × S) = {np.sin(gamma_AS * S):.6f}")
    print(f"|sin(γ × S)| = f_geo = {f_geo:.6f}")

    # Check: is f_geo ~ 10⁻⁴?
    log10_f = np.log10(f_geo) if f_geo > 0 else float('-inf')
    print(f"log₁₀(f_geo) = {log10_f:.2f}")
    print(f"Target: log₁₀(10⁻⁴) = -4.00")

    # The baryogenesis prediction
    nu = 1e-3
    delta_CP = 1e-3
    eta = nu * delta_CP * f_geo
    print(f"\nη = ν × δ_CP × f_geo = {nu:.0e} × {delta_CP:.0e} × {f_geo:.2e} = {eta:.2e}")
    print(f"Observed: η = 6.1 × 10⁻¹⁰")

    return f_geo, eta


def sensitivity_analysis():
    """
    How sensitive is f_geo to γ and S_Regge?
    """
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS: f_geo vs γ")
    print("="*60)

    S = 5 * np.arccos(0.25)

    gamma_values = np.linspace(0.20, 0.35, 31)
    for g in gamma_values:
        f = abs(np.sin(g * S))
        eta = 1e-3 * 1e-3 * f
        marker = " ◄◄◄" if abs(np.log10(eta + 1e-30) - (-10)) < 0.5 else ""
        print(f"γ = {g:.3f}: f_geo = {f:.6f}, η = {eta:.2e}{marker}")

    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS: f_geo vs S_Regge")
    print("="*60)

    gamma = 0.274
    # S_Regge depends on the simplex geometry (not necessarily equilateral)
    # CDT C-phase simplices are approximately but not exactly equilateral
    S_values = np.linspace(4, 10, 25)
    for S in S_values:
        f = abs(np.sin(gamma * S))
        eta = 1e-3 * 1e-3 * f
        marker = " ◄◄◄" if abs(np.log10(eta + 1e-30) - (-10)) < 0.5 else ""
        print(f"S = {S:.2f}: f_geo = {f:.6f}, η = {eta:.2e}{marker}")


# Main computation
f_geo, eta = compute_f_geo_prediction()
sensitivity_analysis()
```

### Step 5: Higher-Spin Corrections

```python
# higher_spin_corrections.py
import numpy as np

def f_geo_with_higher_spins(gamma=0.274, j_max=10):
    """
    Compute f_geo including contributions from higher spins.

    The dominant contribution is j = 1/2, but higher spins
    contribute corrections suppressed by (l_Pl / l_CDT)^2 per spin increment.

    f_geo(total) = Σ_j w(j) × |sin(γ × S_Regge(j))|

    where w(j) ~ (2j+1) × exp(-j(j+1) × l²_Pl / l²_CDT)
    is the Boltzmann weight of the spin-j representation,
    and S_Regge(j) = j × S_Regge(1/2) scales linearly with spin.
    """

    S_half = 5 * np.arccos(0.25)  # Regge action for j=1/2

    # l_CDT / l_Pl ratio (from CDT Monte Carlo — typically O(1) to O(10))
    l_ratio = 1.0  # Planck-scale simplex (conservative)

    total_weight = 0
    total_f_geo = 0

    print(f"{'j':>5} {'weight':>12} {'S_Regge':>10} {'f_geo(j)':>12} {'contribution':>14}")
    print("-" * 60)

    for j2 in range(1, 2*j_max + 1):
        j = j2 / 2.0
        d_j = 2*j + 1

        # Weight: Boltzmann factor
        w = d_j * np.exp(-j * (j + 1) / l_ratio**2)

        # Regge action scales with j
        S_j = j * 2 * S_half  # S(j) = 2j × S(1/2)

        # f_geo for this spin
        f_j = abs(np.sin(gamma * S_j))

        # Contribution
        contrib = w * f_j
        total_weight += w
        total_f_geo += contrib

        if j <= 3 or j2 % 4 == 0:
            print(f"{j:5.1f} {w:12.6f} {S_j:10.4f} {f_j:12.6f} {contrib:14.8f}")

    f_geo_avg = total_f_geo / total_weight

    print(f"\nTotal weight: {total_weight:.4f}")
    print(f"f_geo (weighted average): {f_geo_avg:.6f}")
    print(f"f_geo (j=1/2 only): {abs(np.sin(gamma * S_half)):.6f}")
    print(f"Correction from higher spins: {(f_geo_avg / abs(np.sin(gamma * S_half)) - 1) * 100:.1f}%")

    return f_geo_avg

f_geo_corrected = f_geo_with_higher_spins()
```

### Step 6: Full Numerical Verification with sl2cfoam-next

```python
# full_numerical_verification.py
import numpy as np
import subprocess
import os

def run_sl2cfoam(gamma, j2_all, dl_max=30):
    """
    Run sl2cfoam-next and parse the result.
    """
    spins_str = " ".join([str(j2_all)] * 10)
    cmd = f"./sl2cfoam {gamma} {spins_str} 0 {dl_max}"

    result = subprocess.run(cmd.split(), capture_output=True, text=True,
                           cwd="sl2cfoam-next/")

    # Parse output (format depends on sl2cfoam version)
    lines = result.stdout.strip().split('\n')
    for line in reversed(lines):
        parts = line.split()
        if len(parts) >= 2:
            try:
                re_a = float(parts[0])
                im_a = float(parts[1])
                return complex(re_a, im_a)
            except ValueError:
                continue
    return None


def full_verification(gamma_target=0.274):
    """
    Full numerical verification of f_geo using sl2cfoam-next.
    """
    print("FULL NUMERICAL VERIFICATION OF f_geo")
    print("=" * 60)

    # Compute at the target γ for multiple j values
    j2_values = [1, 2, 3, 4, 5, 6]

    for j2 in j2_values:
        j = j2 / 2.0
        amplitude = run_sl2cfoam(gamma_target, j2)

        if amplitude is not None:
            phase = np.angle(amplitude)
            S_expected = j * 2 * 5 * np.arccos(0.25)

            print(f"\nj = {j}: A = {amplitude.real:.6e} + {amplitude.imag:.6e}i")
            print(f"  |A| = {abs(amplitude):.6e}")
            print(f"  phase = {phase:.4f} rad")
            print(f"  Expected γS = {gamma_target * S_expected:.4f}")
            print(f"  sin(γS) = {np.sin(gamma_target * S_expected):.6f}")

            # f_geo from numerical amplitude
            if abs(amplitude) > 0:
                f_geo_num = 2 * amplitude.real * amplitude.imag / abs(amplitude)**2
                print(f"  f_geo (numerical) = {abs(f_geo_num):.6f}")
        else:
            print(f"\nj = {j}: sl2cfoam computation failed")

    # The critical result
    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)

    S_half = 5 * np.arccos(0.25)
    f_geo_analytic = abs(np.sin(gamma_target * S_half))
    eta_predicted = 1e-3 * 1e-3 * f_geo_analytic

    print(f"f_geo (analytic, j=1/2) = {f_geo_analytic:.6f}")
    print(f"η = ν × δ_CP × f_geo = {eta_predicted:.2e}")
    print(f"η (observed) = 6.1 × 10⁻¹⁰")

    ratio = eta_predicted / 6.1e-10
    print(f"Predicted / Observed = {ratio:.2f}")

    if 0.1 < ratio < 10:
        print("✓ CONSISTENT within order of magnitude")
    else:
        print("✗ INCONSISTENT — check γ, S_Regge, or baryogenesis formula")


# Run if sl2cfoam is available
if os.path.exists("sl2cfoam-next/sl2cfoam"):
    full_verification()
else:
    print("sl2cfoam-next not found — running analytic computation only")
    f_geo, eta = compute_f_geo_prediction()
```

---

## Expected Results

| Quantity | Expected | Notes |
|---|---|---|
| S_Regge (equilateral) | 6.59 | 5 × arccos(1/4) |
| γ_AS | 0.274 | From Reuter's AS fixed point |
| γ × S | 1.806 | Product |
| sin(γ × S) | 0.971 | Large — NOT 10⁻⁴ |

**Critical observation**: With the compact formula S_Regge = 5 arccos(1/4) ≈ 6.59, the product γ × S ≈ 1.81, and |sin(γS)| ≈ 0.97 — NOT 10⁻⁴.

This means f_geo ~ 10⁻⁴ requires **either**:
1. S_Regge at the CDT C-phase saddle point is significantly different from the single equilateral simplex value (because the CDT ensemble averages over many simplices with varying geometries), **or**
2. The relevant S_Regge includes the **sum over multiple simplices** in a coarse-grained block, giving a much larger total action, **or**
3. The f_geo formula involves sin(γS) evaluated at a saddle point where γS is near nπ (a zero of sin), requiring fine-tuning of γ and S.

**This is the most important diagnostic result of P18**: the naive analytic formula gives f_geo ~ O(1), not 10⁻⁴. If the numerical sl2cfoam computation confirms f_geo ~ O(1), the baryogenesis formula η = ν × δ_CP × f_geo gives η ~ 10⁻⁶ (too large by 10⁴), and the geometric factor must come from a different mechanism (e.g., the sum over an ensemble of non-equilateral simplices, or the coarse-grained block action).

**This is why P18 is the most urgent computation**: it tests whether the baryogenesis formula works as written, or whether it needs modification.

---

## Decision Tree Based on P18 Result

```
f_geo ~ 10⁻⁴ (from numerical sl2cfoam)
  → η ~ 10⁻¹⁰ ✓ → baryogenesis confirmed parameter-free
  → Implies: CDT ensemble averaging or block action effects reduce f_geo

f_geo ~ O(1) (from numerical sl2cfoam)
  → η ~ 10⁻⁶ ✗ → too large by 10⁴
  → Option A: f_geo comes from CDT ensemble average ⟨sin(γS)⟩ ~ 10⁻⁴
              (destructive interference over many simplex orientations)
  → Option B: f_geo comes from the coarse-grained block action
              S_block >> S_single, pushing γS near nπ
  → Option C: the baryogenesis formula needs a fourth factor
              η = ν × δ_CP × f_geo × f_ensemble

f_geo = 0 (exact cancellation)
  → No chiral asymmetry from EPRL → baryogenesis mechanism fails
  → Must seek alternative CP source within the synthesis
```

---

## Timeline
- **Week 1**: Install sl2cfoam-next. Run test cases from Dona & Sarno (2019).
- **Weeks 2–3**: Compute EPRL amplitude for equilateral 4-simplex at γ = 0.274, j = 1/2 through j = 5.
- **Weeks 4–5**: Extract chiral ratio. Compare analytic vs numerical f_geo.
- **Weeks 6–8**: If f_geo ~ O(1): investigate CDT ensemble averaging (requires coupling to CDT-plusplus output).
- **Months 3–4**: Compute ⟨f_geo⟩ over CDT C-phase ensemble. This is the definitive test.
- **Months 5–6**: Publication-ready result with uncertainty analysis.

## References
- Engle, Pereira, Rovelli & Livine (2008). NPB 799, 136.
- Dona, Gozzini & Sarno (2019). PRD 100, 106003.
- Barrett, Dowdall, Fairbairn, Gomes & Hellmann (2009). CQG 27, 165009.
- Bahr & Steinhaus (2016). PRL 117, 141302.
- Sakharov (1967). JETP Lett. 5, 24.
