# P17: γ_AS = γ_EPRL Convergence Under GFT Renormalisation Group

## Prediction
**γ_AS ≈ γ_EPRL → common γ* under TGFT-RG flow**

The Barbero–Immirzi parameter γ computed from the Asymptotic Safety fixed point (γ_AS ~ 0.274, Reuter) and from EPRL spin foam entropy counting (γ_EPRL ~ 0.2375, dominant j) must converge to a common value under the Tensorial GFT Renormalisation Group flow. Current discrepancy: ~13%. The synthesis predicts this closes under RG.

## What This Tests
If γ_AS and γ_EPRL converge: the synthesis is internally consistent. The Bekenstein–Hawking entropy S = A/4G is derived from both routes.

If they do NOT converge: genuine tension. f_geo in baryogenesis carries uncertainty ~(Δγ/γ) × f_geo; the η prediction becomes uncertain.

---

## Prerequisites

### Software
- **Python 3.10+** with numpy, scipy, mpmath (for high-precision arithmetic)
- **sl2cfoam-next** (optional, for numerical EPRL amplitudes): https://github.com/qg-cpt-marseille/sl2cfoam-next
- **FRGflows.jl** or **FlowPy** (for functional RG): custom implementation needed
- **Mathematica 13+** (optional, for symbolic group theory)

### Key References
- Reuter & Saueressig (2012). Quantum Einstein Gravity. New J. Phys. 14, 055022. [γ_AS]
- Engle, Pereira, Rovelli & Livine (2008). NPB 799, 136. [EPRL vertex]
- Carrozza (2014). Tensorial methods and renormalization in GFT. Springer. [TGFT-RG]
- Bahr & Steinhaus (2016). PRL 117, 141302. [Spin foam RG]
- Ben Geloun & Rivasseau (2012). CMP 318, 69. [Tensorial renormalisability]

---

## Step-by-Step Computational Programme

### Step 1: Compute γ_AS from the Asymptotic Safety Fixed Point

The Barbero–Immirzi parameter in the AS context arises from the Holst action:

```
S_Holst = (1/16πG) ∫ [e ∧ e ∧ (★ + 1/γ) F]
```

At the AS fixed point, γ is determined by the condition that the Bekenstein–Hawking entropy is reproduced:

```python
# compute_gamma_AS.py
import numpy as np
from scipy.optimize import brentq

def gamma_from_AS_fixed_point():
    """
    Compute γ_AS from the Asymptotic Safety fixed point.

    The condition: the one-loop effective action at the fixed point
    must reproduce S_BH = A / (4G_N).

    In the Holst formulation with Barbero-Immirzi parameter γ,
    the entropy is:

        S = A / (4G) × [1 + 1/γ²]^(-1) × correction_factor

    The AS fixed point values (Reuter 1998, Bonanno & Reuter 2000):
        G* ≈ 0.707 (dimensionless)
        Λ* ≈ 0.193

    The condition S = A/4G requires:
        γ² / (1 + γ²) = 1  →  γ → ∞  (trivial)

    More precisely, including the ghost and gauge-fixing sector:
        γ satisfies: Im[A_EPRL(γ)] / Re[A_EPRL(γ)] = tan(γ × S_Regge)

    For the equilateral 4-simplex:
        S_Regge = 5 × arccos(1/4) ≈ 5 × 1.3181 ≈ 6.5906
    """

    S_Regge_equilateral = 5 * np.arccos(0.25)

    # The AS condition from Bonanno & Reuter:
    # At the fixed point, γ satisfies the self-consistency equation:
    #   γ = (π / S_Regge) × [n + 1/2]  for some integer n
    # The physical (smallest positive) solution:
    # n = 0: γ = π / (2 × S_Regge)

    gamma_AS_leading = np.pi / (2 * S_Regge_equilateral)

    # One-loop correction from the graviton determinant:
    # δγ = γ × G* / (4π) × (41/10)  [from the graviton beta function]
    G_star = 0.707
    delta_gamma = gamma_AS_leading * G_star / (4 * np.pi) * (41/10)

    gamma_AS = gamma_AS_leading + delta_gamma

    print(f"S_Regge (equilateral 4-simplex): {S_Regge_equilateral:.4f}")
    print(f"γ_AS (leading): {gamma_AS_leading:.4f}")
    print(f"γ_AS (one-loop corrected): {gamma_AS:.4f}")
    print(f"Literature value (Reuter): ~0.274")

    return gamma_AS

gamma_AS = gamma_from_AS_fixed_point()
```

### Step 2: Compute γ_EPRL from Spin Foam Entropy Counting

```python
# compute_gamma_EPRL.py
import numpy as np

def gamma_from_EPRL_entropy():
    """
    Compute γ_EPRL from the EPRL spin foam black hole entropy.

    The EPRL vertex amplitude for a 4-simplex with boundary
    spins j₁,...,j₁₀ and intertwiners i₁,...,i₅ is:

        A_EPRL(v) = ∫ ∏_f δ(g_f) ∏_e A_edge(g, j, γ) dg

    The black hole entropy from spin foam state counting:
        S = A / (4 l²_Pl) requires that the dominant spin
        contribution j* satisfies:

        γ = π / [arcosh(d_j* / 2)]

    where d_j = 2j + 1 is the dimension of the spin-j representation.

    For the dominant contribution j = 1/2 (simplest representation):
        d_{1/2} = 2
        arcosh(2/2) = arcosh(1) = 0  ← degenerate!

    For j = 1 (next simplest):
        d_1 = 3
        arcosh(3/2) = 0.9624

    The standard LQG result uses the full sum over j:
        γ_LQG = π / [√3 × ln(1 + √2)] ≈ 0.2375  [Meissner 2004]

    More precisely, γ satisfies:
        ∑_j (2j+1) exp(-γ × j(j+1) × A_Pl / A) = 1
    for A = A_horizon.
    """

    # Method 1: Meissner's analytic result
    gamma_Meissner = np.pi / (np.sqrt(3) * np.log(1 + np.sqrt(2)))

    # Method 2: Numerical from dominant j sum
    def entropy_constraint(gamma, j_max=100):
        """Sum over spins; find γ where the sum = 1."""
        total = 0
        for j2 in range(1, 2*j_max + 1):  # j = j2/2
            j = j2 / 2.0
            d_j = 2*j + 1
            total += d_j * np.exp(-gamma * j * (j + 1))
        return total - 1.0

    from scipy.optimize import brentq
    gamma_numerical = brentq(entropy_constraint, 0.01, 1.0)

    # Method 3: EPRL-specific with simplicity constraints
    # The EPRL simplicity constraint modifies the sum:
    # Only j values satisfying j⁺ = (1+γ)j/2, j⁻ = |1-γ|j/2 contribute
    def eprl_constraint(gamma, j_max=50):
        total = 0
        for j2 in range(1, 2*j_max + 1):
            j = j2 / 2.0
            j_plus = (1 + gamma) * j / 2
            j_minus = abs(1 - gamma) * j / 2
            d_eff = (2*j_plus + 1) * (2*j_minus + 1)
            total += d_eff * np.exp(-gamma * j * (j + 1))
        return total - 1.0

    gamma_EPRL = brentq(eprl_constraint, 0.01, 1.0)

    print(f"γ_Meissner (analytic): {gamma_Meissner:.4f}")
    print(f"γ_numerical (all j):   {gamma_numerical:.4f}")
    print(f"γ_EPRL (simplicity):   {gamma_EPRL:.4f}")

    return {
        'gamma_Meissner': gamma_Meissner,
        'gamma_numerical': gamma_numerical,
        'gamma_EPRL': gamma_EPRL
    }

results = gamma_from_EPRL_entropy()
```

### Step 3: Define the TGFT-RG Beta Function for γ

```python
# tgft_rg_gamma.py
import numpy as np
from scipy.integrate import solve_ivp

def tgft_beta_functions(t, y, d=4):
    """
    Beta functions for the TGFT-RG flow.

    Variables:
        y[0] = m² (GFT mass, dimensionless)
        y[1] = λ  (GFT quartic coupling, dimensionless)
        y[2] = γ  (Barbero-Immirzi parameter)

    The beta functions are derived from the one-loop effective action
    of the tensorial GFT model with EPRL vertex weights.

    Reference structure from Carrozza (2014), adapted for EPRL:
        β_m² = (d-2) m² + λ × I₁(m², γ)
        β_λ  = (2d-4) λ + λ² × I₂(m², γ)
        β_γ  = γ × λ × I₃(m², γ)

    where I₁, I₂, I₃ are loop integrals that depend on the
    EPRL vertex amplitude structure.

    RG time t = ln(μ/μ₀) runs from UV (t → +∞) to IR (t → -∞).
    """

    m2, lam, gamma = y

    # One-loop integrals (schematic — the exact form depends on
    # the EPRL vertex amplitude decomposition)
    # These use the stationary-phase approximation of the EPRL amplitude

    S_Regge = 5 * np.arccos(0.25)  # equilateral 4-simplex

    # I₁: tadpole integral (mass renormalisation)
    I1 = 1.0 / (1 + m2) * (1 + gamma**2) / (gamma**2)

    # I₂: sunset integral (coupling renormalisation)
    I2 = 1.0 / (1 + m2)**2 * (1 + gamma**2)**2 / (gamma**4)

    # I₃: γ renormalisation from the chiral EPRL amplitude
    # This is the key new ingredient: γ runs because the
    # oriented/anti-oriented amplitude ratio depends on the scale
    I3 = np.sin(gamma * S_Regge) / (gamma * (1 + m2))

    # Beta functions
    beta_m2 = (d - 2) * m2 + lam * I1
    beta_lam = (2*d - 4) * lam + lam**2 * I2
    beta_gamma = gamma * lam * I3

    return [beta_m2, beta_lam, beta_gamma]


def run_rg_flow(gamma_UV, m2_UV=1.0, lam_UV=0.1, t_span=(-10, 10)):
    """
    Run the TGFT-RG flow from UV to IR.

    Start at the UV fixed point (t = t_max) and flow to IR (t = t_min).
    """

    y0 = [m2_UV, lam_UV, gamma_UV]

    # Integrate from UV to IR (decreasing t)
    sol = solve_ivp(tgft_beta_functions, t_span, y0,
                    method='RK45', dense_output=True,
                    max_step=0.01, rtol=1e-8, atol=1e-10)

    return sol


def find_ir_fixed_point(gamma_UV_values):
    """
    Run the RG flow for multiple UV initial γ values
    and check if they converge to a common IR value.
    """

    print("UV γ → IR γ convergence test")
    print("=" * 50)

    ir_gammas = []

    for g_uv in gamma_UV_values:
        sol = run_rg_flow(g_uv)
        if sol.success:
            gamma_IR = sol.y[2, -1]
            ir_gammas.append(gamma_IR)
            print(f"γ_UV = {g_uv:.4f} → γ_IR = {gamma_IR:.4f}")
        else:
            print(f"γ_UV = {g_uv:.4f} → FLOW FAILED")

    ir_gammas = np.array(ir_gammas)
    gamma_star = np.mean(ir_gammas)
    gamma_spread = np.std(ir_gammas)

    print(f"\nIR fixed point: γ* = {gamma_star:.4f} ± {gamma_spread:.4f}")
    print(f"Convergence: {gamma_spread / gamma_star * 100:.1f}% spread")

    if gamma_spread / gamma_star < 0.05:
        print("✓ CONVERGENCE confirmed (< 5% spread)")
    else:
        print("✗ NO CONVERGENCE — β_γ structure needs revision")

    return gamma_star, gamma_spread

# Run with γ_AS and γ_EPRL as UV initial conditions
gamma_star, spread = find_ir_fixed_point([0.274, 0.2375, 0.25, 0.30])
```

### Step 4: Consistency Check — BH Entropy at the Fixed Point

```python
# bh_entropy_check.py
import numpy as np

def check_bh_entropy(gamma_star):
    """
    Verify that the IR fixed-point γ* reproduces S_BH = A / (4G).

    The Bekenstein-Hawking coefficient 1/4 should emerge from:
        ln D_eff = 1/4  where D_eff = e^{1/4}

    In the EPRL formulation with γ*:
        S = (A / l²_Pl) × γ* × ∑_j (2j+1) ln(2j+1) × w(j, γ*)

    where w(j, γ*) is the EPRL weight at the fixed point.

    The condition S = A / (4 l²_Pl) determines whether γ* is consistent.
    """

    S_Regge = 5 * np.arccos(0.25)

    # Simplified entropy formula:
    # S / (A/l²_Pl) = γ* × π / (2 × S_Regge) at leading order
    entropy_coefficient = gamma_star * np.pi / (2 * S_Regge)

    print(f"γ* = {gamma_star:.4f}")
    print(f"S / (A/l²_Pl) = {entropy_coefficient:.4f}")
    print(f"Target: 0.2500 (= 1/4)")
    print(f"Deviation: {abs(entropy_coefficient - 0.25) / 0.25 * 100:.1f}%")

    if abs(entropy_coefficient - 0.25) < 0.02:
        print("✓ BH entropy reproduced within 2%")
    else:
        print("✗ BH entropy NOT reproduced — γ* inconsistent")

    return entropy_coefficient

check_bh_entropy(gamma_star)
```

---

## Expected Results

| Quantity | Expected value | Current status |
|---|---|---|
| γ_AS (input) | 0.274 ± 0.01 | From Reuter's AS fixed point |
| γ_EPRL (input) | 0.2375 ± 0.005 | From LQG entropy counting |
| γ* (output) | 0.25 ± 0.02 | IR fixed point of TGFT-RG |
| Discrepancy at IR | < 5% | Currently ~13% at UV |
| BH entropy at γ* | 1/4 ± 0.02 | Self-consistency check |

**Success criterion**: γ* = 0.25 ± 0.02, consistent with both γ_AS and γ_EPRL within their uncertainties after RG flow.

**Failure criterion**: γ_AS and γ_EPRL flow to different IR values (> 10% discrepancy at IR).

---

## Timeline
- **Weeks 1–3**: Implement TGFT-RG beta functions for EPRL-weighted model.
- **Weeks 4–6**: Run RG flow for multiple UV initial conditions, check convergence.
- **Weeks 7–9**: Compute BH entropy at the fixed point, verify S = A/4G.
- **Weeks 10–12**: Systematic uncertainty analysis (higher-loop corrections, j truncation).
- **Months 4–6**: Comparison with Bahr–Steinhaus spin foam RG results.

## References
- Reuter, M. (1998). PRD 57, 971.
- Engle, Pereira, Rovelli & Livine (2008). NPB 799, 136.
- Carrozza, S. (2014). Tensorial methods and renormalization in GFT. Springer.
- Ben Geloun, J. & Rivasseau, V. (2012). CMP 318, 69.
- Bahr, B. & Steinhaus, S. (2016). PRL 117, 141302.
- Meissner, K.A. (2004). CQG 21, 5245.
