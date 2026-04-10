#!/usr/bin/env python3
"""
P8b_gw_spectral_index.py

Tests whether the NANOGrav 15-yr GW background spectral index γ is
consistent with the CCDR prediction γ = 13/3 - 2κν ≈ 4.331
(for κ = 1, ν = 10⁻³).

This is a quick consistency check, NOT a precision test:
the current NANOGrav uncertainty (~0.6) is ~600× larger than the
predicted CCDR shift (~10⁻³). So the only meaningful question is:
"Does γ_CCDR fall within the 1σ band of the measured γ?"
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

DATA_DIR = Path("data/p8b_spectral_index")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# NANOGrav 15-yr published values (Agazie et al. 2023)
GAMMA_MEASURED = 3.2
GAMMA_SIGMA = 0.6  # 1-sigma uncertainty (free spectral index fit)
GAMMA_SMBHB = 13.0 / 3.0  # = 4.333 (fixed SMBHB prediction)

# CCDR parameters (from Layer 1: GFT condensate)
NU_CCDR = 1e-3
KAPPA = 1.0  # O(1) constant from CCDR derivation

# Predicted CCDR γ
GAMMA_CCDR = GAMMA_SMBHB - 2 * KAPPA * NU_CCDR

# Alternative: if α_CCDR = -2/3 + κν then γ = 3 - 2α
ALPHA_STANDARD = -2.0 / 3.0
ALPHA_CCDR = ALPHA_STANDARD + KAPPA * NU_CCDR
GAMMA_CCDR_ALT = 3.0 - 2 * ALPHA_CCDR


def evaluate_consistency(gamma_pred, gamma_meas, sigma):
    """How many sigma is the prediction from the measurement?"""
    z = (gamma_pred - gamma_meas) / sigma
    p_value = 2 * (1 - norm.cdf(abs(z)))  # two-sided
    return {
        "predicted": float(gamma_pred),
        "measured": float(gamma_meas),
        "sigma": float(sigma),
        "z_score": float(z),
        "p_value": float(p_value),
        "in_1sigma": bool(abs(z) < 1),
        "in_2sigma": bool(abs(z) < 2),
        "in_3sigma": bool(abs(z) < 3),
    }


def main():
    print("=" * 70)
    print("P8b: GW Background Spectral Index Test")
    print("=" * 70)
    print(f"NANOGrav 15-yr measured γ:     {GAMMA_MEASURED} ± {GAMMA_SIGMA}")
    print(f"SMBHB standard prediction:     γ = 13/3 = {GAMMA_SMBHB:.4f}")
    print(f"CCDR parameters: ν = {NU_CCDR}, κ = {KAPPA}")
    print(f"CCDR prediction:               γ = {GAMMA_CCDR:.6f}")
    print(f"  (= SMBHB - 2κν = {GAMMA_SMBHB:.4f} - {2*KAPPA*NU_CCDR:.6f})")
    print(f"Predicted shift from SMBHB:    Δγ = {-2*KAPPA*NU_CCDR:.6f}")
    print(f"Current uncertainty:           σ(γ) = {GAMMA_SIGMA}")
    print(f"Sensitivity ratio:             σ / |Δγ| = {GAMMA_SIGMA/abs(2*KAPPA*NU_CCDR):.0f}")
    print()

    # Test consistency with CCDR
    print("=" * 70)
    print("CONSISTENCY TESTS")
    print("=" * 70)

    print("\n1. Is the SMBHB prediction (γ = 13/3) consistent with NANOGrav?")
    smbhb_test = evaluate_consistency(GAMMA_SMBHB, GAMMA_MEASURED, GAMMA_SIGMA)
    print(f"   |z| = {abs(smbhb_test['z_score']):.2f}σ, p = {smbhb_test['p_value']:.3f}")
    print(f"   In 1σ: {smbhb_test['in_1sigma']}, In 2σ: {smbhb_test['in_2sigma']}")

    print("\n2. Is the CCDR prediction (γ = 13/3 - 2κν) consistent?")
    ccdr_test = evaluate_consistency(GAMMA_CCDR, GAMMA_MEASURED, GAMMA_SIGMA)
    print(f"   |z| = {abs(ccdr_test['z_score']):.2f}σ, p = {ccdr_test['p_value']:.3f}")
    print(f"   In 1σ: {ccdr_test['in_1sigma']}, In 2σ: {ccdr_test['in_2sigma']}")

    # Differential test: can we distinguish CCDR from SMBHB?
    print("\n3. Can we distinguish CCDR from SMBHB?")
    delta_gamma = abs(GAMMA_CCDR - GAMMA_SMBHB)
    sensitivity = GAMMA_SIGMA / delta_gamma
    print(f"   |γ_SMBHB - γ_CCDR| = {delta_gamma:.6f}")
    print(f"   Current σ(γ) = {GAMMA_SIGMA}")
    print(f"   Need {sensitivity:.0f}× tighter measurement to discriminate at 1σ")
    print(f"   Need {3*sensitivity:.0f}× tighter measurement for 3σ discrimination")

    # When could we test it?
    print("\n4. When can the prediction become testable?")
    print(f"   NANOGrav 15-yr σ:    {GAMMA_SIGMA}")
    print(f"   NANOGrav 20-yr σ:    ~{GAMMA_SIGMA * (15/20)**0.5:.2f}  (sqrt(T) scaling)")
    print(f"   SKA-PTA σ:           ~{GAMMA_SIGMA * 0.05:.3f}  (rough estimate)")
    print(f"   Required for CCDR:   {abs(2*KAPPA*NU_CCDR)/3:.6f}  (3σ test)")
    print(f"   Conclusion: even SKA-PTA cannot reach the CCDR precision.")
    print(f"   This test is essentially UNTESTABLE with foreseeable PTAs.")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if ccdr_test['in_1sigma']:
        verdict = "CONSISTENT (within 1σ) — but uncertainty too large to discriminate"
    elif ccdr_test['in_2sigma']:
        verdict = "MARGINALLY CONSISTENT (within 2σ)"
    else:
        verdict = "INCONSISTENT"
    print(f"  {verdict}")
    print()
    print("  Honest assessment: this test cannot rule out CCDR with current")
    print("  data. The CCDR prediction is buried in the SMBHB prediction at")
    print(f"  ~10⁻³ precision, while NANOGrav 15-yr measures γ to ±0.6.")
    print("  The SMBHB prediction itself is only at 1.9σ from the measured")
    print("  central value, so any prediction near 13/3 will be 'consistent'.")

    out = {
        "data_source": "Agazie et al. 2023, ApJL 951, L8",
        "gamma_measured": GAMMA_MEASURED,
        "gamma_sigma": GAMMA_SIGMA,
        "gamma_smbhb": GAMMA_SMBHB,
        "ccdr_parameters": {"nu": NU_CCDR, "kappa": KAPPA},
        "gamma_ccdr": GAMMA_CCDR,
        "delta_gamma_ccdr": -2 * KAPPA * NU_CCDR,
        "smbhb_test": smbhb_test,
        "ccdr_test": ccdr_test,
        "discrimination_factor_needed": float(GAMMA_SIGMA / abs(GAMMA_CCDR - GAMMA_SMBHB)),
        "verdict": verdict,
    }
    out_path = DATA_DIR / "result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))

        # Show measurement and predictions
        gamma_axis = np.linspace(GAMMA_MEASURED - 3 * GAMMA_SIGMA,
                                  GAMMA_MEASURED + 3 * GAMMA_SIGMA, 200)
        likelihood = norm.pdf(gamma_axis, GAMMA_MEASURED, GAMMA_SIGMA)
        ax.fill_between(gamma_axis, 0, likelihood, alpha=0.3, color="blue",
                       label=f"NANOGrav 15-yr: {GAMMA_MEASURED}±{GAMMA_SIGMA}")
        ax.axvline(GAMMA_SMBHB, color="green", lw=2, ls="--",
                  label=f"SMBHB: 13/3 = {GAMMA_SMBHB:.3f}")
        ax.axvline(GAMMA_CCDR, color="red", lw=2, ls=":",
                  label=f"CCDR: {GAMMA_CCDR:.3f}")
        ax.set_xlabel("Spectral index γ")
        ax.set_ylabel("Likelihood")
        ax.set_title("GW Background Spectral Index: NANOGrav vs Predictions")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("P8b_spectral_index.png", dpi=150)
        print("Plot: P8b_spectral_index.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
