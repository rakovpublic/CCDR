# P9a: Hadronic Angular Correlations at LHC — Hexagonal Lattice Signature

## Prediction
The cos(6θ) hexagonal fingerprint from the spacetime crystal BZ structure
should appear in dihadron angular correlations at ~10⁻³ level above QCD background.

## Software
```bash
pip install numpy scipy matplotlib pyhf uproot awkward
# ROOT: https://root.cern/ (optional but recommended)
```

## Data (public)
- **CERN Open Data Portal**: http://opendata.cern.ch/
  - CMS minimum-bias pp at √s = 7 TeV: /record/6021 (2010 data)
  - CMS dihadron correlations: already published (CMS-HIN-14-006)
  - ATLAS minimum-bias: /record/15001
- **HEPData**: https://www.hepdata.net/
  - Published dihadron correlation measurements from ALICE, CMS, ATLAS

## Method
1. Download minimum-bias pp collision data from CERN Open Data
2. Compute two-particle angular correlation function:
   C(Δφ, Δη) = (N_pairs_same / N_pairs_mixed) - 1
3. Project onto Δφ: C(Δφ) = ∫ C(Δφ, Δη) dΔη
4. Fit: C(Δφ) = a₀ + a₁cos(Δφ) + a₂cos(2Δφ) + ... + a₆cos(6Δφ)
5. Test: is a₆ significantly nonzero? (a₆ ~ 10⁻³ predicted)

Note: v₂ (cos(2Δφ), elliptic flow) is well-known in heavy-ion collisions.
The v₆ (cos(6Δφ)) coefficient has been measured in Pb-Pb but not carefully
examined in pp for a LATTICE origin. The CCDR prediction is that v₆ in pp
has a component from the spacetime crystal structure, not from flow.

```python
#!/usr/bin/env python3
"""P9a_dihadron.py — Look for cos(6θ) in published dihadron data."""
import numpy as np

# Example: use published Fourier coefficients from CMS
# CMS-HIN-14-006 measured v_n in pp at √s = 7 TeV
# v₂ ~ 0.06, v₃ ~ 0.02, v₄ ~ 0.01 in pp (measured)
# The question: what is v₆?

# Published values (approximate, from HEPData):
v_n_measured = {
    2: 0.06,   # ± 0.005
    3: 0.025,  # ± 0.005
    4: 0.012,  # ± 0.003
    5: 0.006,  # ± 0.002
    6: None,   # not separately reported in most pp analyses
}

# QCD prediction (no lattice): v_n ~ (v_2)^{n/2} for flow-like correlations
v6_qcd = v_n_measured[2]**3  # ~ 2×10⁻⁴

# CCDR prediction: additional cos(6θ) component ~ 10⁻³
v6_ccdr = v6_qcd + 1e-3

print("P9a: Hexagonal Signature in Dihadron Correlations")
print(f"v₂ (measured): {v_n_measured[2]:.4f}")
print(f"v₆ (QCD only): {v6_qcd:.6f}")
print(f"v₆ (QCD + CCDR lattice): {v6_ccdr:.6f}")
print(f"CCDR excess: {1e-3:.6f}")
print()
print("TO DO: Download CMS open data, compute v₆ in pp directly.")
print("If v₆ >> v₂³ by ~10⁻³, consistent with crystal lattice signature.")
```

## Expected: v₆ excess ~10⁻³ above QCD prediction v₂³.
## Timeline: 2–4 weeks (data processing heavy).
## This is the most speculative of the desktop-testable predictions.
