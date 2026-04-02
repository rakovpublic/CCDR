# BA2: Cosmic Web Chirality — Spiral Galaxy Handedness Asymmetry

## Prediction
~10⁻³ statistical excess of one handedness of spiral galaxies (CCDR: crystal chirality from CDT causal ordering propagates to large-scale structure).

## Software
```bash
pip install numpy scipy matplotlib pandas astropy
```

## Data (public)
- **Galaxy Zoo 2**: https://data.galaxyzoo.org/
  - ~300,000 galaxies with morphological classifications
  - Download: `gz2_classifications.csv` and `gz2_hart16.csv`
  - Columns: spiral arm direction (clockwise / anticlockwise votes)
- **Alternative**: SDSS DR17 spectroscopic catalogue with spiral classifications

## Method
1. Download Galaxy Zoo 2 data
2. Filter: galaxies with >50% spiral classification confidence
3. For each galaxy: assign handedness from majority vote (CW vs ACW)
4. Compute: N_CW / (N_CW + N_ACW) across full sky
5. Measure: dipole in handedness ratio as function of galactic coordinates
6. Compare: asymmetry amplitude with ~10⁻³ prediction

```python
#!/usr/bin/env python3
"""BA2_spiral_chirality.py — Test cosmic web handedness asymmetry."""
import numpy as np
import pandas as pd

def analyse_gz2(filename='gz2_hart16.csv'):
    df = pd.read_csv(filename)
    # Galaxy Zoo 2 columns for spiral direction:
    # t04_spiral_a08_cw_fraction, t04_spiral_a09_acw_fraction
    spiral = df[df['t01_smooth_or_features_a02_features_or_disk_fraction'] > 0.5]
    cw = spiral['t04_spiral_a08_cw_fraction'].values
    acw = spiral['t04_spiral_a09_acw_fraction'].values
    ra = spiral['ra'].values
    dec = spiral['dec'].values

    # Classify each galaxy
    is_cw = cw > acw
    N_cw = np.sum(is_cw)
    N_acw = np.sum(~is_cw)
    N_total = N_cw + N_acw
    asymmetry = (N_cw - N_acw) / N_total
    error = 1 / np.sqrt(N_total)

    print(f"N_CW = {N_cw}, N_ACW = {N_acw}, N_total = {N_total}")
    print(f"Asymmetry = {asymmetry:.6f} ± {error:.6f}")
    print(f"Significance: {abs(asymmetry)/error:.1f}σ")
    print(f"Target: ~10⁻³ = 0.001")

    # Dipole analysis: split sky into hemispheres
    for axis in ['ra', 'dec']:
        coords = ra if axis == 'ra' else dec
        median = np.median(coords)
        h1 = is_cw[coords < median]
        h2 = is_cw[coords >= median]
        a1 = np.mean(h1) - 0.5
        a2 = np.mean(h2) - 0.5
        dipole = a1 - a2
        print(f"Dipole ({axis}): {dipole:.6f}")

if __name__ == '__main__':
    print("Download gz2_hart16.csv from data.galaxyzoo.org")
    print("Then: python BA2_spiral_chirality.py")
    # analyse_gz2()  # uncomment after download
```

## Expected: Asymmetry ~ 10⁻³. Longo (2011) reported tentative evidence.
## Timeline: 1–2 weeks (data download + analysis)
## Success: >3σ dipole in handedness. Failure: consistent with zero at 3σ.
