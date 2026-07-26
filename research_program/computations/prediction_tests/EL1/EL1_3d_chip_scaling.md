# EL1: 3D Chip Information Throughput — Area vs Volume Scaling

## Prediction
Holographic principle: 3D NAND throughput scales with inter-layer AREA, not VOLUME. Doubling layers doubles effective capacity only if bandwidth scales with area.

## Software
```bash
pip install numpy matplotlib pandas
```

## Data (all public from datasheets/reviews)
Published specs from Samsung, Micron, SK Hynix, Kioxia:
- 32-layer V-NAND (2013): ~40 MB/s sequential read
- 48-layer (2015): ~50 MB/s
- 64-layer (2017): ~60 MB/s
- 96-layer (2018): ~80 MB/s
- 128-layer (2020): ~100 MB/s
- 176-layer (2021): ~130 MB/s
- 232-layer (2022): ~160 MB/s
- 300+ layer (2024): ~200 MB/s

Sources: AnandTech reviews, TechPowerUp, manufacturer press releases.
Also: HBM (High Bandwidth Memory) specs from SK Hynix.

## Code
```python
#!/usr/bin/env python3
"""EL1_3d_scaling.py — Test area vs volume scaling in 3D NAND."""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Data: (layers, sequential_read_MB_s, total_capacity_GB)
# Approximate values from public benchmarks
data = np.array([
    [32,  40,  256],
    [48,  50,  512],
    [64,  60,  512],
    [96,  80,  1024],
    [128, 100, 2048],
    [176, 130, 2048],
    [232, 160, 4096],
    [300, 200, 4096],
])

layers = data[:, 0]
throughput = data[:, 1]
capacity = data[:, 2]

def area_model(L, a, b): return a * L + b        # throughput ~ layers (area)
def volume_model(L, a, b): return a * L**1.5 + b  # throughput ~ layers^1.5
def linear(L, a, b): return a * L + b

popt_area, _ = curve_fit(area_model, layers, throughput)
popt_vol, _ = curve_fit(volume_model, layers, throughput)

resid_area = np.sum((throughput - area_model(layers, *popt_area))**2)
resid_vol = np.sum((throughput - volume_model(layers, *popt_vol))**2)

print("3D NAND Throughput Scaling")
print(f"Area model (T ~ L):     residual = {resid_area:.1f}")
print(f"Volume model (T ~ L^1.5): residual = {resid_vol:.1f}")
if resid_area < resid_vol:
    print("✓ AREA scaling preferred (holographic prediction)")
else:
    print("✗ VOLUME scaling preferred")

# Also fit power law: T ~ L^α
def power_law(L, a, alpha): return a * L**alpha
popt_pl, _ = curve_fit(power_law, layers, throughput, p0=[1, 1])
print(f"Power law: T ~ L^{popt_pl[1]:.2f}")
print(f"  α = 1.0 → area scaling (CCDR)")
print(f"  α = 1.5 → volume scaling")
print(f"  α = {popt_pl[1]:.2f} → measured")
```

## Expected: α ≈ 1.0 (area scaling). Timeline: 2–3 days.
