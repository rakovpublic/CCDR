# T1 on CDT-plusplus: Complete Instructions

## Goal
Measure the spectral dimension d_s(σ) of CDT C-phase triangulations
and search for oscillations with amplitude δd ~ ν ~ 10⁻³.

## Important Caveat
CDT-plusplus is actively developed. The API, build system, and command-line
options may have changed since these instructions were written (March 2026).
Always check the repository README and documentation first:
  https://github.com/acgetchell/CDT-plusplus

If anything below contradicts the current README, follow the README.

---

## PART 1: INSTALLATION

### 1.1 System Requirements

**Operating System:** Linux (recommended), macOS, or Windows with WSL2.

**Compiler:** C++20 compatible:
- GCC 11+ (recommended)
- Clang 14+
- MSVC 2022+ (Windows)

**Your machine (i9 Kaby Lake, 32 GB RAM):** More than sufficient for 2D and 3D.
4D with N₄ > 10⁵ will be slow but feasible.

### 1.2 Install Dependencies

```bash
# Ubuntu / Debian / WSL2
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libcgal-dev \
    libeigen3-dev \
    libtbb-dev \
    libboost-all-dev \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-matplotlib

# Verify CGAL version (need 5.x+)
apt list --installed 2>/dev/null | grep cgal
```

```powershell
# Windows (native, using vcpkg — but WSL2 is easier)
# Install Visual Studio 2022 with C++ workload
# Then:
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install cgal eigen3 tbb boost
```

**Recommended: Use WSL2 on your Windows machine.**
```powershell
# In PowerShell (admin):
wsl --install -d Ubuntu-24.04
# Then follow the Linux instructions inside WSL2
```

### 1.3 Clone and Build CDT-plusplus

```bash
# Clone
cd ~
git clone https://github.com/acgetchell/CDT-plusplus.git
cd CDT-plusplus

# Read the current build instructions
cat README.md

# The standard build (as of early 2026):
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -j$(nproc)

# If cmake fails, try without Ninja:
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j$(nproc)

# Run the test suite to verify everything works
ctest --test-dir . --output-on-failure
```

**If build fails — common fixes:**

```bash
# CGAL not found:
sudo apt install libcgal-dev
# or specify path:
cmake .. -DCGAL_DIR=/usr/lib/cmake/CGAL

# TBB not found:
sudo apt install libtbb-dev

# Boost not found:
sudo apt install libboost-all-dev

# C++20 not supported:
sudo apt install gcc-12 g++-12
export CC=gcc-12 CXX=g++-12
cmake .. -DCMAKE_BUILD_TYPE=Release
```

### 1.4 Verify Installation

```bash
# After successful build, you should have executables in build/src/
ls build/src/cdt-opt  # or similar name — check what was built
# The exact executable name depends on the version

# Also check for any Python utilities:
find . -name "*.py" -path "*/scripts/*" | head -20
```

---

## PART 2: UNDERSTANDING CDT-PLUSPLUS STRUCTURE

### 2.1 What CDT-plusplus Does

CDT-plusplus implements Causal Dynamical Triangulations in 3D and 4D
(and sometimes 2D). It:

1. Generates an initial triangulation with a specified number of simplices
2. Performs Pachner moves (bistellar flips) with Metropolis acceptance
   using the Regge action with coupling constants (k0, k3/k4, lambda)
3. Records configurations at regular intervals
4. Computes observables (volume profile, etc.)

### 2.2 Key Parameters

```
Dimension: 3 or 4 (start with 3 — much faster)
N_simplices: target number of top-dimensional simplices
  - 3D: N₃ = number of tetrahedra (start with 10000)
  - 4D: N₄ = number of 4-simplices (start with 10000)
k0: inverse bare Newton coupling (controls geometry)
  - C-phase in 3D: k0 ~ 1.0-3.0
  - C-phase in 4D: k0 ~ 2.0-2.5
k3 (or k4): cosmological constant tuning
  - Adjusted to maintain target volume
lambda: cosmological constant (alternative parameterisation)
passes: number of Monte Carlo sweeps
  - Each sweep = N_simplices attempted moves
  - Thermalisation: 100-500 passes
  - Production: 1000-10000 passes
checkpoint: save configuration every N passes
```

### 2.3 The C-Phase

The C-phase (the physically interesting phase) is characterised by:
- Volume profile ⟨N_{d-1}(t)⟩ ~ cos³(πt/T) (de Sitter shape)
- Spectral dimension d_s → d (topological dimension) at large σ
- d_s → ~2 at small σ (dynamical dimensional reduction)
- No blob (A-phase) or crumpled (B-phase) pathology

**You MUST verify you are in the C-phase before any measurement.**

---

## PART 3: GENERATING C-PHASE TRIANGULATIONS

### 3.1 Explore the CLI

```bash
# Check available options (the exact syntax depends on version)
cd ~/CDT-plusplus/build

# Try:
./src/cdt-opt --help
# or:
./src/cdt --help
# or check the source for the main function:
grep -r "main(" ../src/*.cpp | head -5
```

### 3.2 Run a 3D CDT Simulation (Start Here)

3D is much faster than 4D and still shows the spectral dimension flow.
Your i9 can handle 3D with N₃ = 50000 in reasonable time.

```bash
# The exact command depends on CDT-plusplus version.
# Typical invocation (check --help for your version):

./src/cdt-opt \
    --dimension 3 \
    --simplices 10000 \
    --timeslices 16 \
    --k0 2.0 \
    --lambda 0.6 \
    --passes 2000 \
    --checkpoint 100 \
    --output ~/cdt_data/run_3d_k2p0

# This should:
# 1. Create an initial triangulation with ~10000 tetrahedra
# 2. Run 2000 sweeps of Pachner moves
# 3. Save a checkpoint every 100 sweeps
# 4. Output to ~/cdt_data/run_3d_k2p0
```

**If the CLI is different in your version**, look at:
```bash
# The source code for parameter handling:
grep -r "simplices\|timeslices\|k0\|lambda\|passes" ../src/*.cpp ../src/*.hpp
# Or check for a config file format:
find .. -name "*.toml" -o -name "*.json" -o -name "*.yaml" | head -10
```

### 3.3 Parameter Scan to Find C-Phase

The C-phase occupies a specific region in (k0, lambda) space.
Run a parameter scan:

```bash
mkdir -p ~/cdt_data/phase_scan

for k0 in 1.0 1.5 2.0 2.5 3.0; do
    for lam in 0.2 0.4 0.6 0.8 1.0; do
        echo "Running k0=$k0, lambda=$lam"
        ./src/cdt-opt \
            --dimension 3 \
            --simplices 5000 \
            --timeslices 16 \
            --k0 $k0 \
            --lambda $lam \
            --passes 500 \
            --output ~/cdt_data/phase_scan/k${k0}_l${lam}
    done
done
```

Then examine the volume profiles to identify which (k0, lambda) give the
de Sitter cos³ shape (C-phase).

### 3.4 Reading CDT-plusplus Output

CDT-plusplus saves triangulations in a format that can be read back.
Check what format your version uses:

```bash
# Look for output files
ls ~/cdt_data/run_3d_k2p0/

# Common formats:
# - .off files (Object File Format — vertices + simplices)
# - custom binary format
# - serialised CGAL triangulation

# Check the source for I/O:
grep -r "save\|write\|output\|serialize" ../src/*.cpp ../src/*.hpp | head -20
```

---

## PART 4: MEASURING SPECTRAL DIMENSION

### 4.1 The Method

The spectral dimension is computed from the return probability of a
diffusion process (random walk) on the triangulation:

```
P(σ) = probability that a random walk returns to its starting simplex
       after σ steps

d_s(σ) = -2 × d ln P(σ) / d ln σ
```

For a healthy CDT triangulation:
- d_s → d (topological dimension) at large σ
- d_s → ~2 at small σ (UV dimensional reduction)
- The transition happens at σ ~ N^(2/d) (the diffusion scale)

### 4.2 Check if CDT-plusplus Has Built-in Spectral Dimension

```bash
# Search for spectral dimension in the source:
grep -ri "spectral\|random.walk\|return.prob\|diffusion" \
    ../src/*.cpp ../src/*.hpp ../include/*.hpp 2>/dev/null

# If found: use the built-in measurement
# If not found: implement it externally (see Section 4.3)
```

### 4.3 External Spectral Dimension Measurement (Python)

If CDT-plusplus does not have a built-in spectral dimension measurement,
or if you want more control, implement it in Python.

You need to:
1. Load the triangulation from CDT-plusplus output
2. Build the dual graph (adjacency of top-dimensional simplices)
3. Run random walks on the dual graph
4. Compute return probability and spectral dimension

```python
#!/usr/bin/env python3
"""
T1_spectral_dimension_cdtpp.py
Compute spectral dimension from CDT-plusplus output.

IMPORTANT: You need to adapt the load_triangulation() function
to match the output format of YOUR version of CDT-plusplus.
"""
import numpy as np
from collections import defaultdict
import glob
import os
import struct
import time

# ============================================================
# SECTION A: LOADING CDT-PLUSPLUS OUTPUT
# ============================================================

def load_triangulation_off(filename):
    """
    Load a triangulation from .off (Object File Format).
    Returns: adjacency dict (simplex_id -> set of neighbour simplex_ids)

    OFF format:
    OFF
    n_vertices n_faces n_edges
    x y z       (vertex coordinates, n_vertices lines)
    3 v1 v2 v3  (faces as vertex index lists, n_faces lines)
    """
    adj = defaultdict(set)
    vertices = []
    faces = []

    with open(filename, 'r') as f:
        header = f.readline().strip()
        if header != 'OFF':
            raise ValueError(f"Not an OFF file: {header}")
        counts = f.readline().split()
        n_v, n_f, n_e = int(counts[0]), int(counts[1]), int(counts[2])

        # Read vertices
        for _ in range(n_v):
            coords = [float(x) for x in f.readline().split()]
            vertices.append(coords)

        # Read faces (simplices)
        for i in range(n_f):
            parts = [int(x) for x in f.readline().split()]
            n_verts = parts[0]
            face_verts = tuple(sorted(parts[1:1+n_verts]))
            faces.append(face_verts)

    # Build adjacency: two simplices are adjacent if they share d vertices
    # (For tetrahedra in 3D: share a triangular face = 3 vertices)
    # This is O(n²) for simplicity; for large N use a hash map

    from collections import Counter
    # Build face-to-simplex map
    d = len(faces[0]) - 1  # dimension
    face_map = defaultdict(list)
    for i, simplex in enumerate(faces):
        # Each d-simplex has d+1 sub-faces of dimension d-1
        for j in range(len(simplex)):
            sub_face = tuple(v for k, v in enumerate(simplex) if k != j)
            face_map[sub_face].append(i)

    # Two simplices sharing a (d-1)-face are neighbours
    for sub_face, simplex_ids in face_map.items():
        if len(simplex_ids) == 2:
            adj[simplex_ids[0]].add(simplex_ids[1])
            adj[simplex_ids[1]].add(simplex_ids[0])

    n_simplices = len(faces)
    print(f"Loaded: {n_simplices} simplices, {len(vertices)} vertices, dim={d}")

    # Verify connectivity
    max_neighbours = max(len(v) for v in adj.values()) if adj else 0
    avg_neighbours = np.mean([len(v) for v in adj.values()]) if adj else 0
    print(f"Adjacency: max neighbours={max_neighbours}, avg={avg_neighbours:.1f}")

    return adj, n_simplices, vertices, faces


def load_triangulation_cgal(filename):
    """
    Load a CGAL serialised triangulation.
    This is more complex — CGAL uses its own binary/text format.

    You may need to write a small C++ program that:
    1. Loads the CGAL triangulation
    2. Outputs the adjacency list as a text file
    3. Then read that text file in Python

    See Section 4.4 for the C++ helper.
    """
    raise NotImplementedError(
        "CGAL format requires a C++ helper. See Section 4.4 in the .md file.")


def load_triangulation_auto(filename):
    """Try to auto-detect format and load."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.off':
        return load_triangulation_off(filename)
    else:
        # Try text format: each line = list of vertex indices of a simplex
        try:
            simplices = []
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = [int(x) for x in line.split()]
                        if len(parts) >= 3:
                            simplices.append(tuple(sorted(parts)))

            if simplices:
                print(f"Loaded {len(simplices)} simplices from text format")
                # Build adjacency
                d = len(simplices[0]) - 1
                adj = defaultdict(set)
                face_map = defaultdict(list)
                for i, simplex in enumerate(simplices):
                    for j in range(len(simplex)):
                        sub = tuple(v for k, v in enumerate(simplex) if k != j)
                        face_map[sub].append(i)
                for sub, ids in face_map.items():
                    if len(ids) == 2:
                        adj[ids[0]].add(ids[1])
                        adj[ids[1]].add(ids[0])
                return adj, len(simplices), None, simplices
        except:
            pass

        raise ValueError(f"Cannot load {filename} — unknown format. "
                        f"Use the C++ helper (Section 4.4) to export adjacency.")


# ============================================================
# SECTION B: SPECTRAL DIMENSION MEASUREMENT
# ============================================================

def random_walk_return_probability(adj, n_simplices, sigma, n_walks=1000):
    """
    Compute the return probability P(σ) by random walks on the dual graph.

    adj: adjacency dict (simplex_id -> set of neighbours)
    sigma: number of walk steps
    n_walks: number of independent walks to average over
    """
    simplex_ids = list(adj.keys())
    if not simplex_ids:
        return 0.0

    returns = 0
    valid_walks = 0

    for _ in range(n_walks):
        # Start at a random simplex
        s0 = simplex_ids[np.random.randint(len(simplex_ids))]
        s = s0

        # Walk sigma steps
        valid = True
        for _ in range(sigma):
            neighbours = list(adj[s])
            if not neighbours:
                valid = False
                break
            s = neighbours[np.random.randint(len(neighbours))]

        if valid:
            valid_walks += 1
            if s == s0:
                returns += 1

    if valid_walks == 0:
        return 0.0
    return returns / valid_walks


def compute_spectral_dimension(adj, n_simplices,
                                sigma_min=5, sigma_max=500,
                                n_points=40, n_walks=2000):
    """
    Compute d_s(σ) = -2 d ln P / d ln σ.

    Returns: sigma_values, d_s_values, P_values
    """
    sigma_values = np.unique(
        np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_points).astype(int)
    )
    sigma_values = sigma_values[sigma_values >= 2]

    print(f"Computing P(σ) for {len(sigma_values)} σ values "
          f"({sigma_values[0]} to {sigma_values[-1]}), "
          f"{n_walks} walks each...")

    P_values = []
    for i, sigma in enumerate(sigma_values):
        t0 = time.time()
        P = random_walk_return_probability(adj, n_simplices, sigma, n_walks)
        P_values.append(max(P, 1e-10))  # avoid log(0)
        elapsed = time.time() - t0
        if (i + 1) % 5 == 0:
            print(f"  σ={sigma:5d}: P={P:.6f}, "
                  f"time={elapsed:.1f}s")

    P_values = np.array(P_values)
    log_sigma = np.log(sigma_values.astype(float))
    log_P = np.log(P_values)

    # Numerical derivative using central differences
    d_s = np.zeros_like(log_P)
    for i in range(1, len(log_P) - 1):
        d_s[i] = -2 * (log_P[i+1] - log_P[i-1]) / \
                      (log_sigma[i+1] - log_sigma[i-1])
    d_s[0] = d_s[1]
    d_s[-1] = d_s[-2]

    return sigma_values, d_s, P_values


# ============================================================
# SECTION C: VALIDATION
# ============================================================

def validate_c_phase(adj, n_simplices, dimension=3):
    """
    Validate that the triangulation is in the C-phase.

    Check 1: spectral dimension → d at large σ
    Check 2: volume profile is de Sitter (cos³)
    Check 3: no pathological connectivity (blob or crumpled)
    """
    print("\n" + "=" * 60)
    print("C-PHASE VALIDATION")
    print("=" * 60)

    # Check 1: Quick spectral dimension at large σ
    print("\nCheck 1: Spectral dimension at large σ...")
    sigma_large = min(n_simplices // 2, 500)
    P_large = random_walk_return_probability(adj, n_simplices,
                                              sigma_large, n_walks=500)
    # For a d-dimensional space: P(σ) ~ σ^(-d/2)
    # So at large σ, d_s ≈ d
    P_small = random_walk_return_probability(adj, n_simplices,
                                              sigma_large // 5, n_walks=500)
    if P_large > 0 and P_small > 0:
        d_s_estimate = -2 * (np.log(P_large) - np.log(P_small)) / \
                            (np.log(sigma_large) - np.log(sigma_large // 5))
    else:
        d_s_estimate = 0

    print(f"  d_s(σ={sigma_large}) ≈ {d_s_estimate:.2f}")
    print(f"  Expected: {dimension:.1f}")
    check1_pass = abs(d_s_estimate - dimension) < 1.0

    # Check 2: Connectivity (no blob)
    print("\nCheck 2: Connectivity...")
    degrees = [len(adj[s]) for s in adj]
    mean_deg = np.mean(degrees)
    std_deg = np.std(degrees)
    max_deg = max(degrees)
    min_deg = min(degrees)
    print(f"  Mean degree: {mean_deg:.1f}")
    print(f"  Std degree:  {std_deg:.1f}")
    print(f"  Range: [{min_deg}, {max_deg}]")
    # In C-phase: relatively uniform degree distribution
    # In A-phase (blob): one or few simplices have very high degree
    check2_pass = (max_deg < 10 * mean_deg) and (min_deg >= 1)

    # Check 3: Giant component (not fragmented)
    print("\nCheck 3: Connectivity (not fragmented)...")
    visited = set()
    queue = [list(adj.keys())[0]]
    while queue:
        s = queue.pop()
        if s in visited:
            continue
        visited.add(s)
        for n in adj[s]:
            if n not in visited:
                queue.append(n)
    frac_connected = len(visited) / len(adj)
    print(f"  Fraction in giant component: {frac_connected:.4f}")
    check3_pass = frac_connected > 0.95

    # Summary
    all_pass = check1_pass and check2_pass and check3_pass
    print(f"\n  Check 1 (d_s → {dimension}): {'PASS' if check1_pass else 'FAIL'}")
    print(f"  Check 2 (connectivity):   {'PASS' if check2_pass else 'FAIL'}")
    print(f"  Check 3 (giant component): {'PASS' if check3_pass else 'FAIL'}")
    print(f"  Overall: {'✓ C-PHASE VALIDATED' if all_pass else '✗ NOT C-PHASE'}")

    if not all_pass:
        print("\n  RECOMMENDATIONS:")
        if not check1_pass:
            print("  - d_s wrong: try different k0/lambda; need more thermalisation")
        if not check2_pass:
            print("  - Blob detected: you may be in A-phase; decrease k0")
        if not check3_pass:
            print("  - Fragmented: triangulation degenerated; restart")

    return all_pass, d_s_estimate


# ============================================================
# SECTION D: OSCILLATION SEARCH
# ============================================================

def search_oscillations(sigma_values, d_s_values, dimension=3):
    """
    Search for oscillations in d_s(σ) around the smooth trend.

    Method:
    1. Fit a smooth curve (polynomial in log σ) to d_s(σ)
    2. Compute residuals
    3. Measure RMS of residuals = amplitude of oscillation
    4. Compare with target δd ~ ν ~ 10⁻³
    """
    valid = np.isfinite(d_s_values) & (d_s_values > 0) & (d_s_values < 2*dimension)
    if np.sum(valid) < 10:
        print("Too few valid d_s points for oscillation search")
        return None, None

    log_s = np.log(sigma_values[valid].astype(float))
    ds = d_s_values[valid]

    # Fit polynomial of degree 3 to capture the smooth trend
    # (d_s goes from ~2 at small σ to ~d at large σ)
    try:
        coeffs = np.polyfit(log_s, ds, 3)
        ds_smooth = np.polyval(coeffs, log_s)
    except:
        ds_smooth = np.mean(ds) * np.ones_like(ds)

    residuals = ds - ds_smooth
    delta_d = np.sqrt(np.mean(residuals**2))

    return delta_d, residuals


def run_T1_ensemble(data_dir, dimension=3, n_walks=2000):
    """
    Run the full T1 test across an ensemble of CDT configurations.

    data_dir: directory containing CDT-plusplus checkpoint files
    """
    print("=" * 70)
    print("T1 TEST: Spectral Dimension Oscillation on CDT-plusplus Output")
    print(f"Target: δd ~ ν ~ 10⁻³")
    print(f"Data directory: {data_dir}")
    print("=" * 70)

    # Find checkpoint files
    patterns = ['*.off', '*.dat', '*.tri', '*.cgal', 'checkpoint_*']
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(os.path.join(data_dir, pat))))
        files.extend(sorted(glob.glob(os.path.join(data_dir, '**', pat),
                                       recursive=True)))
    files = sorted(set(files))

    if not files:
        print(f"\nNo triangulation files found in {data_dir}")
        print("Check the output format of your CDT-plusplus version.")
        print("You may need to use the C++ helper (Section 4.4) to export.")
        return

    print(f"Found {len(files)} checkpoint files")

    all_ds = []
    sigma_ref = None
    valid_configs = 0

    for i, filepath in enumerate(files):
        print(f"\nConfig {i+1}/{len(files)}: {os.path.basename(filepath)}")
        try:
            adj, n_sim, _, _ = load_triangulation_auto(filepath)
        except Exception as e:
            print(f"  Error loading: {e}")
            continue

        # Validate C-phase (only on first config)
        if valid_configs == 0:
            is_c, d_s_est = validate_c_phase(adj, n_sim, dimension)
            if not is_c:
                print("  ✗ First config is not C-phase. Check parameters.")
                print("  Continuing to check other configs...")
                continue

        # Compute spectral dimension
        sigmas, ds, Ps = compute_spectral_dimension(
            adj, n_sim,
            sigma_min=5,
            sigma_max=min(n_sim // 3, 1000),
            n_points=30,
            n_walks=n_walks
        )

        if sigma_ref is None:
            sigma_ref = sigmas

        if len(ds) == len(sigma_ref):
            all_ds.append(ds)
            valid_configs += 1
            print(f"  ✓ d_s at large σ: {ds[-5:].mean():.3f}")

    if valid_configs < 5:
        print(f"\nOnly {valid_configs} valid configs. Need at least 5.")
        print("Generate more CDT-plusplus checkpoints.")
        return

    # Ensemble analysis
    all_ds = np.array(all_ds)
    ds_mean = np.mean(all_ds, axis=0)
    ds_std = np.std(all_ds, axis=0) / np.sqrt(valid_configs)

    # Validate ensemble d_s
    ds_large_sigma = ds_mean[-5:]
    ds_ensemble = np.mean(ds_large_sigma)
    print(f"\nEnsemble ⟨d_s⟩ at large σ: {ds_ensemble:.3f} (expected: {dimension})")

    if abs(ds_ensemble - dimension) > 1.0:
        print(f"✗ VALIDATION FAILED: d_s = {ds_ensemble:.2f} ≠ {dimension}")
        print("Results below are UNRELIABLE.")

    # Oscillation search
    delta_d, residuals = search_oscillations(sigma_ref, ds_mean, dimension)

    if delta_d is not None:
        mean_err = np.mean(ds_std[np.isfinite(ds_std) & (ds_std > 0)])
        significance = delta_d / mean_err if mean_err > 0 else 0

        print(f"\n{'=' * 70}")
        print(f"T1 RESULTS")
        print(f"{'=' * 70}")
        print(f"Valid configurations: {valid_configs}")
        print(f"⟨d_s⟩ at large σ: {ds_ensemble:.3f}")
        print(f"Residual RMS (δd): {delta_d:.6f}")
        print(f"Target δd = ν: 0.001")
        print(f"Mean statistical error: {mean_err:.6f}")
        print(f"Significance: {significance:.1f}σ")

        if abs(ds_ensemble - dimension) > 1.0:
            print(f"\n⚠ WARNING: d_s validation FAILED. Result unreliable.")
        elif significance > 3 and delta_d < 0.1:
            print(f"\n✓ OSCILLATION DETECTED at {significance:.1f}σ")
            if abs(np.log10(delta_d) - (-3)) < 0.5:
                print(f"  δd = {delta_d:.2e} ≈ 10⁻³ — CONSISTENT with ν!")
            else:
                print(f"  δd = {delta_d:.2e} ≠ 10⁻³")
        else:
            print(f"\n✗ No significant oscillation at current sensitivity")
            if mean_err > 0 and significance > 0:
                n_needed = int(valid_configs * (3 / significance)**2)
                print(f"  Need ~{n_needed} configs for 3σ")

        # Save
        np.savez('T1_cdtpp_results.npz',
                 sigma=sigma_ref, ds_mean=ds_mean, ds_std=ds_std,
                 delta_d=delta_d, significance=significance,
                 ds_ensemble=ds_ensemble, n_configs=valid_configs)

        # Plot
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(12, 9))

            valid = np.isfinite(ds_mean)
            axes[0].errorbar(sigma_ref[valid], ds_mean[valid],
                            yerr=ds_std[valid], fmt='o', markersize=3,
                            capsize=2, label=f'CDT-plusplus ({valid_configs} configs)')
            axes[0].axhline(dimension, color='green', ls='--', alpha=0.5,
                           label=f'd_s = {dimension}')
            axes[0].axhline(2, color='orange', ls=':', alpha=0.5, label='d_s = 2')
            axes[0].set_xlabel('σ (diffusion steps)', fontsize=13)
            axes[0].set_ylabel('d_s(σ)', fontsize=13)
            axes[0].set_xscale('log')
            axes[0].set_title('T1: Spectral Dimension from CDT-plusplus', fontsize=15)
            axes[0].legend(fontsize=11)

            if residuals is not None:
                log_s = np.log(sigma_ref[valid].astype(float))
                axes[1].plot(log_s[:len(residuals)], residuals, 'o',
                            markersize=3)
                axes[1].axhline(0, color='k', ls='--')
                axes[1].axhline(1e-3, color='r', ls=':', label='ν = 10⁻³')
                axes[1].axhline(-1e-3, color='r', ls=':')
                axes[1].set_xlabel('ln σ', fontsize=13)
                axes[1].set_ylabel('Residual', fontsize=13)
                axes[1].set_title(f'Oscillatory residual: δd = {delta_d:.6f}, '
                                 f'{significance:.1f}σ', fontsize=14)
                axes[1].legend(fontsize=11)

            plt.tight_layout()
            plt.savefig('T1_cdtpp_spectral_dimension.png', dpi=150)
            print(f"\nPlot: T1_cdtpp_spectral_dimension.png")
        except ImportError:
            pass


# ============================================================
# SECTION E: MAIN
# ============================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python T1_spectral_dimension_cdtpp.py <data_dir> [dimension]")
        print()
        print("  data_dir:  directory with CDT-plusplus checkpoint files")
        print("  dimension: 3 or 4 (default: 3)")
        print()
        print("Example:")
        print("  python T1_spectral_dimension_cdtpp.py ~/cdt_data/run_3d_k2p0 3")
        sys.exit(1)

    data_dir = sys.argv[1]
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    run_T1_ensemble(data_dir, dimension=dim)
```

---

## PART 4 (continued): C++ ADJACENCY EXPORTER

### 4.4 C++ Helper to Export Adjacency

If CDT-plusplus uses CGAL serialisation (not .off), you need a small
C++ program to load the triangulation and export the adjacency list:

```cpp
// export_adjacency.cpp
// Compile: g++ -std=c++20 -O2 export_adjacency.cpp -lCGAL -lgmp -o export_adj
// Usage:   ./export_adj input_triangulation.cgal output_adjacency.txt

// NOTE: This is a TEMPLATE. You need to adapt it to match the specific
// CGAL triangulation type used by YOUR version of CDT-plusplus.
// Check the CDT-plusplus source for the typedef of the triangulation.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <fstream>
#include <iostream>
#include <map>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Triangulation;
typedef Triangulation::Cell_handle Cell_handle;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " input.cgal output_adj.txt" << std::endl;
        return 1;
    }

    // Load triangulation
    Triangulation T;
    std::ifstream fin(argv[1]);
    fin >> T;
    fin.close();

    std::cout << "Loaded: " << T.number_of_cells() << " cells, "
              << T.number_of_vertices() << " vertices" << std::endl;

    // Map cells to integer IDs
    std::map<Cell_handle, int> cell_id;
    int id = 0;
    for (auto c = T.finite_cells_begin(); c != T.finite_cells_end(); ++c) {
        cell_id[c] = id++;
    }

    // Export adjacency
    std::ofstream fout(argv[2]);
    fout << "# cell_id neighbour_id_1 neighbour_id_2 ..." << std::endl;
    for (auto c = T.finite_cells_begin(); c != T.finite_cells_end(); ++c) {
        fout << cell_id[c];
        for (int i = 0; i < 4; ++i) {  // 4 neighbours in 3D
            Cell_handle nb = c->neighbor(i);
            if (!T.is_infinite(nb)) {
                fout << " " << cell_id[nb];
            }
        }
        fout << std::endl;
    }
    fout.close();

    std::cout << "Exported " << id << " cells to " << argv[2] << std::endl;
    return 0;
}
```

Then load the adjacency in Python:

```python
def load_adjacency_txt(filename):
    """Load adjacency list from the C++ helper output."""
    adj = defaultdict(set)
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = [int(x) for x in line.split()]
            if len(parts) >= 2:
                cell_id = parts[0]
                for nb in parts[1:]:
                    adj[cell_id].add(nb)
                    adj[nb].add(cell_id)
    n_simplices = max(adj.keys()) + 1 if adj else 0
    print(f"Loaded adjacency: {n_simplices} simplices")
    return adj, n_simplices, None, None
```

---

## PART 5: COMPLETE WORKFLOW

### Step-by-step (copy-paste ready):

```bash
# STEP 1: Install (once)
sudo apt update && sudo apt install -y build-essential cmake git \
    libcgal-dev libeigen3-dev libtbb-dev libboost-all-dev ninja-build \
    python3 python3-pip python3-numpy python3-scipy python3-matplotlib
cd ~
git clone https://github.com/acgetchell/CDT-plusplus.git
cd CDT-plusplus && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -j$(nproc)

# STEP 2: Generate C-phase triangulations
# (adapt the command to your CDT-plusplus version — check --help)
mkdir -p ~/cdt_data/t1_run
# Run CDT simulation — CHECK YOUR VERSION'S CLI:
./src/cdt-opt --dimension 3 --simplices 10000 --timeslices 16 \
    --k0 2.0 --lambda 0.6 --passes 5000 --checkpoint 50 \
    --output ~/cdt_data/t1_run

# This should produce ~100 checkpoint files (5000 passes / 50 = 100)

# STEP 3: Run the T1 analysis
cd ~/CDT-plusplus
python3 T1_spectral_dimension_cdtpp.py ~/cdt_data/t1_run 3

# STEP 4: Check results
# The script prints:
#   - C-phase validation (d_s → 3, connectivity, giant component)
#   - Spectral dimension d_s(σ)
#   - Oscillation amplitude δd
#   - Significance
#   - Plot: T1_cdtpp_spectral_dimension.png
```

### Expected timeline on your machine:

| Step | Time |
|---|---|
| Install deps | 10 min |
| Build CDT-plusplus | 5–15 min |
| Generate 100 configs (3D, N₃=10000) | 2–6 hours |
| T1 analysis (100 configs) | 1–3 hours |
| **Total** | **4–10 hours** |

For better statistics (500 configs): run overnight (12–24 hours total).

### Expected results:

| Quantity | Expected (3D C-phase) |
|---|---|
| ⟨d_s⟩ at large σ | 3.0 ± 0.2 |
| d_s at small σ | ~2 (UV reduction) |
| δd (residual RMS) | depends on ν |
| Acceptance rate | 30–50% |

---

## PART 6: TROUBLESHOOTING

### Problem: CDT-plusplus won't build
```bash
# Check CMake output for missing dependencies
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -i "not found\|error\|missing"
# Most common: CGAL, TBB, or Boost missing
```

### Problem: Don't know the CLI options
```bash
# Check the source:
grep -r "po::options_description\|add_options\|argparse\|CLI::" ../src/*.cpp
# This will show you what command-line options exist
```

### Problem: Output format unknown
```bash
# Check what files were created:
ls -la ~/cdt_data/t1_run/
file ~/cdt_data/t1_run/*  # shows file types
head -20 ~/cdt_data/t1_run/*  # shows text content if any
```

### Problem: d_s ≠ 3 (not in C-phase)
```bash
# You're probably in the wrong phase. Try:
# - Lower k0 (toward A-phase boundary → then increase slightly)
# - More thermalisation passes
# - Different lambda
# Check the literature for C-phase parameters:
# Ambjorn, Jurkiewicz, Loll (2004) for 4D
# Ambjorn, Loll (1998) for 2D
```

### Problem: Python can't load CDT-plusplus output
```bash
# Use the C++ helper (Section 4.4):
# 1. Find the triangulation type in CDT-plusplus source:
grep -r "typedef.*Triangulation\|using.*Triangulation" ../src/ ../include/
# 2. Adapt export_adjacency.cpp to use the same type
# 3. Compile and run
# 4. Load the text adjacency in Python
```

---

## PART 7: WHAT SUCCESS LOOKS LIKE

A successful T1 run produces a plot with two panels:

**Top panel**: d_s(σ) showing:
- d_s ≈ 2 at small σ (UV dimensional reduction — the known CDT result)
- d_s ≈ 3 (for 3D) or 4 (for 4D) at large σ
- Smooth transition between them

**Bottom panel**: residuals showing:
- Oscillation amplitude δd ~ 10⁻³ (if the CCDR prediction is correct)
- OR: flat residuals consistent with zero (if the prediction is wrong)

**The d_s → 2 flow is already a confirmed CDT result** (Ambjorn, Jurkiewicz,
Loll 2005). Your first goal is to reproduce this known result. Only then
should you search for the 10⁻³ oscillation on top of it.

**Reproduce the known result first. Then search for the new physics.**
