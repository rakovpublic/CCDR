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
            face_verts = tuple(sorted(parts[1:1 + n_verts]))
            faces.append(face_verts)

    # Build adjacency: two simplices are adjacent if they share d vertices
    # (For tetrahedra in 3D: share a triangular face = 3 vertices)
    # This is O(n^2) for simplicity; for large N use a hash map

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


def load_triangulation_cgal(filename):
    """
    Load a CGAL serialised triangulation.

    This is more complex -- CGAL uses its own binary/text format.

    You may need to write a small C++ program that:
    1. Loads the CGAL triangulation
    2. Outputs the adjacency list as a text file
    3. Then read that text file in Python

    See export_adjacency.cpp for the C++ helper.
    """
    raise NotImplementedError(
        "CGAL format requires a C++ helper. See export_adjacency.cpp.")


def load_triangulation_auto(filename):
    """Try to auto-detect format and load."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.off':
        return load_triangulation_off(filename)
    elif ext == '.txt' or ext == '.adj':
        return load_adjacency_txt(filename)
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
        except Exception:
            pass

        raise ValueError(f"Cannot load {filename} -- unknown format. "
                         f"Use the C++ helper (export_adjacency.cpp) "
                         f"to export adjacency.")


# ============================================================
# SECTION B: SPECTRAL DIMENSION MEASUREMENT
# ============================================================


def random_walk_return_probability(adj, n_simplices, sigma, n_walks=1000):
    """
    Compute the return probability P(sigma) by random walks on the dual graph.

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
    Compute d_s(sigma) = -2 d ln P / d ln sigma.

    Returns: sigma_values, d_s_values, P_values
    """
    sigma_values = np.unique(
        np.logspace(np.log10(sigma_min), np.log10(sigma_max),
                    n_points).astype(int)
    )
    sigma_values = sigma_values[sigma_values >= 2]

    print(f"Computing P(sigma) for {len(sigma_values)} sigma values "
          f"({sigma_values[0]} to {sigma_values[-1]}), "
          f"{n_walks} walks each...")

    P_values = []
    for i, sigma in enumerate(sigma_values):
        t0 = time.time()
        P = random_walk_return_probability(adj, n_simplices, sigma, n_walks)
        P_values.append(max(P, 1e-10))  # avoid log(0)
        elapsed = time.time() - t0
        if (i + 1) % 5 == 0:
            print(f"  sigma={sigma:5d}: P={P:.6f}, "
                  f"time={elapsed:.1f}s")

    P_values = np.array(P_values)
    log_sigma = np.log(sigma_values.astype(float))
    log_P = np.log(P_values)

    # Numerical derivative using central differences
    d_s = np.zeros_like(log_P)
    for i in range(1, len(log_P) - 1):
        d_s[i] = -2 * (log_P[i + 1] - log_P[i - 1]) / \
                      (log_sigma[i + 1] - log_sigma[i - 1])
    d_s[0] = d_s[1]
    d_s[-1] = d_s[-2]

    return sigma_values, d_s, P_values


# ============================================================
# SECTION C: VALIDATION
# ============================================================


def validate_c_phase(adj, n_simplices, dimension=3):
    """
    Validate that the triangulation is in the C-phase.

    Check 1: spectral dimension -> d at large sigma
    Check 2: volume profile is de Sitter (cos^3)
    Check 3: no pathological connectivity (blob or crumpled)
    """
    print("\n" + "=" * 60)
    print("C-PHASE VALIDATION")
    print("=" * 60)

    # Check 1: Quick spectral dimension at large sigma
    print("\nCheck 1: Spectral dimension at large sigma...")
    sigma_large = min(n_simplices // 2, 500)
    P_large = random_walk_return_probability(adj, n_simplices,
                                             sigma_large, n_walks=500)
    # For a d-dimensional space: P(sigma) ~ sigma^(-d/2)
    # So at large sigma, d_s ~ d
    P_small = random_walk_return_probability(adj, n_simplices,
                                             sigma_large // 5, n_walks=500)
    if P_large > 0 and P_small > 0:
        d_s_estimate = -2 * (np.log(P_large) - np.log(P_small)) / \
                            (np.log(sigma_large) - np.log(sigma_large // 5))
    else:
        d_s_estimate = 0

    print(f"  d_s(sigma={sigma_large}) ~ {d_s_estimate:.2f}")
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
    print(f"\n  Check 1 (d_s -> {dimension}): {'PASS' if check1_pass else 'FAIL'}")
    print(f"  Check 2 (connectivity):   {'PASS' if check2_pass else 'FAIL'}")
    print(f"  Check 3 (giant component): {'PASS' if check3_pass else 'FAIL'}")
    print(f"  Overall: {'[PASS] C-PHASE VALIDATED' if all_pass else '[FAIL] NOT C-PHASE'}")

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
    Search for oscillations in d_s(sigma) around the smooth trend.

    Method:
    1. Fit a smooth curve (polynomial in log sigma) to d_s(sigma)
    2. Compute residuals
    3. Measure RMS of residuals = amplitude of oscillation
    4. Compare with target delta_d ~ nu ~ 10^-3
    """
    valid = (np.isfinite(d_s_values) & (d_s_values > 0)
             & (d_s_values < 2 * dimension))
    if np.sum(valid) < 10:
        print("Too few valid d_s points for oscillation search")
        return None, None

    log_s = np.log(sigma_values[valid].astype(float))
    ds = d_s_values[valid]

    # Fit polynomial of degree 3 to capture the smooth trend
    # (d_s goes from ~2 at small sigma to ~d at large sigma)
    try:
        coeffs = np.polyfit(log_s, ds, 3)
        ds_smooth = np.polyval(coeffs, log_s)
    except Exception:
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
    print(f"Target: delta_d ~ nu ~ 10^-3")
    print(f"Data directory: {data_dir}")
    print("=" * 70)

    # Find checkpoint files
    patterns = ['*.off', '*.dat', '*.tri', '*.cgal', '*.txt', '*.adj',
                'checkpoint_*']
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(os.path.join(data_dir, pat))))
        files.extend(sorted(glob.glob(os.path.join(data_dir, '**', pat),
                                      recursive=True)))
    files = sorted(set(files))

    if not files:
        print(f"\nNo triangulation files found in {data_dir}")
        print("Check the output format of your CDT-plusplus version.")
        print("You may need to use the C++ helper (export_adjacency.cpp) "
              "to export.")
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
                print("  [FAIL] First config is not C-phase. Check parameters.")
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
            print(f"  [OK] d_s at large sigma: {ds[-5:].mean():.3f}")

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
    print(f"\nEnsemble <d_s> at large sigma: {ds_ensemble:.3f} "
          f"(expected: {dimension})")

    if abs(ds_ensemble - dimension) > 1.0:
        print(f"[FAIL] VALIDATION FAILED: d_s = {ds_ensemble:.2f} != {dimension}")
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
        print(f"<d_s> at large sigma: {ds_ensemble:.3f}")
        print(f"Residual RMS (delta_d): {delta_d:.6f}")
        print(f"Target delta_d = nu: 0.001")
        print(f"Mean statistical error: {mean_err:.6f}")
        print(f"Significance: {significance:.1f} sigma")

        if abs(ds_ensemble - dimension) > 1.0:
            print(f"\n[WARN] d_s validation FAILED. Result unreliable.")
        elif significance > 3 and delta_d < 0.1:
            print(f"\n[PASS] OSCILLATION DETECTED at {significance:.1f} sigma")
            if abs(np.log10(delta_d) - (-3)) < 0.5:
                print(f"  delta_d = {delta_d:.2e} ~ 10^-3 -- CONSISTENT with nu!")
            else:
                print(f"  delta_d = {delta_d:.2e} != 10^-3")
        else:
            print(f"\n[INSUFFICIENT] No significant oscillation at current "
                  f"sensitivity")
            if mean_err > 0 and significance > 0:
                n_needed = int(valid_configs * (3 / significance)**2)
                print(f"  Need ~{n_needed} configs for 3 sigma")

        # Save
        np.savez('T1_cdtpp_results.npz',
                 sigma=sigma_ref, ds_mean=ds_mean, ds_std=ds_std,
                 delta_d=delta_d, significance=significance,
                 ds_ensemble=ds_ensemble, n_configs=valid_configs)

        # Plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(12, 9))

            valid = np.isfinite(ds_mean)
            axes[0].errorbar(sigma_ref[valid], ds_mean[valid],
                             yerr=ds_std[valid], fmt='o', markersize=3,
                             capsize=2,
                             label=f'CDT-plusplus ({valid_configs} configs)')
            axes[0].axhline(dimension, color='green', ls='--', alpha=0.5,
                            label=f'd_s = {dimension}')
            axes[0].axhline(2, color='orange', ls=':', alpha=0.5,
                            label='d_s = 2')
            axes[0].set_xlabel('sigma (diffusion steps)', fontsize=13)
            axes[0].set_ylabel('d_s(sigma)', fontsize=13)
            axes[0].set_xscale('log')
            axes[0].set_title('T1: Spectral Dimension from CDT-plusplus',
                              fontsize=15)
            axes[0].legend(fontsize=11)

            if residuals is not None:
                log_s = np.log(sigma_ref[valid].astype(float))
                axes[1].plot(log_s[:len(residuals)], residuals, 'o',
                             markersize=3)
                axes[1].axhline(0, color='k', ls='--')
                axes[1].axhline(1e-3, color='r', ls=':',
                                label='nu = 1e-3')
                axes[1].axhline(-1e-3, color='r', ls=':')
                axes[1].set_xlabel('ln sigma', fontsize=13)
                axes[1].set_ylabel('Residual', fontsize=13)
                axes[1].set_title(
                    f'Oscillatory residual: delta_d = {delta_d:.6f}, '
                    f'{significance:.1f} sigma', fontsize=14)
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
        print("Usage: python T1_spectral_dimension_cdtpp.py "
              "<data_dir> [dimension]")
        print()
        print("  data_dir:  directory with CDT-plusplus checkpoint files")
        print("  dimension: 3 or 4 (default: 3)")
        print()
        print("Example:")
        print("  python T1_spectral_dimension_cdtpp.py "
              "~/cdt_data/run_3d_k2p0 3")
        sys.exit(1)

    data_dir = sys.argv[1]
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    run_T1_ensemble(data_dir, dimension=dim)
