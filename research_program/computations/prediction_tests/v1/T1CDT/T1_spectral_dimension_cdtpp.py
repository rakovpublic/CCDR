#!/usr/bin/env python3
"""
T1_spectral_dimension_cdtpp.py

Compute spectral dimension from CDT-plusplus output.

FIXES from previous version:
  - sigma_max scaled to triangulation size: sigma_max = N^(2/d) / 10
  - n_walks increased to 10000-50000 for measurable return probability
  - Uses HEAT KERNEL method as fallback when direct return counting fails
  - Proper sigma range: starts at sigma=2, ends at sigma ~ sqrt(N)
  - Reports P(sigma) values so you can verify they're nonzero

The problem was: sigma=500 in 3D gives P ~ 10^-6, so 500 walks never
see a return. Fix: use sigma = 2..50 with 20000+ walks per sigma.

Usage:
  python T1_spectral_dimension_cdtpp.py <data_dir_or_file> [dimension]
"""
import numpy as np
from collections import defaultdict
import glob
import os
import time
import sys


# ============================================================
# SECTION A: LOADING CDT-PLUSPLUS OUTPUT
# ============================================================


def load_triangulation_off(filename):
    """
    Load a triangulation from standard OFF or CDT-plusplus CGAL format.

    Standard OFF format:
        OFF
        n_vertices n_faces n_edges
        x y z
        3 v1 v2 v3

    CDT-plusplus CGAL format (.off files):
        dimension        (e.g. 3)
        n_vertices       (e.g. 383)
        x y z            (vertex coords, n_vertices lines)
        n_simplices      (e.g. 1400)
        v1 v2 v3 v4      (simplex vertex indices, n_simplices lines)
        ... (trailing data ignored)
    """
    adj = defaultdict(set)
    vertices = []
    simplices = []

    with open(filename, 'r') as f:
        first_line = f.readline().strip()

        if first_line == 'OFF':
            # Standard OFF format
            counts = f.readline().split()
            n_v, n_f = int(counts[0]), int(counts[1])
            for _ in range(n_v):
                coords = [float(x) for x in f.readline().split()]
                vertices.append(coords)
            for _ in range(n_f):
                parts = [int(x) for x in f.readline().split()]
                n_verts = parts[0]
                simplices.append(tuple(sorted(parts[1:1 + n_verts])))
        else:
            # CDT-plusplus CGAL format: first line is dimension
            dim = int(first_line)
            n_v = int(f.readline().strip())

            # Read vertices
            for _ in range(n_v):
                line = f.readline().strip()
                if not line:
                    continue
                coords = [float(x) for x in line.split()]
                vertices.append(coords)

            # Read number of simplices
            n_s = int(f.readline().strip())

            # Read simplices (each line: v1 v2 ... v_{dim+1})
            for _ in range(n_s):
                line = f.readline().strip()
                if not line:
                    continue
                parts = [int(x) for x in line.split()]
                if len(parts) >= dim + 1:
                    simplices.append(tuple(sorted(parts[:dim + 1])))

    if not simplices:
        raise ValueError(f"No simplices found in {filename}")

    # Build adjacency via shared sub-faces
    d = len(simplices[0]) - 1  # dimension
    face_map = defaultdict(list)
    for i, simplex in enumerate(simplices):
        for j in range(len(simplex)):
            sub_face = tuple(v for k, v in enumerate(simplex) if k != j)
            face_map[sub_face].append(i)

    for sub_face, simplex_ids in face_map.items():
        for a in range(len(simplex_ids)):
            for b in range(a + 1, len(simplex_ids)):
                adj[simplex_ids[a]].add(simplex_ids[b])
                adj[simplex_ids[b]].add(simplex_ids[a])

    n_simplices_count = len(simplices)
    print(f"Loaded: {n_simplices_count} simplices, "
          f"{len(vertices)} vertices, dim={d}")

    max_neighbours = max(len(v) for v in adj.values()) if adj else 0
    avg_neighbours = np.mean([len(v) for v in adj.values()]) if adj else 0
    print(f"Adjacency: max neighbours={max_neighbours}, "
          f"avg={avg_neighbours:.1f}")

    return adj, n_simplices_count


def load_adjacency_txt(filename):
    """Load adjacency from text file (C++ helper output)."""
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
    n = max(adj.keys()) + 1 if adj else 0
    print(f"Loaded adjacency: {n} simplices")
    return adj, n


def load_auto(filename):
    """Auto-detect and load."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.off':
        return load_triangulation_off(filename)
    elif ext in ('.txt', '.adj', '.dat'):
        return load_adjacency_txt(filename)
    else:
        # Try text adjacency format
        try:
            return load_adjacency_txt(filename)
        except Exception:
            pass
        # Try OFF / CGAL
        try:
            return load_triangulation_off(filename)
        except Exception:
            pass
        raise ValueError(f"Cannot load {filename}")


# ============================================================
# SECTION B: SPECTRAL DIMENSION -- FIXED VERSION
# ============================================================


def compute_return_probability(adj, n_simplices, sigma, n_walks):
    """
    Compute P(sigma) = return probability after sigma random walk steps.

    CRITICAL: n_walks must be large enough that P * n_walks >> 1.
    For P ~ sigma^{-d/2}: need n_walks >> sigma^{d/2}.
    """
    simplex_ids = list(adj.keys())
    N = len(simplex_ids)
    if N == 0:
        return 0.0, 0.0

    returns = 0
    valid = 0

    for _ in range(n_walks):
        s0 = simplex_ids[np.random.randint(N)]
        s = s0
        ok = True
        for _ in range(sigma):
            nbrs = list(adj[s])
            if not nbrs:
                ok = False
                break
            s = nbrs[np.random.randint(len(nbrs))]
        if ok:
            valid += 1
            if s == s0:
                returns += 1

    P = returns / valid if valid > 0 else 0.0
    # Binomial error: sigma_P = sqrt(P(1-P)/n)
    P_err = np.sqrt(P * (1 - P) / valid) if valid > 0 else 0.0
    return P, P_err


def determine_sigma_range(n_simplices, dimension=3):
    """
    Choose sigma range appropriate for the triangulation size.

    Rule of thumb:
    - sigma_min = 2 (minimum meaningful walk)
    - sigma_max ~ N^(2/d) / 5 (walk should explore a fraction of volume)
    - For N=10000, d=3: sigma_max ~ 10000^(0.67) / 5 ~ 93
    - But we also need P(sigma) to be measurable with ~20000 walks
    """
    sigma_max_geometric = int(n_simplices ** (2.0 / dimension) / 5)
    # Cap based on return probability being measurable with 20000 walks
    sigma_max_stats = int((2000) ** (2.0 / dimension) / (4 * np.pi))

    sigma_max = min(sigma_max_geometric, sigma_max_stats, 200)
    sigma_max = max(sigma_max, 10)  # at least 10

    sigma_min = 2
    n_points = 25

    sigma_values = np.unique(
        np.logspace(np.log10(sigma_min), np.log10(sigma_max),
                    n_points).astype(int)
    )
    sigma_values = sigma_values[sigma_values >= 2]

    return sigma_values, sigma_max


def determine_n_walks(sigma, dimension=3, target_returns=20):
    """
    Choose n_walks for a given sigma to get ~target_returns expected returns.

    P(sigma) ~ (4*pi*sigma)^(-d/2)
    n_walks = target_returns / P(sigma)
    """
    P_estimate = (4 * np.pi * sigma) ** (-dimension / 2.0)
    n_walks = int(target_returns / P_estimate)
    # Clamp to reasonable range
    n_walks = max(1000, min(n_walks, 200000))
    return n_walks


def measure_spectral_dimension(adj, n_simplices, dimension=3):
    """
    Measure d_s(sigma) with properly scaled parameters.
    """
    sigma_values, sigma_max = determine_sigma_range(n_simplices, dimension)

    print(f"\n  N_simplices = {n_simplices}")
    print(f"  sigma range: {sigma_values[0]} to {sigma_values[-1]} "
          f"({len(sigma_values)} points)")

    P_values = []
    P_errors = []

    for i, sigma in enumerate(sigma_values):
        n_walks = determine_n_walks(sigma, dimension, target_returns=30)
        t0 = time.time()
        P, P_err = compute_return_probability(adj, n_simplices, sigma, n_walks)
        elapsed = time.time() - t0

        P_values.append(P)
        P_errors.append(P_err)

        expected_returns = int(P * n_walks)
        status = "[OK]" if P > 0 else "[ZERO]"

        if (i + 1) % 5 == 0 or P == 0 or i == 0:
            print(f"  sigma={sigma:4d}: P={P:.6f} +/- {P_err:.6f}, "
                  f"n_walks={n_walks:6d}, returns~{expected_returns:3d}, "
                  f"time={elapsed:.1f}s {status}")

    P_values = np.array(P_values)
    P_errors = np.array(P_errors)

    # Check: did we get nonzero P at most points?
    nonzero = np.sum(P_values > 0)
    print(f"\n  Nonzero P(sigma): {nonzero}/{len(P_values)}")

    if nonzero < 5:
        print("  [FAIL] Too few nonzero P values. Cannot compute d_s.")
        print("  Try: increase n_walks, decrease sigma_max, "
              "or use Laplacian method.")
        return sigma_values, None, P_values, P_errors

    # Compute d_s = -2 d(ln P) / d(ln sigma) only where P > 0
    valid = P_values > 0
    log_sigma = np.log(sigma_values[valid].astype(float))
    log_P = np.log(P_values[valid])

    d_s = np.zeros_like(log_P)
    for i in range(1, len(log_P) - 1):
        d_s[i] = -2 * (log_P[i + 1] - log_P[i - 1]) / \
                      (log_sigma[i + 1] - log_sigma[i - 1])
    d_s[0] = d_s[1]
    d_s[-1] = d_s[-2]

    # Error on d_s from error propagation
    d_s_err = np.zeros_like(d_s)
    P_err_valid = P_errors[valid]
    for i in range(1, len(d_s) - 1):
        dlogs = log_sigma[i + 1] - log_sigma[i - 1]
        err2 = (P_err_valid[i + 1] / P_values[valid][i + 1])**2 + \
               (P_err_valid[i - 1] / P_values[valid][i - 1])**2
        d_s_err[i] = 2 * np.sqrt(err2) / dlogs if dlogs > 0 else 0

    # Report d_s at small and large sigma
    if len(d_s) >= 5:
        d_s_small = np.mean(d_s[:3])
        d_s_large = np.mean(d_s[-5:])
        d_s_large_err = np.sqrt(np.mean(d_s_err[-5:]**2)) / np.sqrt(5)
        print(f"\n  d_s at small sigma (UV): {d_s_small:.2f}")
        print(f"  d_s at large sigma (IR): {d_s_large:.2f} "
              f"+/- {d_s_large_err:.2f}")
        print(f"  Expected (IR): {dimension}.0")
        print(f"  Expected (UV): ~2.0 (CDT dimensional reduction)")

        if abs(d_s_large - dimension) < max(1.0, 2 * d_s_large_err):
            print(f"  [OK] d_s -> {dimension} at large sigma: CONSISTENT")
        else:
            print(f"  [WARN] d_s -> {dimension} at large sigma: INCONSISTENT")
            print(f"    (may need more walks or larger sigma)")

    # Full arrays with NaN for invalid points
    d_s_full = np.full(len(sigma_values), np.nan)
    d_s_err_full = np.full(len(sigma_values), np.nan)
    d_s_full[valid] = d_s
    d_s_err_full[valid] = d_s_err

    return sigma_values, d_s_full, P_values, P_errors


# ============================================================
# SECTION C: VALIDATION
# ============================================================


def validate_c_phase(adj, n_simplices, dimension=3):
    """Validate C-phase with properly scaled walks."""
    print("\n" + "=" * 60)
    print("C-PHASE VALIDATION (FIXED PARAMETERS)")
    print("=" * 60)

    # Check 1: connectivity
    degrees = [len(adj[s]) for s in adj]
    mean_deg = np.mean(degrees)
    std_deg = np.std(degrees)
    print(f"\n  Degree: mean={mean_deg:.1f}, std={std_deg:.1f}, "
          f"range=[{min(degrees)}, {max(degrees)}]")

    if mean_deg == dimension + 1 and std_deg == 0:
        print(f"  Note: ALL simplices have exactly {int(mean_deg)} neighbours.")
        print(f"  This is expected for a closed {dimension}D manifold")
        print(f"  (each {dimension}-simplex has {dimension+1} faces).")
        print(f"  [OK] Degree check PASSED")
    elif mean_deg < 2:
        print(f"  [FAIL] Very low connectivity -- triangulation may be "
              f"degenerate")
        return False

    # Check 2: giant component
    visited = set()
    queue = [list(adj.keys())[0]]
    while queue:
        s = queue.pop()
        if s in visited:
            continue
        visited.add(s)
        for nb in adj[s]:
            if nb not in visited:
                queue.append(nb)
    frac = len(visited) / len(adj)
    print(f"  Giant component: {frac:.4f}")
    if frac < 0.95:
        print(f"  [FAIL] Fragmented!")
        return False
    print(f"  [OK] Fully connected")

    # Check 3: spectral dimension with CORRECT walk parameters
    print(f"\n  Quick d_s check (sigma=5 and sigma=20)...")
    n_walks_small = determine_n_walks(5, dimension, target_returns=50)
    n_walks_large = determine_n_walks(20, dimension, target_returns=50)

    P5, _ = compute_return_probability(adj, n_simplices, 5, n_walks_small)
    P20, _ = compute_return_probability(adj, n_simplices, 20, n_walks_large)

    print(f"  P(sigma=5) = {P5:.6f} (n_walks={n_walks_small})")
    print(f"  P(sigma=20) = {P20:.6f} (n_walks={n_walks_large})")

    if P5 > 0 and P20 > 0:
        d_s_estimate = -2 * (np.log(P20) - np.log(P5)) / \
                            (np.log(20) - np.log(5))
        print(f"  d_s estimate: {d_s_estimate:.2f} (expected: ~{dimension})")

        if abs(d_s_estimate - dimension) < 1.5:
            print(f"  [OK] d_s ~ {d_s_estimate:.1f} -- plausible C-phase")
            return True
        elif d_s_estimate > 1.5:
            print(f"  [WARN] d_s = {d_s_estimate:.1f} -- not exactly "
                  f"{dimension} but nonzero")
            print(f"    May need finer sigma sampling or more walks")
            return True  # proceed cautiously
        else:
            print(f"  [FAIL] d_s too low -- not C-phase")
            return False
    elif P5 > 0:
        print(f"  P(20) = 0 but P(5) > 0 -- try smaller sigma_max")
        return True  # proceed with smaller sigma range
    else:
        print(f"  [FAIL] Both P(5) and P(20) = 0 -- walks never return")
        print(f"    Triangulation may be too large for random walk method.")
        print(f"    Try the Laplacian eigenvalue method instead.")
        return False


# ============================================================
# SECTION D: LAPLACIAN EIGENVALUE METHOD (FALLBACK)
# ============================================================


def spectral_dimension_laplacian(adj, n_simplices, dimension=3,
                                 n_eigenvalues=200):
    """
    Compute spectral dimension from the Laplacian eigenvalues.

    This avoids the random walk return probability problem entirely.

    The graph Laplacian L has eigenvalues 0 = lam_0 <= lam_1 <= ... <= lam_N.
    The heat kernel trace is: K(t) = sum exp(-lam_i t)
    The spectral dimension: d_s(t) = -2 d ln K(t) / d ln t

    For a d-dimensional manifold: K(t) ~ t^(-d/2) at small t.

    This method works even when the triangulation is very large
    (where random walk P -> 0).
    """
    try:
        from scipy import sparse
        from scipy.sparse.linalg import eigsh
    except ImportError:
        print("  Need scipy for Laplacian method: pip install scipy")
        return None, None, None

    print(f"\n  Computing Laplacian eigenvalues (n={n_eigenvalues})...")

    # Build sparse Laplacian matrix
    N = max(adj.keys()) + 1
    rows, cols, vals = [], [], []
    for i in adj:
        deg = len(adj[i])
        rows.append(i)
        cols.append(i)
        vals.append(float(deg))  # diagonal = degree
        for j in adj[i]:
            rows.append(i)
            cols.append(j)
            vals.append(-1.0)  # off-diagonal = -1

    L = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

    # Compute smallest eigenvalues
    n_eig = min(n_eigenvalues, N - 2)
    print(f"  Matrix size: {N}x{N}, computing {n_eig} smallest "
          f"eigenvalues...")
    t0 = time.time()
    try:
        eigenvalues, _ = eigsh(L, k=n_eig, which='SM', tol=1e-6)
        eigenvalues = np.sort(np.real(eigenvalues))
        # Remove the zero eigenvalue
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s. Got {len(eigenvalues)} nonzero "
              f"eigenvalues.")
    except Exception as e:
        print(f"  Eigenvalue computation failed: {e}")
        return None, None, None

    # Compute heat kernel trace K(t) = sum exp(-lam_i t)
    t_values = np.logspace(-2, 2, 50)
    K_values = np.array([np.sum(np.exp(-eigenvalues * t)) for t in t_values])

    # Spectral dimension d_s(t) = -2 d ln K / d ln t
    log_t = np.log(t_values)
    log_K = np.log(K_values + 1e-30)
    d_s = np.zeros_like(log_K)
    for i in range(1, len(log_K) - 1):
        d_s[i] = -2 * (log_K[i + 1] - log_K[i - 1]) / \
                      (log_t[i + 1] - log_t[i - 1])
    d_s[0] = d_s[1]
    d_s[-1] = d_s[-2]

    # Report
    d_s_small_t = np.mean(d_s[5:10])  # small t = UV
    d_s_large_t = np.mean(d_s[-10:-5])  # large t = IR
    print(f"  d_s at small t (UV): {d_s_small_t:.2f} (expected: ~2)")
    print(f"  d_s at large t (IR): {d_s_large_t:.2f} (expected: "
          f"~{dimension})")

    return t_values, d_s, K_values


# ============================================================
# SECTION E: OSCILLATION SEARCH
# ============================================================


def search_oscillations(sigma_values, d_s_values, dimension=3):
    """
    Search for oscillations in d_s(sigma) around the smooth trend.
    """
    valid = (np.isfinite(d_s_values) & (d_s_values > 0)
             & (d_s_values < 2 * dimension))
    if np.sum(valid) < 10:
        print("Too few valid d_s points for oscillation search")
        return None, None

    log_s = np.log(sigma_values[valid].astype(float))
    ds = d_s_values[valid]

    try:
        coeffs = np.polyfit(log_s, ds, 3)
        ds_smooth = np.polyval(coeffs, log_s)
    except Exception:
        ds_smooth = np.mean(ds) * np.ones_like(ds)

    residuals = ds - ds_smooth
    delta_d = np.sqrt(np.mean(residuals**2))

    return delta_d, residuals


# ============================================================
# SECTION F: MAIN
# ============================================================


def main():
    if len(sys.argv) < 2:
        print("Usage: python T1_spectral_dimension_cdtpp.py "
              "<data_dir_or_file> [dim]")
        print()
        print("  data_dir_or_file: CDT-plusplus output "
              "(directory or single file)")
        print("  dim: spatial dimension (3 or 4, default 3)")
        print()
        print("The script will:")
        print("  1. Load the triangulation (CGAL or OFF format)")
        print("  2. Validate C-phase (with correctly scaled walks)")
        print("  3. Measure d_s(sigma) using random walks "
              "OR Laplacian eigenvalues")
        print("  4. Search for oscillations")
        sys.exit(1)

    target = sys.argv[1]
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    # Find files
    if os.path.isfile(target):
        files = [target]
    else:
        files = []
        for ext in ['*.off', '*.txt', '*.adj', '*.dat', 'checkpoint_*']:
            files.extend(sorted(glob.glob(os.path.join(target, ext))))
            files.extend(sorted(glob.glob(
                os.path.join(target, '**', ext), recursive=True)))
        files = sorted(set(files))

    if not files:
        print(f"No files found in {target}")
        sys.exit(1)

    print(f"Found {len(files)} file(s)")

    # Load first file
    filepath = files[0]
    print(f"\nLoading: {filepath}")
    try:
        adj, n_sim = load_auto(filepath)
        print(f"Loaded: {n_sim} simplices, {len(adj)} in adjacency")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Validate
    is_valid = validate_c_phase(adj, n_sim, dim)

    if not is_valid:
        print("\nC-phase validation failed with random walks.")
        print("Trying Laplacian eigenvalue method (more robust)...")
        t_vals, d_s_lap, K_vals = spectral_dimension_laplacian(
            adj, n_sim, dim)
        if d_s_lap is not None:
            print("\nLaplacian method succeeded. Results above.")
        sys.exit(0)

    # Full measurement
    print("\n" + "=" * 60)
    print("FULL SPECTRAL DIMENSION MEASUREMENT")
    print("=" * 60)

    sigmas, d_s, P_vals, P_errs = measure_spectral_dimension(
        adj, n_sim, dim)

    if d_s is None:
        print("\nRandom walk method failed. Falling back to Laplacian...")
        t_vals, d_s_lap, K_vals = spectral_dimension_laplacian(
            adj, n_sim, dim)
    else:
        # Search for oscillations
        valid = np.isfinite(d_s)
        if np.sum(valid) >= 10:
            log_s = np.log(sigmas[valid].astype(float))
            ds_v = d_s[valid]
            try:
                coeffs = np.polyfit(log_s, ds_v, 3)
                ds_smooth = np.polyval(coeffs, log_s)
                residuals = ds_v - ds_smooth
                delta_d = np.sqrt(np.mean(residuals**2))
                print(f"\n  Oscillation search:")
                print(f"  Residual RMS (delta_d): {delta_d:.6f}")
                print(f"  Target: delta_d = nu ~ 0.001")
            except Exception:
                print("  Could not fit smooth trend for oscillation search")

    # Save results
    try:
        save_data = {'n_simplices': n_sim, 'dimension': dim}
        if d_s is not None:
            save_data.update(sigma=sigmas, d_s=d_s,
                             P=P_vals, P_err=P_errs)
        np.savez('T1_cdtpp_results.npz', **save_data)
        print("\nResults saved: T1_cdtpp_results.npz")
    except Exception:
        pass

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 9))

        if d_s is not None:
            valid = np.isfinite(d_s)
            axes[0].plot(sigmas[valid], d_s[valid], 'o-', markersize=4)
            axes[0].axhline(dim, color='green', ls='--',
                            label=f'd_s = {dim}')
            axes[0].axhline(2, color='orange', ls=':',
                            label='d_s = 2 (UV)')
            axes[0].set_xlabel('sigma')
            axes[0].set_ylabel('d_s(sigma)')
            axes[0].set_xscale('log')
            axes[0].legend()
            axes[0].set_title(f'Spectral dimension (N = {n_sim})')

            axes[1].semilogy(sigmas, P_vals, 'o-', markersize=4)
            axes[1].set_xlabel('sigma')
            axes[1].set_ylabel('P(sigma)')
            axes[1].set_xscale('log')
            axes[1].set_title('Return probability')

        plt.tight_layout()
        plt.savefig('T1_cdtpp_spectral_dimension.png', dpi=150)
        print(f"Plot saved: T1_cdtpp_spectral_dimension.png")
    except ImportError:
        pass


if __name__ == '__main__':
    main()
