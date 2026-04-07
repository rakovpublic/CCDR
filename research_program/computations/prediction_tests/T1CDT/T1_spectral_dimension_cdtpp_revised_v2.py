#!/usr/bin/env python3
"""
T1_spectral_dimension_cdtpp_revised.py

Revised spectral-dimension pipeline for CDT-plusplus output.

What this revision changes relative to the uploaded script:
  1. Uses EVEN diffusion times by default to avoid bipartite/parity artifacts
     that can severely bias return probabilities on simplex-dual graphs.
  2. Uses a LAZY random walk (stay-put probability 1/2) by default; this also
     suppresses bipartite oscillation artifacts and makes RW and heat-kernel
     estimators more comparable.
  3. Adds exact / semi-exact validation on known graphs before trusting CDT
     output. If the pipeline fails on a cubic torus graph, the CDT result is
     rejected.
  4. Uses the same operator family for both methods:
       - random walk transition matrix T
       - random-walk Laplacian L_rw = I - T
     instead of mixing a simple walk with the combinatorial Laplacian.
  5. Includes the zero mode in the heat trace and reports a normalized trace.
  6. Refuses to claim success when the estimators disagree strongly.
  7. Adds a null / permutation control by rewiring the graph while preserving
     degree approximately, to estimate a noise floor for residual oscillations.
  8. Searches oscillations only after a valid spectral-dimension curve is found.

Usage:
  python3 T1_spectral_dimension_cdtpp_revised.py <data_dir_or_file> [dimension]

Outputs:
  - T1_cdtpp_revised_results.npz
  - T1_cdtpp_revised_spectral_dimension.png
"""
from __future__ import annotations

import glob
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


Adjacency = Dict[int, Set[int]]


# ============================================================
# SECTION A: LOADING CDT-PLUSPLUS OUTPUT
# ============================================================


def load_triangulation_off(filename: str) -> Tuple[Adjacency, int]:
    """Load standard OFF or CDT-plusplus CGAL-like OFF and build simplex adjacency."""
    adj: Adjacency = defaultdict(set)
    vertices: List[List[float]] = []
    simplices: List[Tuple[int, ...]] = []

    with open(filename, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

        if first_line == "OFF":
            counts = f.readline().split()
            if len(counts) < 2:
                raise ValueError("Malformed OFF counts line")
            n_v, n_f = int(counts[0]), int(counts[1])
            for _ in range(n_v):
                coords = [float(x) for x in f.readline().split()]
                vertices.append(coords)
            for _ in range(n_f):
                parts = [int(x) for x in f.readline().split()]
                n_verts = parts[0]
                simplices.append(tuple(sorted(parts[1:1 + n_verts])))
        else:
            dim = int(first_line)
            n_v = int(f.readline().strip())
            for _ in range(n_v):
                line = f.readline().strip()
                if line:
                    vertices.append([float(x) for x in line.split()])
            n_s = int(f.readline().strip())
            for _ in range(n_s):
                line = f.readline().strip()
                if not line:
                    continue
                parts = [int(x) for x in line.split()]
                if len(parts) >= dim + 1:
                    simplices.append(tuple(sorted(parts[: dim + 1])))

    if not simplices:
        raise ValueError(f"No simplices found in {filename}")

    d = len(simplices[0]) - 1
    face_map: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for i, simplex in enumerate(simplices):
        for j in range(len(simplex)):
            sub_face = tuple(v for k, v in enumerate(simplex) if k != j)
            face_map[sub_face].append(i)

    for simplex_ids in face_map.values():
        for a in range(len(simplex_ids)):
            for b in range(a + 1, len(simplex_ids)):
                i, j = simplex_ids[a], simplex_ids[b]
                adj[i].add(j)
                adj[j].add(i)

    for i in range(len(simplices)):
        adj[i]  # ensure all simplices are present, even isolated

    n_simplices_count = len(simplices)
    print(f"Loaded: {n_simplices_count} simplices, {len(vertices)} vertices, dim={d}")
    degrees = [len(v) for v in adj.values()]
    print(f"Adjacency: max neighbours={max(degrees) if degrees else 0}, avg={np.mean(degrees) if degrees else 0:.1f}")
    return adj, n_simplices_count



def load_adjacency_txt(filename: str) -> Tuple[Adjacency, int]:
    """Load adjacency from text file (helper output)."""
    adj: Adjacency = defaultdict(set)
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = [int(x) for x in line.split()]
            if len(parts) >= 2:
                cell_id = parts[0]
                for nb in parts[1:]:
                    adj[cell_id].add(nb)
                    adj[nb].add(cell_id)
    n = max(adj.keys()) + 1 if adj else 0
    for i in range(n):
        adj[i]
    print(f"Loaded adjacency: {n} simplices")
    return adj, n



def load_auto(filename: str) -> Tuple[Adjacency, int]:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".off":
        return load_triangulation_off(filename)
    if ext in (".txt", ".adj", ".dat"):
        return load_adjacency_txt(filename)
    for loader in (load_adjacency_txt, load_triangulation_off):
        try:
            return loader(filename)
        except Exception:
            pass
    raise ValueError(f"Cannot load {filename}")


# ============================================================
# SECTION B: GRAPH UTILITIES / OPERATORS
# ============================================================


def degree_stats(adj: Adjacency) -> Tuple[float, float, int, int]:
    degrees = [len(adj[i]) for i in adj]
    return float(np.mean(degrees)), float(np.std(degrees)), int(min(degrees)), int(max(degrees))



def giant_component_fraction(adj: Adjacency) -> float:
    if not adj:
        return 0.0
    start = next(iter(adj))
    seen: Set[int] = set()
    q: deque[int] = deque([start])
    while q:
        x = q.popleft()
        if x in seen:
            continue
        seen.add(x)
        for y in adj[x]:
            if y not in seen:
                q.append(y)
    return len(seen) / len(adj)



def is_bipartite(adj: Adjacency) -> bool:
    color: Dict[int, int] = {}
    for start in adj:
        if start in color:
            continue
        color[start] = 0
        q: deque[int] = deque([start])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in color:
                    color[y] = 1 - color[x]
                    q.append(y)
                elif color[y] == color[x]:
                    return False
    return True



def adjacency_to_sparse(adj: Adjacency) -> "sparse.csr_matrix":
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for sparse operators")
    n = max(adj.keys()) + 1 if adj else 0
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for i in range(n):
        for j in adj[i]:
            rows.append(i)
            cols.append(j)
            vals.append(1.0)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)



def random_walk_operator(adj: Adjacency, lazy: bool = True) -> "sparse.csr_matrix":
    """Return row-stochastic transition matrix T."""
    A = adjacency_to_sparse(adj).tocsr()
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    if np.any(deg <= 0):
        raise ValueError("Graph contains isolated vertices; random walk undefined")
    inv_deg = sparse.diags(1.0 / deg)
    T = inv_deg @ A
    if lazy:
        I = sparse.identity(n, format="csr", dtype=float)
        T = 0.5 * I + 0.5 * T
    return T.tocsr()



def random_walk_laplacian(adj: Adjacency, lazy: bool = True) -> "sparse.csr_matrix":
    T = random_walk_operator(adj, lazy=lazy)
    I = sparse.identity(T.shape[0], format="csr", dtype=float)
    return (I - T).tocsr()


# ============================================================
# SECTION C: SANITY TESTS ON KNOWN GRAPHS
# ============================================================


def build_periodic_cubic_lattice(L: int = 8, dim: int = 3) -> Adjacency:
    """Build a d-dimensional periodic cubic lattice."""
    if dim != 3:
        raise ValueError("Only dim=3 implemented for cubic lattice sanity test")
    adj: Adjacency = defaultdict(set)

    def idx(x: int, y: int, z: int) -> int:
        return x + L * (y + L * z)

    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = idx(x, y, z)
                for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    j = idx((x + dx) % L, (y + dy) % L, (z + dz) % L)
                    adj[i].add(j)
    return adj


@dataclass
class SanityResult:
    passed: bool
    details: str



def quick_rw_dimension(adj: Adjacency, sigmas: Sequence[int], rng: np.random.Generator, lazy: bool = True,
                       starts_per_sigma: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Monte Carlo return probability using a lazy random walk on adjacency."""
    nodes = np.array(sorted(adj.keys()), dtype=int)
    probs: List[float] = []
    errs: List[float] = []
    for sigma in sigmas:
        starts = rng.choice(nodes, size=starts_per_sigma, replace=True)
        cur = starts.copy()
        for _ in range(sigma):
            if lazy:
                stay = rng.random(starts_per_sigma) < 0.5
            else:
                stay = np.zeros(starts_per_sigma, dtype=bool)
            move_idx = np.where(~stay)[0]
            for idx in move_idx:
                nbrs = tuple(adj[int(cur[idx])])
                cur[idx] = nbrs[rng.integers(len(nbrs))]
        returned = (cur == starts)
        p = returned.mean()
        e = np.sqrt(max(p * (1 - p) / starts_per_sigma, 0.0))
        probs.append(float(p))
        errs.append(float(e))
    return np.asarray(probs), np.asarray(errs)



def estimate_ds_from_curve(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    log_t = np.log(times.astype(float))
    log_v = np.log(np.clip(values.astype(float), 1e-300, None))
    ds = np.full_like(log_v, np.nan, dtype=float)
    for i in range(1, len(log_t) - 1):
        ds[i] = -2.0 * (log_v[i + 1] - log_v[i - 1]) / (log_t[i + 1] - log_t[i - 1])
    if len(ds) >= 3:
        ds[0] = ds[1]
        ds[-1] = ds[-2]
    return ds


def heat_trace_from_eigenvalues(vals: np.ndarray, t_vals: np.ndarray, normalize: str = "mean") -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    if normalize not in {"mean", "sum"}:
        raise ValueError("normalize must be 'mean' or 'sum'")
    K = np.array([np.sum(np.exp(-vals * t)) for t in t_vals], dtype=float)
    if normalize == "mean":
        K = K / max(len(vals), 1)
    return K


def full_spectrum_heat_trace(adj: Adjacency, lazy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Exact eigenvalue spectrum for small graphs; suitable for sanity tests."""
    Lrw = random_walk_laplacian(adj, lazy=lazy)
    dense = Lrw.toarray()
    vals = np.linalg.eigvalsh(dense)
    vals = np.sort(np.real(vals))
    return dense, vals



def run_sanity_suite(rng: np.random.Generator) -> SanityResult:
    print("\n" + "=" * 60)
    print("SANITY SUITE")
    print("=" * 60)
    lattice = build_periodic_cubic_lattice(L=8, dim=3)
    sigmas = np.array([2, 4, 6, 8, 12, 16, 24], dtype=int)
    P, _ = quick_rw_dimension(lattice, sigmas, rng=rng, lazy=True, starts_per_sigma=4000)
    ds = estimate_ds_from_curve(sigmas, P)
    core = ds[2:-2]
    core_mean = float(np.nanmean(core))
    print(f"  3D cubic torus random-walk sanity: mean d_s ≈ {core_mean:.2f} (expected ~3)")
    if not (2.2 <= core_mean <= 3.8):
        return SanityResult(False, f"sanity RW failed: cubic torus gave d_s≈{core_mean:.2f}")

    if SCIPY_AVAILABLE:
        try:
            _, ds_lap, _ = spectral_dimension_laplacian(lattice, len(lattice), dimension=3,
                                                        n_eigenvalues=min(512, len(lattice) - 2),
                                                        lazy=True, quiet=True)
            if ds_lap is None:
                return SanityResult(False, "sanity Laplacian failed to compute")
            lap_core = float(np.nanmean(ds_lap[8:18]))
            print(f"  3D cubic torus Laplacian sanity: mean d_s ≈ {lap_core:.2f} (expected ~3)")
            if not (2.0 <= lap_core <= 3.8):
                return SanityResult(False, f"sanity Laplacian failed: cubic torus gave d_s≈{lap_core:.2f}")
        except Exception as exc:
            return SanityResult(False, f"sanity Laplacian exception: {exc}")

    print("  [OK] Sanity suite passed")
    return SanityResult(True, "sanity suite passed")


# ============================================================
# SECTION D: RANDOM WALK METHOD
# ============================================================


def choose_even_sigmas(n_simplices: int, dimension: int = 3) -> np.ndarray:
    sigma_min = 2
    sigma_max_geom = int(max(12, n_simplices ** (2.0 / max(dimension, 1)) / 4.0))
    sigma_max = min(160, sigma_max_geom)
    raw = np.unique(np.logspace(np.log10(sigma_min), np.log10(sigma_max), 24).astype(int))
    even = np.unique(2 * np.ceil(raw / 2).astype(int))
    even = even[even >= 2]
    return even



def determine_n_walks(sigma: int, dimension: int = 3, target_returns: int = 200, lazy: bool = True) -> int:
    # Conservative heuristic; lazy walk increases return probability modestly.
    pref = 2.0 if lazy else 1.0
    p_est = pref * (4.0 * np.pi * max(sigma, 1)) ** (-dimension / 2.0)
    n_walks = int(np.ceil(target_returns / max(p_est, 1e-8)))
    return int(max(4000, min(n_walks, 300000)))



def compute_return_probability(adj: Adjacency, sigma: int, n_walks: int, rng: np.random.Generator,
                               lazy: bool = True) -> Tuple[float, float]:
    nodes = np.array(sorted(adj.keys()), dtype=int)
    if len(nodes) == 0:
        return 0.0, 0.0
    starts = rng.choice(nodes, size=n_walks, replace=True)
    cur = starts.copy()
    for _ in range(sigma):
        if lazy:
            stay = rng.random(n_walks) < 0.5
        else:
            stay = np.zeros(n_walks, dtype=bool)
        move_idx = np.where(~stay)[0]
        for idx in move_idx:
            nbrs = tuple(adj[int(cur[idx])])
            cur[idx] = nbrs[rng.integers(len(nbrs))]
    returned = (cur == starts)
    p = float(returned.mean())
    err = float(np.sqrt(max(p * (1 - p) / n_walks, 0.0)))
    return p, err



def measure_spectral_dimension_rw(adj: Adjacency, n_simplices: int, dimension: int, rng: np.random.Generator,
                                  lazy: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sigmas = choose_even_sigmas(n_simplices, dimension)
    print(f"\n  Random-walk sigma range: {sigmas[0]} to {sigmas[-1]} ({len(sigmas)} even points)")
    P_vals: List[float] = []
    P_errs: List[float] = []
    for i, sigma in enumerate(sigmas):
        n_walks = determine_n_walks(int(sigma), dimension=dimension, target_returns=250, lazy=lazy)
        t0 = time.time()
        P, P_err = compute_return_probability(adj, int(sigma), n_walks, rng=rng, lazy=lazy)
        dt = time.time() - t0
        P_vals.append(P)
        P_errs.append(P_err)
        if i == 0 or (i + 1) % 4 == 0 or P == 0:
            est_returns = int(round(P * n_walks))
            print(f"  sigma={int(sigma):4d}: P={P:.6e} ± {P_err:.2e}, n_walks={n_walks:6d}, returns≈{est_returns:4d}, t={dt:.1f}s")

    P_arr = np.asarray(P_vals)
    P_err_arr = np.asarray(P_errs)
    valid = P_arr > 0
    if valid.sum() < 6:
        raise RuntimeError("Too few nonzero return probabilities for a stable d_s estimate")
    ds = estimate_ds_from_curve(sigmas[valid], P_arr[valid])
    ds_full = np.full_like(sigmas, np.nan, dtype=float)
    ds_full[valid] = ds
    ds_err = np.full_like(sigmas, np.nan, dtype=float)
    log_s = np.log(sigmas[valid].astype(float))
    for i in range(1, valid.sum() - 1):
        denom = log_s[i + 1] - log_s[i - 1]
        rel = 0.0
        for j in (i - 1, i + 1):
            rel += (P_err_arr[valid][j] / max(P_arr[valid][j], 1e-300)) ** 2
        ds_err[np.where(valid)[0][i]] = 2.0 * np.sqrt(rel) / max(denom, 1e-12)
    if valid.sum() >= 3:
        first_valid = np.where(valid)[0][0]
        last_valid = np.where(valid)[0][-1]
        if first_valid + 1 < len(ds_err):
            ds_err[first_valid] = ds_err[min(first_valid + 1, last_valid)]
        if last_valid - 1 >= 0:
            ds_err[last_valid] = ds_err[max(first_valid, last_valid - 1)]
    return sigmas, ds_full, P_arr, P_err_arr


# ============================================================
# SECTION E: LAPLACIAN / HEAT-KERNEL METHOD
# ============================================================


def spectral_dimension_laplacian(adj: Adjacency, n_simplices: int, dimension: int = 3,
                                 n_eigenvalues: int = 400, lazy: bool = True,
                                 quiet: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if not SCIPY_AVAILABLE:
        if not quiet:
            print("  scipy not available; Laplacian method skipped")
        return None, None, None

    if not quiet:
        print(f"\n  Computing random-walk Laplacian eigenvalues (k={n_eigenvalues}, lazy={lazy})...")
    Lrw = random_walk_laplacian(adj, lazy=lazy)
    N = Lrw.shape[0]
    k = int(min(max(16, n_eigenvalues), max(2, N - 2)))
    if not quiet:
        print(f"  Matrix size: {N}x{N}, computing {k} smallest eigenpairs...")
    t0 = time.time()
    vals, _ = eigsh(Lrw, k=k, which="SM", tol=1e-6)
    vals = np.sort(np.real(vals))
    elapsed = time.time() - t0
    if not quiet:
        print(f"  Done in {elapsed:.1f}s. Smallest eigenvalues: {vals[:5]}")

    # Keep the zero mode; use normalized heat trace based on available eigenvalues.
    t_vals = np.logspace(-2, 2, 64)
    # Approximate trace using only the lowest modes. This is reliable for IR / intermediate scales,
    # but it CANNOT recover the deep-UV dimension unless a large fraction of the spectrum is included.
    K_vals = heat_trace_from_eigenvalues(vals, t_vals, normalize="mean")
    ds = estimate_ds_from_curve(t_vals, K_vals)

    if not quiet:
        uv_band = float(np.nanmean(ds[6:14]))
        ir_band = float(np.nanmean(ds[-14:-6]))
        print(f"  d_s at intermediate-small t: {uv_band:.2f} (CDT UV target ~2, only meaningful with many modes)")
        print(f"  d_s at large t: {ir_band:.2f} (expected ~{dimension})")
        if k < min(2000, N - 2):
            print("  [NOTE] This Laplacian run is low-mode truncated: trust IR, do NOT trust UV amplitude yet.")

    return t_vals, ds, K_vals


# ============================================================
# SECTION F: VALIDATION / CROSS-CHECKS
# ============================================================


def compare_estimators(sigmas: np.ndarray, ds_rw: np.ndarray, t_vals: Optional[np.ndarray],
                       ds_lap: Optional[np.ndarray], dimension: int) -> Tuple[bool, str]:
    valid_rw = np.isfinite(ds_rw)
    if valid_rw.sum() < 6:
        return False, "RW estimator has too few valid points"

    rw_ir = float(np.nanmean(ds_rw[np.where(valid_rw)[0][-5:]]))
    parts = [f"RW IR≈{rw_ir:.2f}"]
    if not (dimension - 1.2 <= rw_ir <= dimension + 1.2):
        return False, f"random-walk IR dimension implausible: {rw_ir:.2f}"

    if ds_lap is not None and t_vals is not None:
        lap_ir = float(np.nanmean(ds_lap[-14:-6]))
        parts.append(f"Lap IR≈{lap_ir:.2f}")
        if not (dimension - 1.2 <= lap_ir <= dimension + 1.2):
            return False, f"Laplacian IR dimension implausible: {lap_ir:.2f}"
        if abs(rw_ir - lap_ir) > 0.8:
            return False, f"estimators disagree too much (RW {rw_ir:.2f} vs Lap {lap_ir:.2f})"
    return True, ", ".join(parts)



def validate_graph(adj: Adjacency, n_simplices: int, dimension: int, rng: np.random.Generator) -> bool:
    print("\n" + "=" * 60)
    print("CDT GRAPH VALIDATION")
    print("=" * 60)
    mean_deg, std_deg, min_deg, max_deg = degree_stats(adj)
    print(f"\n  Degree: mean={mean_deg:.2f}, std={std_deg:.2f}, range=[{min_deg}, {max_deg}]")
    frac = giant_component_fraction(adj)
    print(f"  Giant component fraction: {frac:.4f}")
    bip = is_bipartite(adj)
    print(f"  Bipartite: {'YES' if bip else 'NO'}")
    if bip:
        print("  [NOTE] Graph bipartiteness can strongly bias odd-step return probabilities.")
        print("         This script therefore uses EVEN sigma and a LAZY walk.")
    if frac < 0.999:
        print("  [FAIL] Graph is not fully connected enough for reliable diffusion")
        return False

    # Quick random-walk check
    quick_sigmas = np.array([2, 4, 8, 16, 32], dtype=int)
    P_quick, _ = quick_rw_dimension(adj, quick_sigmas, rng=rng, lazy=True, starts_per_sigma=6000)
    ds_quick = estimate_ds_from_curve(quick_sigmas, P_quick)
    quick_mid = float(np.nanmean(ds_quick[1:-1]))
    print(f"  Quick lazy-even RW d_s estimate: {quick_mid:.2f} (target IR ~{dimension})")
    if quick_mid < 1.5:
        print("  [FAIL] Quick RW estimate is too low; geometry or measurement pipeline not trustworthy yet")
        return False
    print("  [OK] Basic validation passed")
    return True


# ============================================================
# SECTION G: NULL CONTROL / OSCILLATIONS
# ============================================================


def degree_preservingish_rewire(adj: Adjacency, n_swaps: int, rng: np.random.Generator) -> Adjacency:
    """Approximate degree-preserving rewiring via edge swaps."""
    edges = {(min(i, j), max(i, j)) for i in adj for j in adj[i] if i < j}
    edge_list = list(edges)
    new_adj: Adjacency = defaultdict(set)
    for i, j in edge_list:
        new_adj[i].add(j)
        new_adj[j].add(i)
    if len(edge_list) < 2:
        return new_adj
    for _ in range(n_swaps):
        (a, b), (c, d) = edge_list[rng.integers(len(edge_list))], edge_list[rng.integers(len(edge_list))]
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            x1, y1 = min(a, c), max(a, c)
            x2, y2 = min(b, d), max(b, d)
        else:
            x1, y1 = min(a, d), max(a, d)
            x2, y2 = min(b, c), max(b, c)
        if x1 == y1 or x2 == y2:
            continue
        if y1 in new_adj[x1] or y2 in new_adj[x2]:
            continue
        # remove old
        if b in new_adj[a] and d in new_adj[c]:
            new_adj[a].remove(b); new_adj[b].remove(a)
            new_adj[c].remove(d); new_adj[d].remove(c)
            new_adj[x1].add(y1); new_adj[y1].add(x1)
            new_adj[x2].add(y2); new_adj[y2].add(x2)
    for i in list(adj.keys()):
        new_adj[i]
    return new_adj



def oscillation_residual(sigmas: np.ndarray, ds_values: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    valid = np.isfinite(ds_values)
    if valid.sum() < 10:
        return None, None, None
    x = np.log(sigmas[valid].astype(float))
    y = ds_values[valid]
    coeffs = np.polyfit(x, y, 3)
    trend = np.polyval(coeffs, x)
    resid = y - trend
    delta_d = float(np.sqrt(np.mean(resid ** 2)))
    return delta_d, trend, resid


# ============================================================
# SECTION H: MAIN
# ============================================================


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 T1_spectral_dimension_cdtpp_revised.py <data_dir_or_file> [dim]")
        sys.exit(1)

    target = sys.argv[1]
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rng = np.random.default_rng(123456789)

    if os.path.isfile(target):
        files = [target]
    else:
        files: List[str] = []
        for ext in ["*.off", "*.txt", "*.adj", "*.dat", "checkpoint_*"]:
            files.extend(sorted(glob.glob(os.path.join(target, ext))))
            files.extend(sorted(glob.glob(os.path.join(target, "**", ext), recursive=True)))
        files = sorted(set(files))

    if not files:
        print(f"No files found in {target}")
        sys.exit(1)

    print(f"Found {len(files)} file(s)")
    filepath = files[0]
    print(f"\nLoading: {filepath}")
    adj, n_sim = load_auto(filepath)
    print(f"Loaded: {n_sim} simplices, {len(adj)} in adjacency")

    sanity = run_sanity_suite(rng)
    if not sanity.passed:
        print(f"\n[ABORT] {sanity.details}")
        sys.exit(2)

    if not validate_graph(adj, n_sim, dim, rng):
        print("\n[ABORT] Graph validation failed. Not attempting oscillation extraction.")
        sys.exit(3)

    print("\n" + "=" * 60)
    print("FULL SPECTRAL DIMENSION MEASUREMENT")
    print("=" * 60)

    sigmas, ds_rw, P_vals, P_errs = measure_spectral_dimension_rw(adj, n_sim, dim, rng, lazy=True)
    print(f"\n  RW IR estimate: {float(np.nanmean(ds_rw[np.isfinite(ds_rw)][-5:])):.2f}")

    t_vals: Optional[np.ndarray] = None
    ds_lap: Optional[np.ndarray] = None
    K_vals: Optional[np.ndarray] = None
    if SCIPY_AVAILABLE:
        t_vals, ds_lap, K_vals = spectral_dimension_laplacian(adj, n_sim, dimension=dim,
                                                              n_eigenvalues=min(600, max(16, n_sim - 2)),
                                                              lazy=True, quiet=False)

    ok, reason = compare_estimators(sigmas, ds_rw, t_vals, ds_lap, dim)
    print(f"\nEstimator agreement check: {reason}")
    if not ok:
        print("[ABORT] Estimators do not agree well enough. Fix pipeline or geometry before claiming T1.")
        np.savez(
            "T1_cdtpp_revised_results.npz",
            n_simplices=n_sim,
            dimension=dim,
            sigma=sigmas,
            d_s_rw=ds_rw,
            P=P_vals,
            P_err=P_errs,
            t=t_vals if t_vals is not None else np.array([]),
            d_s_lap=ds_lap if ds_lap is not None else np.array([]),
            valid_pipeline=False,
            validity_reason=reason,
        )
        sys.exit(4)

    print("\n" + "=" * 60)
    print("OSCILLATION SEARCH (ONLY AFTER VALIDATION)")
    print("=" * 60)
    delta_d, trend, resid = oscillation_residual(sigmas, ds_rw)
    delta_null = None
    if delta_d is not None:
        rewired = degree_preservingish_rewire(adj, n_swaps=max(5000, 10 * n_sim), rng=rng)
        sig_null, ds_null, _, _ = measure_spectral_dimension_rw(rewired, len(rewired), dim, rng, lazy=True)
        delta_null, _, _ = oscillation_residual(sig_null, ds_null)
        print(f"  Observed residual RMS delta_d = {delta_d:.6f}")
        if delta_null is not None:
            print(f"  Rewired-null residual RMS     = {delta_null:.6f}")
            if delta_d <= delta_null:
                print("  [WARN] Oscillation amplitude does not exceed null/noise floor")
            else:
                print("  [OK] Oscillation amplitude exceeds rewired-null floor")
        print("  Target CCDR scale: delta_d ~ 1e-3")
    else:
        print("  Too few valid points for oscillation search")

    np.savez(
        "T1_cdtpp_revised_results.npz",
        n_simplices=n_sim,
        dimension=dim,
        sigma=sigmas,
        d_s_rw=ds_rw,
        P=P_vals,
        P_err=P_errs,
        t=t_vals if t_vals is not None else np.array([]),
        d_s_lap=ds_lap if ds_lap is not None else np.array([]),
        K=K_vals if K_vals is not None else np.array([]),
        delta_d=(np.array([delta_d]) if delta_d is not None else np.array([])),
        delta_null=(np.array([delta_null]) if delta_null is not None else np.array([])),
        valid_pipeline=True,
        validity_reason=reason,
    )
    print("\nResults saved: T1_cdtpp_revised_results.npz")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3 if ds_lap is not None else 2, 1, figsize=(11, 12))
        axes[0].plot(sigmas[np.isfinite(ds_rw)], ds_rw[np.isfinite(ds_rw)], "o-", label="RW d_s")
        axes[0].axhline(dim, color="tab:green", ls="--", label=f"IR target = {dim}")
        axes[0].axhline(2, color="tab:orange", ls=":", label="UV target ≈ 2")
        axes[0].set_xscale("log")
        axes[0].set_ylabel("d_s")
        axes[0].set_title("Random-walk spectral dimension (lazy, even sigma)")
        axes[0].legend()

        axes[1].semilogy(sigmas, np.clip(P_vals, 1e-12, None), "o-")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("sigma")
        axes[1].set_ylabel("P_return")
        axes[1].set_title("Return probability")

        if ds_lap is not None:
            axes[2].plot(t_vals, ds_lap, "-", label="Laplacian d_s")
            axes[2].axhline(dim, color="tab:green", ls="--")
            axes[2].axhline(2, color="tab:orange", ls=":")
            axes[2].set_xscale("log")
            axes[2].set_ylabel("d_s")
            axes[2].set_xlabel("diffusion time t")
            axes[2].set_title("Random-walk Laplacian heat-kernel estimate")
            axes[2].legend()

        plt.tight_layout()
        plt.savefig("T1_cdtpp_revised_spectral_dimension.png", dpi=160)
        print("Plot saved: T1_cdtpp_revised_spectral_dimension.png")
    except Exception as exc:
        print(f"Plotting skipped: {exc}")


if __name__ == "__main__":
    main()
