#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import spatial, stats

from _common_public_data import (
    add_source,
    build_argparser,
    finalize_result,
    json_result_template,
    load_euclid_q1_sample,
    local_density_proxy,
    quantile_split,
    simple_exponential_scale,
)


def _unit_sphere_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Unit-vector sky coordinates from RA/Dec in degrees."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])


def _local_tangent_basis(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """East/north tangent basis vectors on the unit sphere."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    east = np.column_stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)])
    north = np.column_stack([
        -np.sin(dec) * np.cos(ra),
        -np.sin(dec) * np.sin(ra),
        np.cos(dec),
    ])
    return east, north


def _json_float(x: float | np.floating | None) -> float | None:
    if x is None:
        return None
    xf = float(x)
    return xf if math.isfinite(xf) else None


def _json_list(arr: np.ndarray) -> list[float | None]:
    return [_json_float(x) for x in np.asarray(arr, dtype=float)]


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if weights is None:
        if not np.any(good):
            return float("nan")
        return float(np.nanmean(values[good]))
    weights = np.asarray(weights, dtype=float)
    good &= np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return float("nan")
    return float(np.sum(values[good] * weights[good]) / np.sum(weights[good]))


def _empirical_p_greater(null_values: np.ndarray, observed: float) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0 or not math.isfinite(float(observed)):
        return float("nan")
    return float((1 + np.sum(null_values >= observed)) / (len(null_values) + 1))


def _normal_p_greater(z: float) -> float:
    if not math.isfinite(float(z)):
        return float("nan")
    return float(stats.norm.sf(z))


def estimate_local_axes_tangent(
    ra: np.ndarray,
    dec: np.ndarray,
    k: int = 12,
) -> np.ndarray:
    """
    Estimate a local sky-filament axis for each source.

    The old version ran PCA directly in global 3D unit-vector space.  That can
    make the survey footprint itself look like a coherent axis.  Here each
    source's neighbours are projected into that source's tangent plane, PCA is
    done in the local 2D plane, and the winning axis is mapped back to a 3D
    tangent vector.  Orientation is axial, so sign does not matter downstream.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    n = len(ra)
    if n < 4:
        raise ValueError(f"Need at least 4 points to estimate local axes; got {n}.")

    xyz = _unit_sphere_xyz(ra, dec)
    east, north = _local_tangent_basis(ra, dec)
    k_eff = max(3, min(int(k), n - 1))

    tree = spatial.cKDTree(xyz)
    _, idx = tree.query(xyz, k=k_eff + 1)
    if idx.ndim == 1:
        idx = idx[:, None]

    axes = np.full((n, 3), np.nan, dtype=float)
    for i in range(n):
        neighbours = idx[i, 1:]
        if len(neighbours) < 3:
            continue
        delta = xyz[neighbours] - xyz[i]
        uv = np.column_stack([delta @ east[i], delta @ north[i]])
        uv -= np.nanmean(uv, axis=0)
        if not np.all(np.isfinite(uv)) or uv.shape[0] < 3:
            continue
        cov = np.cov(uv.T)
        if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
            continue
        w, v = np.linalg.eigh(cov)
        direction_2d = v[:, int(np.argmax(w))]
        axis = direction_2d[0] * east[i] + direction_2d[1] * north[i]
        norm = np.linalg.norm(axis)
        if norm > 0 and math.isfinite(float(norm)):
            axes[i] = axis / norm

    good = np.all(np.isfinite(axes), axis=1)
    if np.sum(good) < max(4, min(20, n // 2)):
        raise ValueError(f"Too few finite local axes: {int(np.sum(good))}/{n}.")
    return axes


def upper_triangle_distances_and_dots(
    ra: np.ndarray,
    dec: np.ndarray,
    axes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique-pair chord distances and absolute axial dot products."""
    xyz = _unit_sphere_xyz(ra, dec)
    n = len(xyz)
    ii, jj = np.triu_indices(n, k=1)
    dist = np.linalg.norm(xyz[ii] - xyz[jj], axis=1)
    dots = np.abs(np.einsum("ij,ij->i", axes[ii], axes[jj]))
    good = np.isfinite(dist) & np.isfinite(dots)
    return dist[good], dots[good]


def adaptive_pair_bins(
    dist: np.ndarray,
    n_bins: int = 6,
    min_pairs_per_bin: int = 200,
    min_q: float = 0.001,
    max_q: float = 0.25,
) -> np.ndarray:
    """
    Build robust bins from the actually available pair distances.

    Fixed angular bins caused five empty bins in the uploaded run.  Quantile
    bins keep the test from reporting a headline from one populated bin.
    """
    dist = np.asarray(dist, dtype=float)
    dist = dist[np.isfinite(dist) & (dist > 0)]
    if len(dist) < max(30, min_pairs_per_bin):
        raise ValueError(f"Too few finite pairs to bin: {len(dist)}.")

    min_q = float(np.clip(min_q, 0.0, 0.20))
    max_q = float(np.clip(max_q, min_q + 1e-6, 1.0))
    usable = dist[(dist >= np.quantile(dist, min_q)) & (dist <= np.quantile(dist, max_q))]
    if len(usable) < max(30, min_pairs_per_bin):
        usable = dist

    max_bins_by_pairs = max(1, int(len(usable) // max(1, min_pairs_per_bin)))
    n_bins_eff = max(1, min(int(n_bins), max_bins_by_pairs))
    qs = np.linspace(0.0, 1.0, n_bins_eff + 1)
    edges = np.quantile(usable, qs)
    edges = np.unique(edges)
    if len(edges) < 2:
        raise ValueError("Could not build non-degenerate adaptive distance bins.")

    # Make the rightmost edge inclusive after floating-point rounding.
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def binned_orientation(
    dist: np.ndarray,
    dots: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corr = np.full(len(bins) - 1, np.nan, dtype=float)
    counts = np.zeros(len(bins) - 1, dtype=int)
    rmid = 0.5 * (bins[1:] + bins[:-1])
    for j in range(len(bins) - 1):
        sel = (dist >= bins[j]) & (dist < bins[j + 1])
        counts[j] = int(np.sum(sel))
        if counts[j] > 0:
            corr[j] = float(np.nanmean(dots[sel]))
    return rmid, corr, counts


def run_subset(
    name: str,
    df,
    seed: int = 0,
    axis_neighbors: int = 12,
    n_bins: int = 6,
    n_null: int = 200,
    min_pairs_per_bin: int = 200,
    min_distance_quantile: float = 0.001,
    max_distance_quantile: float = 0.25,
) -> dict[str, Any]:
    ra = df["ra"].to_numpy(dtype=float)
    dec = df["dec"].to_numpy(dtype=float)
    finite = np.isfinite(ra) & np.isfinite(dec)
    ra = ra[finite]
    dec = dec[finite]
    n_objects = int(len(ra))
    if n_objects < max(20, axis_neighbors + 2):
        raise ValueError(f"Subset {name!r} has too few usable objects: {n_objects}.")

    axes = estimate_local_axes_tangent(ra, dec, k=axis_neighbors)
    good_axes = np.all(np.isfinite(axes), axis=1)
    ra = ra[good_axes]
    dec = dec[good_axes]
    axes = axes[good_axes]
    n_axes = int(len(ra))

    dist, dots = upper_triangle_distances_and_dots(ra, dec, axes)
    bins = adaptive_pair_bins(
        dist,
        n_bins=n_bins,
        min_pairs_per_bin=min_pairs_per_bin,
        min_q=min_distance_quantile,
        max_q=max_distance_quantile,
    )
    rmid, corr, pair_counts = binned_orientation(dist, dots, bins)
    valid = np.isfinite(corr) & (pair_counts > 0)

    rng = np.random.default_rng(seed)
    null_curves = np.full((int(n_null), len(corr)), np.nan, dtype=float)
    null_weighted_means = np.full(int(n_null), np.nan, dtype=float)
    for i in range(int(n_null)):
        perm = rng.permutation(len(axes))
        shuf_axes = axes[perm]
        _, null_dots = upper_triangle_distances_and_dots(ra, dec, shuf_axes)
        _, null_corr, _ = binned_orientation(dist, null_dots, bins)
        null_curves[i] = null_corr
        null_weighted_means[i] = _weighted_mean(null_corr, pair_counts)

    null_mean = np.nanmean(null_curves, axis=0)
    null_std = np.nanstd(null_curves, axis=0, ddof=1)
    excess = corr - null_mean

    observed_weighted_mean = _weighted_mean(corr, pair_counts)
    null_weighted_mean = _weighted_mean(null_mean, pair_counts)
    observed_minus_null = observed_weighted_mean - null_weighted_mean
    null_summary_std = float(np.nanstd(null_weighted_means, ddof=1))
    z = observed_minus_null / null_summary_std if null_summary_std > 0 else float("nan")
    p_greater = _empirical_p_greater(null_weighted_means, observed_weighted_mean)

    exp_fit_scale = None
    if np.sum(valid) >= 3 and np.any(np.isfinite(excess[valid])):
        # Fit a rough scale to signal magnitude only.  This is diagnostic, not a
        # proof-level estimator.
        y = np.abs(excess[valid]) + 1e-6
        exp_fit_scale = simple_exponential_scale(rmid[valid], y)

    return {
        "subset": name,
        "n_objects_input": n_objects,
        "n_objects_with_axes": n_axes,
        "distance_metric": "unit-sphere chord distance from RA/Dec only",
        "axis_method": "local tangent-plane PCA over k-nearest sky neighbours",
        "axis_neighbors": int(axis_neighbors),
        "bin_edges": _json_list(bins),
        "rmid": _json_list(rmid),
        "pair_counts": [int(x) for x in pair_counts],
        "valid_bin_count": int(np.sum(valid)),
        "corr": _json_list(corr),
        "corr_null_mean": _json_list(null_mean),
        "corr_null_std": _json_list(null_std),
        "corr_excess": _json_list(excess),
        "corr_mean_weighted": _json_float(observed_weighted_mean),
        "corr_null_mean_weighted": _json_float(null_weighted_mean),
        "observed_minus_null": _json_float(observed_minus_null),
        "null_summary_std": _json_float(null_summary_std),
        "z_score": _json_float(z),
        "p_greater_empirical": _json_float(p_greater),
        "exp_fit_scale": _json_float(exp_fit_scale),
        "n_null": int(n_null),
    }


def _density_difference_summary(low: dict[str, Any], high: dict[str, Any]) -> dict[str, Any]:
    low_excess = low.get("observed_minus_null")
    high_excess = high.get("observed_minus_null")
    if low_excess is None or high_excess is None:
        return {
            "density_dependence_proxy": None,
            "density_z_approx": None,
            "density_p_greater_approx": None,
        }
    density_signal = float(low_excess - high_excess)
    low_std = low.get("null_summary_std") or float("nan")
    high_std = high.get("null_summary_std") or float("nan")
    denom = math.sqrt(float(low_std) ** 2 + float(high_std) ** 2)
    z = density_signal / denom if denom > 0 else float("nan")
    return {
        "density_dependence_proxy": _json_float(density_signal),
        "density_z_approx": _json_float(z),
        "density_p_greater_approx": _json_float(_normal_p_greater(z)),
    }


def main() -> None:
    parser = build_argparser("T5 — Euclid-Q1 filament ordering")
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--axis-neighbors", type=int, default=12)
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--min-pairs-per-bin", type=int, default=200)
    parser.add_argument("--density-quantile", type=float, default=0.25)
    parser.add_argument("--min-distance-quantile", type=float, default=0.001)
    parser.add_argument("--max-distance-quantile", type=float, default=0.25)
    args = parser.parse_args()

    result = json_result_template(
        "T5 — Euclid-Q1 filament ordering",
        "Estimate local sky-filament axes from Euclid Q1 geometry, measure pairwise orientation correlation, and compare full-sample, low-density, and high-density subsets against multi-shuffle nulls. This is a public RA/Dec-only proxy, not a 3D filament reconstruction.",
    )

    df = load_euclid_q1_sample(max_rows=args.max_rows, seed=args.seed)
    df = df.copy()
    finite = np.isfinite(df["ra"].to_numpy(dtype=float)) & np.isfinite(df["dec"].to_numpy(dtype=float))
    df = df.loc[finite].reset_index(drop=True)
    if len(df) < max(20, args.axis_neighbors + 2):
        raise RuntimeError(f"Too few finite Euclid rows after RA/Dec cleaning: {len(df)}")

    density = local_density_proxy(df["ra"], df["dec"], k=10)
    df["density_proxy"] = density
    lo, hi = quantile_split(density, q=float(args.density_quantile))

    common_kwargs = dict(
        seed=args.seed,
        axis_neighbors=args.axis_neighbors,
        n_bins=args.n_bins,
        n_null=args.n_null,
        min_pairs_per_bin=args.min_pairs_per_bin,
        min_distance_quantile=args.min_distance_quantile,
        max_distance_quantile=args.max_distance_quantile,
    )
    full = run_subset("full_sample", df, **common_kwargs)
    low = run_subset("low_density", df.loc[lo].reset_index(drop=True), **common_kwargs)
    high = run_subset("high_density", df.loc[hi].reset_index(drop=True), **common_kwargs)

    density_summary = _density_difference_summary(low, high)

    enough_bins = min(full["valid_bin_count"], low["valid_bin_count"], high["valid_bin_count"]) >= 3
    low_positive = (low.get("observed_minus_null") is not None) and (low["observed_minus_null"] > 0)
    low_significant = (low.get("p_greater_empirical") is not None) and (low["p_greater_empirical"] <= 0.05)
    density_positive = (density_summary.get("density_dependence_proxy") is not None) and (density_summary["density_dependence_proxy"] > 0)
    density_significant = (density_summary.get("density_p_greater_approx") is not None) and (density_summary["density_p_greater_approx"] <= 0.05)

    confirm_like = bool(enough_bins and low_positive and low_significant and density_positive and density_significant)
    null_like = bool(enough_bins and not confirm_like)

    result["full_sample"] = full
    result["low_density"] = low
    result["high_density"] = high
    result["headline"] = {
        "expected_direction_low_density_positive": bool(low_positive),
        "expected_direction_confirm_like": confirm_like,
        "density_dependence_proxy": density_summary["density_dependence_proxy"],
        "density_z_approx": density_summary["density_z_approx"],
        "density_p_greater_approx": density_summary["density_p_greater_approx"],
        "minimum_valid_bin_count": int(min(full["valid_bin_count"], low["valid_bin_count"], high["valid_bin_count"])),
        "screening_assessment": "confirm_like" if confirm_like else ("null_or_inconclusive" if null_like else "insufficient_bins"),
    }
    result["falsification_logic"] = {
        "confirm_like": "Low-density subset has a positive observed-minus-null orientation excess, that excess is significant against multi-shuffle nulls, and the low-minus-high density contrast is also positive/significant with at least three populated bins in every subset.",
        "falsify_like": "With adequate populated bins, the low-density excess is non-positive or indistinguishable from shuffled nulls, or the low-minus-high density contrast is non-positive/insignificant.",
        "proxy_caveat": "This uses public Euclid Q1 RA/Dec geometry only. It cannot prove a 3D cosmic-web filament-ordering claim without redshift-resolved 3D reconstruction and collaboration-grade masks/selection functions.",
    }
    result["notes"].append("Fixed-bin NaN artifact removed: bins are now adaptive to actual pair-distance quantiles and pair counts are reported.")
    result["notes"].append("Single shuffled null replaced by multi-shuffle empirical p-values and z diagnostics.")
    result["notes"].append("NaN values are serialized as JSON nulls for standards-compliant output.")

    add_source(result, "Euclid Q1 IRSA TAP", "https://irsa.ipac.caltech.edu/TAP/sync")
    finalize_result(__file__, result, args.out)


if __name__ == "__main__":
    main()
