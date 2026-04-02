#!/usr/bin/env python3
"""
T1_spectral_dimension_oscillation.py
2D CDT Monte Carlo with spectral dimension measurement.
Targets: delta_d ~ nu ~ 10^-3 oscillation.
"""
import numpy as np
from collections import defaultdict
import time


class CDT2D:
    """Minimal 2D CDT triangulation on a torus."""

    def __init__(self, T=20, N_spatial=20):
        """
        T: number of time slices
        N_spatial: vertices per time slice
        """
        self.T = T
        self.N_s = N_spatial
        self.N_vertices = T * N_spatial
        # Build initial regular triangulation
        self.adjacency = defaultdict(set)
        self._build_regular()

    def _build_regular(self):
        """Build a regular triangulation: each time slice is a ring."""
        for t in range(self.T):
            for i in range(self.N_s):
                v = t * self.N_s + i
                # Spatial neighbours in same slice
                v_left = t * self.N_s + (i - 1) % self.N_s
                v_right = t * self.N_s + (i + 1) % self.N_s
                # Neighbours in next time slice
                t_next = (t + 1) % self.T
                v_up = t_next * self.N_s + i
                v_up_right = t_next * self.N_s + (i + 1) % self.N_s

                self.adjacency[v].update([v_left, v_right, v_up, v_up_right])
                self.adjacency[v_up].add(v)
                self.adjacency[v_up_right].add(v)

    def pachner_move(self):
        """Attempt a random Pachner move (simplified: edge flip)."""
        v = np.random.randint(self.N_vertices)
        neighbours = list(self.adjacency[v])
        if len(neighbours) < 3:
            return False
        # Pick two non-adjacent neighbours
        i, j = np.random.choice(len(neighbours), 2, replace=False)
        n1, n2 = neighbours[i], neighbours[j]
        # Flip: remove v-n1, add n2-n1 (or vice versa) with Metropolis
        if n2 in self.adjacency[n1]:
            return False  # already connected
        # Simple acceptance (detailed balance for flat measure)
        if np.random.random() < 0.5:
            self.adjacency[v].discard(n1)
            self.adjacency[n1].discard(v)
            self.adjacency[n1].add(n2)
            self.adjacency[n2].add(n1)
            return True
        return False

    def thermalise(self, n_sweeps=100):
        """Run n_sweeps * N_vertices attempted moves."""
        accepted = 0
        for _ in range(n_sweeps * self.N_vertices):
            if self.pachner_move():
                accepted += 1
        return accepted / (n_sweeps * self.N_vertices)

    def return_probability(self, sigma, n_walks=500):
        """Average return probability after sigma random walk steps."""
        returns = 0
        for _ in range(n_walks):
            v0 = np.random.randint(self.N_vertices)
            v = v0
            for _ in range(sigma):
                neighbours = list(self.adjacency[v])
                if neighbours:
                    v = neighbours[np.random.randint(len(neighbours))]
            if v == v0:
                returns += 1
        return returns / n_walks

    def spectral_dimension(self, sigma_values=None):
        """Compute d_s(sigma) = -2 d ln P / d ln sigma."""
        if sigma_values is None:
            sigma_values = np.unique(np.logspace(0.5, 2.5, 40).astype(int))
            sigma_values = sigma_values[sigma_values > 0]

        P_values = []
        for sigma in sigma_values:
            P = self.return_probability(sigma, n_walks=1000)
            P_values.append(max(P, 1e-10))

        P_values = np.array(P_values)
        log_sigma = np.log(sigma_values.astype(float))
        log_P = np.log(P_values)

        # Numerical derivative
        d_s = -2 * np.gradient(log_P, log_sigma)
        return sigma_values, d_s, P_values


def run_T1_test(N_configs=500, T=20, N_spatial=25):
    """
    Run the T1 test: measure spectral dimension across an ensemble
    of CDT configurations and look for oscillations with amplitude ~ 10^-3.
    """
    print("=" * 60)
    print("T1 TEST: Spectral Dimension Oscillation")
    print(f"Target: delta_d ~ nu ~ 10^-3")
    print(f"Configs: {N_configs}, T={T}, N_spatial={N_spatial}")
    print("=" * 60)

    cdt = CDT2D(T=T, N_spatial=N_spatial)
    print(f"Vertices: {cdt.N_vertices}")

    # Thermalise
    print("Thermalising...", end=" ", flush=True)
    acc = cdt.thermalise(n_sweeps=200)
    print(f"done (acceptance: {acc:.3f})")

    # Collect spectral dimension measurements
    all_ds = []
    sigma_ref = None
    t0 = time.time()

    for i in range(N_configs):
        # Decorrelation sweeps
        cdt.thermalise(n_sweeps=10)

        sigmas, ds, _ = cdt.spectral_dimension()
        if sigma_ref is None:
            sigma_ref = sigmas
        all_ds.append(ds)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N_configs - i - 1) / rate
            print(f"  Config {i+1}/{N_configs} "
                  f"({rate:.1f} cfg/s, ETA: {eta:.0f}s)")

    all_ds = np.array(all_ds)  # shape: (N_configs, N_sigma)
    ds_mean = np.mean(all_ds, axis=0)
    ds_std = np.std(all_ds, axis=0) / np.sqrt(N_configs)

    # Look for oscillations
    # Subtract smooth trend (polynomial fit) to isolate oscillatory component
    valid = np.isfinite(ds_mean)
    log_s = np.log(sigma_ref[valid].astype(float))
    ds_v = ds_mean[valid]

    # Fit smooth polynomial (degree 3)
    coeffs = np.polyfit(log_s, ds_v, 3)
    ds_smooth = np.polyval(coeffs, log_s)
    residuals = ds_v - ds_smooth

    # RMS of residuals = amplitude of oscillation
    delta_d = np.sqrt(np.mean(residuals**2))

    # Statistical significance
    ds_std_v = ds_std[valid]
    mean_err = np.mean(ds_std_v)
    significance = delta_d / mean_err if mean_err > 0 else 0

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Configurations: {N_configs}")
    print(f"<d_s> at large sigma: {ds_mean[valid][-5:].mean():.3f}")
    print(f"Residual RMS (delta_d): {delta_d:.5f}")
    print(f"Target delta_d = nu: 0.001")
    print(f"Mean statistical error: {mean_err:.5f}")
    print(f"Significance: {significance:.1f} sigma")
    print(f"Time: {elapsed:.1f}s")
    print()

    if significance > 3:
        print("[PASS] OSCILLATION DETECTED at > 3 sigma")
    elif significance > 2:
        print("[MARGINAL] detection (2-3 sigma) -- increase N_configs")
    else:
        print("[INSUFFICIENT] No significant oscillation detected")
        configs_needed = int(N_configs * (3 / max(significance, 0.1))**2)
        print(f"  Need ~{configs_needed} configs for 3 sigma at this amplitude")

    # Save results
    np.savez('T1_results.npz',
             sigma=sigma_ref, ds_mean=ds_mean, ds_std=ds_std,
             residuals=residuals, delta_d=delta_d,
             significance=significance, N_configs=N_configs)

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].errorbar(sigma_ref[valid], ds_v, yerr=ds_std_v,
                         fmt='o', markersize=2, label='Measured')
        axes[0].plot(sigma_ref[valid], ds_smooth, 'r-', label='Smooth fit')
        axes[0].set_xlabel('sigma (walk steps)')
        axes[0].set_ylabel('d_s(sigma)')
        axes[0].set_xscale('log')
        axes[0].set_title(f'T1: Spectral Dimension (N={N_configs} configs)')
        axes[0].legend()

        axes[1].errorbar(log_s, residuals, yerr=ds_std_v,
                         fmt='o', markersize=2)
        axes[1].axhline(0, color='k', linestyle='--')
        axes[1].axhline(1e-3, color='r', linestyle=':', label='nu = 1e-3')
        axes[1].axhline(-1e-3, color='r', linestyle=':')
        axes[1].set_xlabel('ln sigma')
        axes[1].set_ylabel('Residual (d_s - smooth)')
        axes[1].set_title(f'Oscillatory residual: RMS = {delta_d:.5f}, '
                          f'significance = {significance:.1f} sigma')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('T1_spectral_dimension.png', dpi=150)
        print(f"\nPlot saved: T1_spectral_dimension.png")
    except ImportError:
        print("matplotlib not available -- skipping plot")

    return delta_d, significance


if __name__ == '__main__':
    # Scale up for your i9:
    # N_configs=2500 gives 5 sigma sensitivity at delta_d=10^-3
    # With T=20, N_spatial=25 (500 vertices): ~2 hours on i9
    run_T1_test(N_configs=2500, T=20, N_spatial=25)
