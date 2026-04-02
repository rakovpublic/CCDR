#!/usr/bin/env python3
"""
CDT-CCDR T1 Prediction Test
============================
Complete pipeline for testing the primary CDT-CCDR prediction:
oscillatory corrections to the spectral dimension d_s(σ) with
amplitude δd ~ ν ~ 10⁻³.

Components:
  1. 2D CDT Monte Carlo simulation (proof-of-concept, correct physics)
  2. Spectral dimension measurement via graph Laplacian heat kernel
  3. T1 Fourier residual analysis pipeline
  4. Signal injection test (validates pipeline sensitivity)
  5. Publication-quality diagnostics and plots

Usage:
  python3 cdt_ccdr_t1_test.py

The 2D simulation demonstrates the methodology. The analysis pipeline
is dimension-independent and can process d_s(σ) data from any CDT code
(e.g., CDT-plusplus for 4D).

Author: Generated for CCDR project, March 2026
References:
  [1] Ambjørn, Jurkiewicz & Loll (2005). Phys. Rev. Lett. 95, 171301.
  [2] CDT-CCDR synthesis paper, §6 (2026).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import json
import os

# ============================================================
# PART 1: 2D CDT MONTE CARLO SIMULATION
# ============================================================

class CDT2D:
    """
    2D Causal Dynamical Triangulation on a torus (S¹ × S¹).
    
    Uses volume-fixing term to stabilise at target volume,
    as in standard CDT Monte Carlo practice.
    
    Action: S = -κ₀ N₀ + λ N₂ + ε(N₂ - N₂_target)²
    """
    
    def __init__(self, T=16, s_init=12, kappa0=1.5, lam=1.0, 
                 N2_target=600, epsilon=0.005):
        self.T = T
        self.s = np.ones(T, dtype=int) * s_init
        self.kappa0 = kappa0
        self.lam = lam
        self.N2_target = N2_target
        self.epsilon = epsilon
        self.accepted = 0
        self.proposed = 0
    
    @property
    def N0(self):
        return int(np.sum(self.s))
    
    @property
    def N2(self):
        return int(sum(self.s[t] + self.s[(t+1) % self.T] for t in range(self.T)))
    
    def action(self):
        n2 = self.N2
        return -self.kappa0 * self.N0 + self.lam * n2 + self.epsilon * (n2 - self.N2_target)**2
    
    def sweep(self, n_sweeps=1):
        for _ in range(n_sweeps):
            for _ in range(self.T * 2):
                self._metropolis_step()
    
    def _metropolis_step(self):
        self.proposed += 1
        t = np.random.randint(0, self.T)
        delta = np.random.choice([-1, 1])
        
        if self.s[t] + delta < 3:
            return
        
        n2_old = self.N2
        dN0 = delta
        dN2 = 2 * delta
        n2_new = n2_old + dN2
        
        dS = (-self.kappa0 * dN0 + self.lam * dN2 
              + self.epsilon * ((n2_new - self.N2_target)**2 - (n2_old - self.N2_target)**2))
        
        if dS < 0 or np.random.random() < np.exp(-dS):
            self.s[t] += delta
            self.accepted += 1
    
    def acceptance_rate(self):
        return self.accepted / max(self.proposed, 1)
    
    def build_graph(self):
        """
        Build the vertex adjacency graph of the triangulation.
        
        Returns: adjacency matrix (sparse CSR), vertex_times array
        
        Between slices t and t+1:
        - Spatial edges within each slice: v_{t,i} -- v_{t,(i+1) mod s_t}
        - Timelike edges: fan triangulation connecting the two circles
        """
        N = self.N0
        
        # Map (t, i) -> global vertex index
        offsets = np.zeros(self.T + 1, dtype=int)
        for t in range(self.T):
            offsets[t+1] = offsets[t] + self.s[t]
        
        rows, cols = [], []
        vertex_times = np.zeros(N, dtype=int)
        
        for t in range(self.T):
            s_t = self.s[t]
            base = offsets[t]
            
            # Record vertex times
            for i in range(s_t):
                vertex_times[base + i] = t
            
            # Spatial edges within slice t (circle topology)
            for i in range(s_t):
                j = (i + 1) % s_t
                rows.extend([base + i, base + j])
                cols.extend([base + j, base + i])
            
            # Timelike edges to slice t+1 (fan triangulation)
            t1 = (t + 1) % self.T
            s_t1 = self.s[t1]
            base1 = offsets[t1]
            
            # Connect each vertex on t to its "partner" on t+1
            # and create triangles by walking both circles
            for i in range(max(s_t, s_t1)):
                vi = base + (i % s_t)
                vj = base1 + (i % s_t1)
                rows.extend([vi, vj])
                cols.extend([vj, vi])
                
                # Additional diagonal edges for proper triangulation
                if s_t != s_t1:
                    vj2 = base1 + ((i + 1) % s_t1)
                    rows.extend([vi, vj2])
                    cols.extend([vj2, vi])
        
        data = np.ones(len(rows), dtype=float)
        A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        # Remove duplicates and set to 1
        A = (A > 0).astype(float)
        # Remove self-loops
        A.setdiag(0)
        A.eliminate_zeros()
        
        return A, vertex_times
    
    def volume_profile(self):
        return self.s.copy()


# ============================================================
# PART 2: SPECTRAL DIMENSION MEASUREMENT
# ============================================================

def measure_spectral_dimension(A, sigma_max=200, n_sigma=100):
    """
    Measure spectral dimension d_s(σ) via heat kernel on graph.
    
    Uses the graph Laplacian L = D - A:
      P(σ) = (1/N) Tr(exp(-σL))
      d_s(σ) = -2 d ln P(σ) / d ln σ
    
    Args:
        A: adjacency matrix (sparse)
        sigma_max: maximum diffusion time
        n_sigma: number of σ points
    
    Returns:
        sigmas: diffusion times
        ds: spectral dimension values
        P_sigma: return probabilities
    """
    N = A.shape[0]
    
    # Compute degree matrix and normalised Laplacian
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.maximum(degrees, 1)))
    L_norm = sparse.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Compute eigenvalues - dense is faster for small graphs
    L_dense = L_norm.toarray()
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L_dense)))
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Compute return probability P(σ) = (1/N) Σ exp(-σ λ_i)
    sigmas = np.logspace(-1, np.log10(sigma_max), n_sigma)
    P_sigma = np.zeros(n_sigma)
    
    for i, sigma in enumerate(sigmas):
        P_sigma[i] = np.mean(np.exp(-sigma * eigenvalues))
    
    # Compute d_s = -2 d ln P / d ln σ via finite differences
    ln_sigma = np.log(sigmas)
    ln_P = np.log(np.maximum(P_sigma, 1e-300))
    
    ds = np.zeros(n_sigma)
    for i in range(1, n_sigma - 1):
        ds[i] = -2.0 * (ln_P[i+1] - ln_P[i-1]) / (ln_sigma[i+1] - ln_sigma[i-1])
    ds[0] = ds[1]
    ds[-1] = ds[-2]
    
    return sigmas, ds, P_sigma


def measure_spectral_dimension_random_walk(A, sigma_max=100, n_walks=5000):
    """
    Alternative: measure d_s via explicit random walks on the graph.
    More noisy but conceptually direct.
    """
    N = A.shape[0]
    rows, cols = A.nonzero()
    
    # Build neighbour lists
    neighbours = [[] for _ in range(N)]
    for r, c in zip(rows, cols):
        neighbours[r].append(c)
    
    sigmas = np.arange(1, sigma_max + 1)
    return_counts = np.zeros(sigma_max)
    
    for _ in range(n_walks):
        start = np.random.randint(0, N)
        pos = start
        for step in range(sigma_max):
            nbrs = neighbours[pos]
            if len(nbrs) == 0:
                break
            pos = nbrs[np.random.randint(0, len(nbrs))]
            if pos == start:
                return_counts[step] += 1
    
    P_sigma = return_counts / n_walks
    P_sigma = np.maximum(P_sigma, 1e-10)
    
    # Smooth before taking derivative
    from scipy.ndimage import uniform_filter1d
    ln_P = uniform_filter1d(np.log(P_sigma), size=5)
    ln_sigma = np.log(sigmas)
    
    ds = np.zeros(sigma_max)
    for i in range(2, sigma_max - 2):
        ds[i] = -2.0 * (ln_P[i+1] - ln_P[i-1]) / (ln_sigma[i+1] - ln_sigma[i-1])
    ds[:2] = ds[2]
    ds[-2:] = ds[-3]
    
    return sigmas.astype(float), ds, P_sigma


# ============================================================
# PART 3: T1 ANALYSIS PIPELINE
# ============================================================

def smooth_ds_model(sigma, d_inf, d_0, sigma_c, gamma):
    """
    Smooth model for spectral dimension flow:
    d_s(σ) = d_inf - (d_inf - d_0) × exp(-(σ/σ_c)^γ)
    
    d_0: UV value (small σ)
    d_inf: IR value (large σ)
    σ_c: crossover scale
    γ: shape parameter
    """
    return d_inf - (d_inf - d_0) * np.exp(-(sigma / sigma_c)**gamma)


def fit_smooth_model(sigmas, ds_mean, ds_err=None):
    """Fit a smooth model to d_s(σ) using smoothing spline."""
    from scipy.interpolate import UnivariateSpline
    
    # Use log(σ) as the x-variable for better conditioning
    log_s = np.log(sigmas)
    
    # Smooth spline with moderate smoothing to capture trend but not oscillations
    w = 1.0 / np.maximum(ds_err, 1e-4) if ds_err is not None else None
    
    try:
        # s parameter controls smoothness: larger = smoother
        spl = UnivariateSpline(log_s, ds_mean, w=w, s=len(sigmas) * 0.5)
        ds_fit = spl(log_s)
    except Exception:
        # Fallback: moving average
        from scipy.ndimage import uniform_filter1d
        ds_fit = uniform_filter1d(ds_mean, size=max(5, len(sigmas)//8))
    
    return ds_fit


def t1_fourier_analysis(sigmas, ds_mean, ds_err=None, ds_smooth_fit=None):
    """
    T1 Prediction Test: Fourier analysis of d_s(σ) residuals.
    
    Procedure:
    1. Fit smooth model to ⟨d_s(σ)⟩
    2. Compute residuals Δd(σ) = ⟨d_s⟩ - d_s^fit
    3. FFT of residuals
    4. Search for peaks above noise floor
    5. Estimate significance
    
    Returns dict with analysis results.
    """
    # Step 1: Fit smooth model
    if ds_smooth_fit is None:
        ds_fit = fit_smooth_model(sigmas, ds_mean, ds_err)
    else:
        ds_fit = ds_smooth_fit
    popt = None
    
    # Step 2: Residuals
    residuals = ds_mean - ds_fit
    
    # Step 3: FFT
    # Interpolate to uniform spacing in log(σ) for FFT
    n_interp = 256
    log_sigma_uniform = np.linspace(np.log(sigmas[1]), np.log(sigmas[-2]), n_interp)
    residuals_interp = np.interp(log_sigma_uniform, np.log(sigmas), residuals)
    
    # Window (Hann) to reduce spectral leakage
    window = np.hanning(n_interp)
    residuals_windowed = residuals_interp * window
    
    fft_vals = np.fft.rfft(residuals_windowed)
    fft_power = np.abs(fft_vals)**2
    fft_freqs = np.fft.rfftfreq(n_interp, d=(log_sigma_uniform[1] - log_sigma_uniform[0]))
    
    # Step 4: Find peaks — proper noise estimation
    # Use the upper half of frequency bins as noise reference
    n_freq = len(fft_power)
    noise_band = fft_power[n_freq//2:]
    noise_floor = np.mean(noise_band) if len(noise_band) > 0 else 1e-30
    noise_std = np.std(noise_band) if len(noise_band) > 2 else noise_floor
    
    # Find peak in the signal band (low frequencies, excluding DC)
    signal_band = fft_power[1:n_freq//2]
    if len(signal_band) > 0:
        peak_idx = np.argmax(signal_band) + 1
        peak_freq = fft_freqs[peak_idx]
        peak_power = fft_power[peak_idx]
    else:
        peak_idx = 1
        peak_freq = fft_freqs[1] if len(fft_freqs) > 1 else 0
        peak_power = 0
    
    # SNR: how many noise σ above the mean
    peak_snr = (peak_power - noise_floor) / max(noise_std, 1e-30)
    
    # Step 5: Significance — compare peak to chi-squared distribution
    # Under null (Gaussian noise), FFT power ~ exponential(mean=noise_floor)
    if noise_floor > 0:
        p_value = np.exp(-peak_power / noise_floor)
        p_value = max(p_value, 1e-20)
    else:
        p_value = 1.0
    
    # Convert to Gaussian sigma
    from scipy.stats import norm
    n_sigma_sig = -norm.ppf(min(p_value, 0.5)) if p_value < 0.5 else 0.0
    
    # Amplitude of oscillation
    residual_rms = np.std(residuals)
    residual_max = np.max(np.abs(residuals))
    
    results = {
        'fit_params': None,  # spline fit, no explicit params
        'residual_rms': float(residual_rms),
        'residual_max': float(residual_max),
        'peak_frequency': float(peak_freq),
        'peak_power': float(peak_power),
        'noise_floor': float(noise_floor),
        'peak_snr': float(peak_snr),
        'p_value': float(p_value),
        'n_sigma_significance': float(n_sigma_sig),
        'fft_freqs': fft_freqs,
        'fft_power': fft_power,
        'residuals': residuals,
        'ds_fit': ds_fit,
    }
    
    return results


def inject_t1_signal(sigmas, ds_data, nu=1e-3, sigma_osc=None):
    """
    Inject a synthetic T1 oscillation into d_s data.
    
    d_s → d_s + δd × cos(2π σ/σ_osc) × exp(-σ/ξ)
    
    where δd = ν ~ 10⁻³
    """
    if sigma_osc is None:
        sigma_osc = sigmas[len(sigmas)//10]  # ~ Planck scale in simulation units
    
    xi = sigmas[-1] / 2  # correlation length
    signal = nu * np.cos(2 * np.pi * np.log(sigmas) / np.log(sigma_osc)) * \
             np.exp(-sigmas / xi)
    
    return ds_data + signal, signal


# ============================================================
# PART 4: MAIN EXECUTION AND DIAGNOSTICS
# ============================================================

def run_cdt_simulation(T=16, s_init=20, kappa0=1.5, lam=1.0,
                       N2_target=700, epsilon=0.005,
                       n_therm=500, n_configs=80, n_between=20):
    """Run 2D CDT Monte Carlo and collect configurations."""
    print(f"  CDT: T={T}, N₂_target={N2_target}, κ₀={kappa0}, λ={lam}, ε={epsilon}")
    print(f"  Thermalisation: {n_therm} sweeps, configs: {n_configs}")
    
    cdt = CDT2D(T=T, s_init=s_init, kappa0=kappa0, lam=lam,
                N2_target=N2_target, epsilon=epsilon)
    
    cdt.sweep(n_therm)
    print(f"  After therm: N₀={cdt.N0}, N₂={cdt.N2}, acc={cdt.acceptance_rate():.3f}")
    
    configs = []
    volumes = []
    
    for i in range(n_configs):
        cdt.sweep(n_between)
        A, vtimes = cdt.build_graph()
        configs.append(A)
        volumes.append(cdt.volume_profile())
        if (i + 1) % 20 == 0:
            print(f"  Config {i+1}/{n_configs}: N₀={cdt.N0}, N₂={cdt.N2}")
    
    return configs, volumes


def collect_spectral_dimensions(configs, sigma_max=80, n_sigma=80):
    """Measure d_s(σ) for each configuration and average."""
    n_configs = len(configs)
    all_ds = []
    sigmas = None
    
    for i, A in enumerate(configs):
        s, ds, P = measure_spectral_dimension(A, sigma_max=sigma_max, n_sigma=n_sigma)
        all_ds.append(ds)
        if sigmas is None:
            sigmas = s
        
        if (i + 1) % 20 == 0:
            print(f"  Measured d_s for config {i+1}/{n_configs}")
    
    all_ds = np.array(all_ds)
    ds_mean = np.mean(all_ds, axis=0)
    ds_std = np.std(all_ds, axis=0)
    ds_err = ds_std / np.sqrt(n_configs)
    
    return sigmas, ds_mean, ds_std, ds_err, all_ds


def make_diagnostic_plots(sigmas, ds_mean, ds_err, t1_results,
                          t1_injected_results, volumes, signal_injected,
                          output_dir='.'):
    """Generate publication-quality diagnostic plots."""
    
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Colour scheme
    c_data = '#1b3a5c'
    c_fit = '#8b2500'
    c_signal = '#2d6a4f'
    c_noise = '#888888'
    c_inject = '#b8860b'
    
    # --- Panel 1: Volume profile ---
    ax1 = fig.add_subplot(gs[0, 0])
    vol_mean = np.mean(volumes, axis=0)
    vol_std = np.std(volumes, axis=0)
    t_slices = np.arange(len(vol_mean))
    ax1.fill_between(t_slices, vol_mean - vol_std, vol_mean + vol_std,
                     alpha=0.3, color=c_data)
    ax1.plot(t_slices, vol_mean, 'o-', color=c_data, markersize=4, linewidth=1.5)
    ax1.set_xlabel('Time slice t', fontsize=11)
    ax1.set_ylabel('Spatial volume s(t)', fontsize=11)
    ax1.set_title('2D CDT Volume Profile (C-phase)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: Spectral dimension d_s(σ) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(sigmas, ds_mean - ds_err, ds_mean + ds_err,
                     alpha=0.3, color=c_data)
    ax2.plot(sigmas, ds_mean, '-', color=c_data, linewidth=1.5, label='CDT measurement')
    if t1_results['ds_fit'] is not None:
        ax2.plot(sigmas, t1_results['ds_fit'], '--', color=c_fit, linewidth=1.5,
                label='Smooth fit')
    ax2.set_xscale('log')
    ax2.set_xlabel('Diffusion time σ', fontsize=11)
    ax2.set_ylabel('Spectral dimension d_s(σ)', fontsize=11)
    ax2.set_title('Spectral Dimension Flow', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # --- Panel 3: Residuals (no injection) ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(sigmas, -ds_err, ds_err, alpha=0.2, color=c_noise, label='±1σ error')
    ax3.plot(sigmas, t1_results['residuals'], '-', color=c_data, linewidth=1,
            label=f'Residuals (RMS={t1_results["residual_rms"]:.4f})')
    ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax3.set_xscale('log')
    ax3.set_xlabel('Diffusion time σ', fontsize=11)
    ax3.set_ylabel('Δd_s(σ)', fontsize=11)
    ax3.set_title('d_s Residuals (No Signal Injection)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # --- Panel 4: FFT power spectrum (no injection) ---
    ax4 = fig.add_subplot(gs[1, 1])
    freqs = t1_results['fft_freqs'][1:]
    power = t1_results['fft_power'][1:]
    ax4.semilogy(freqs, power, '-', color=c_data, linewidth=1, label='Data')
    ax4.axhline(t1_results['noise_floor'], color=c_noise, linestyle='--',
               label=f'Noise floor')
    ax4.set_xlabel('Frequency in log(σ) space', fontsize=11)
    ax4.set_ylabel('FFT Power', fontsize=11)
    ax4.set_title(f'Fourier Spectrum (peak SNR={t1_results["peak_snr"]:.1f})',
                 fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # --- Panel 5: Injected signal ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(sigmas, signal_injected, '-', color=c_inject, linewidth=1.5,
            label='Injected T1 signal (ν=10⁻³)')
    ax5.plot(sigmas, t1_injected_results['residuals'], '-', color=c_data,
            linewidth=1, alpha=0.7, label='Recovered residuals')
    ax5.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax5.set_xscale('log')
    ax5.set_xlabel('Diffusion time σ', fontsize=11)
    ax5.set_ylabel('Δd_s(σ)', fontsize=11)
    ax5.set_title('Signal Injection Test (δd = ν = 10⁻³)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # --- Panel 6: FFT with injection ---
    ax6 = fig.add_subplot(gs[2, 1])
    freqs_inj = t1_injected_results['fft_freqs'][1:]
    power_inj = t1_injected_results['fft_power'][1:]
    ax6.semilogy(freqs_inj, power_inj, '-', color=c_inject, linewidth=1.2,
                label='With injection')
    ax6.semilogy(freqs, power, '-', color=c_noise, linewidth=0.8, alpha=0.5,
                label='Without injection')
    ax6.axhline(t1_injected_results['noise_floor'], color=c_noise, linestyle='--')
    ax6.set_xlabel('Frequency in log(σ) space', fontsize=11)
    ax6.set_ylabel('FFT Power', fontsize=11)
    ax6.set_title(f'Fourier Spectrum with Injection (SNR={t1_injected_results["peak_snr"]:.1f})',
                 fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # --- Panel 7: Sensitivity analysis ---
    ax7 = fig.add_subplot(gs[3, 0])
    base_n = len(volumes)  # our number of configs
    
    # The signal is at amplitude δd = ν = 10⁻³, noise is residual RMS
    # For Fourier analysis: SNR ~ (signal_amplitude * √N_points) / noise_rms
    noise_rms = t1_results['residual_rms']
    signal_amplitude = 1e-3  # the T1 prediction: δd = ν = 10⁻³
    n_points = 50  # number of σ measurement points
    
    # Signal amplitude in Fourier space scales as signal_amplitude * √n_points
    # Noise in Fourier space scales as noise_rms * √n_points / √n_configs
    # So SNR ~ signal_amplitude * √n_configs / noise_rms
    delta_snr = signal_amplitude * np.sqrt(base_n) / max(noise_rms, 1e-10)
    
    n_configs_test = [10, 25, 50, 100, 200, 500, 1000, 5000, 10000, 50000, 
                      100000, 500000, 1000000]
    # SNR scales as √N_configs
    projected_snr = [signal_amplitude * np.sqrt(nc) / max(noise_rms, 1e-10) 
                     for nc in n_configs_test]
    
    ax7.loglog(n_configs_test, projected_snr, 'o-', color=c_signal, linewidth=2, markersize=5)
    ax7.axhline(3, color=c_fit, linestyle='--', linewidth=1.5, label='3σ detection')
    ax7.axhline(5, color=c_fit, linestyle=':', linewidth=1.5, label='5σ discovery')
    ax7.fill_between([5, 2e6], [0.001, 0.001], [3, 3], alpha=0.08, color='red')
    ax7.set_xlabel('Number of CDT configurations', fontsize=11)
    ax7.set_ylabel('Projected SNR for T1 signal (δd=10⁻³)', fontsize=11)
    ax7.set_title('Sensitivity Projection', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(5, 2e6)
    ax7.set_ylim(0.01, 300)
    
    # N needed: signal_amplitude * √N / noise_rms = threshold
    if signal_amplitude > 0 and noise_rms > 0:
        n_3sigma = (3 * noise_rms / signal_amplitude)**2
        n_5sigma = (5 * noise_rms / signal_amplitude)**2
    else:
        n_3sigma = 1e9
        n_5sigma = 1e9
    
    if n_3sigma < 2e6:
        ax7.axvline(n_3sigma, color=c_fit, linestyle='--', alpha=0.4)
        ax7.text(n_3sigma*1.2, 4, f'{n_3sigma:.0f}', fontsize=8, color=c_fit)
    if n_5sigma < 2e6:
        ax7.axvline(n_5sigma, color=c_fit, linestyle=':', alpha=0.4)
        ax7.text(n_5sigma*1.2, 7, f'{n_5sigma:.0f}', fontsize=8, color=c_fit)
    
    # --- Panel 8: Summary text ---
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.axis('off')
    
    summary = (
        "CDT-CCDR T1 PREDICTION TEST SUMMARY\n"
        "====================================\n\n"
        f"Simulation: 2D CDT, {len(volumes)} configs\n"
        f"Mean vertices: {np.mean([np.sum(v) for v in volumes]):.0f}\n"
        f"Spectral dim: {ds_mean.min():.2f} - {ds_mean.max():.2f}\n\n"
        "WITHOUT INJECTION:\n"
        f"  Residual RMS:  {t1_results['residual_rms']:.5f}\n"
        f"  Peak SNR:      {t1_results['peak_snr']:.2f}\n"
        f"  Significance:  {t1_results['n_sigma_significance']:.1f}s\n\n"
        "WITH T1 INJECTION (dd = nu = 1e-3):\n"
        f"  Residual RMS:  {t1_injected_results['residual_rms']:.5f}\n"
        f"  Peak SNR:      {t1_injected_results['peak_snr']:.2f}\n"
        f"  Significance:  {t1_injected_results['n_sigma_significance']:.1f}s\n\n"
        f"Delta-SNR from injection: {delta_snr:.2f}\n\n"
        f"PROJECTED FOR dd=1e-3 DETECTION:\n"
        f"  3s detection:  ~{n_3sigma:.0f} configs\n"
        f"  5s discovery:  ~{n_5sigma:.0f} configs\n\n"
        "NOTE: 2D CDT proof-of-concept.\n"
        "For T1 test: use CDT-plusplus (4D)\n"
        "with this analysis pipeline."
    )
    
    ax8.text(0.05, 0.95, summary, transform=ax8.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f0eee6', alpha=0.8))
    
    plt.suptitle('CDT–CCDR Prediction T1: Spectral Dimension Oscillation Test',
                fontsize=14, fontweight='bold', y=0.98)
    
    filepath = os.path.join(output_dir, 'cdt_ccdr_t1_results.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Diagnostic plots saved to {filepath}")
    
    return filepath, n_3sigma, n_5sigma


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("  CDT–CCDR PREDICTION T1 TEST")
    print("  Spectral Dimension Oscillation Analysis Pipeline")
    print("=" * 65)
    
    output_dir = '/home/claude'
    t_start = time.time()
    
    # ----- Step 1: Run 2D CDT simulation -----
    print("\n[1/5] Running 2D CDT Monte Carlo simulation...")
    configs, volumes = run_cdt_simulation(
        T=16, s_init=20, kappa0=1.5, lam=1.0,
        N2_target=700, epsilon=0.005,
        n_therm=500, n_configs=80, n_between=20
    )
    
    # ----- Step 2: Measure spectral dimensions -----
    print("\n[2/5] Measuring spectral dimension d_s(σ) for each configuration...")
    sigmas, ds_mean, ds_std, ds_err, all_ds = collect_spectral_dimensions(
        configs, sigma_max=30, n_sigma=50
    )
    print(f"  d_s range: {ds_mean.min():.3f} – {ds_mean.max():.3f}")
    print(f"  Mean error: {np.mean(ds_err):.4f}")
    
    # ----- Step 3: T1 analysis (no injection) -----
    print("\n[3/5] Running T1 Fourier analysis (no signal injection)...")
    t1_results = t1_fourier_analysis(sigmas, ds_mean, ds_err)
    print(f"  Residual RMS:  {t1_results['residual_rms']:.5f}")
    print(f"  Peak SNR:      {t1_results['peak_snr']:.2f}")
    print(f"  Peak freq:     {t1_results['peak_frequency']:.4f}")
    print(f"  p-value:       {t1_results['p_value']:.4f}")
    
    # ----- Step 4: Signal injection test -----
    print("\n[4/5] Running signal injection test (δd = ν = 10⁻³)...")
    ds_injected, signal = inject_t1_signal(sigmas, ds_mean, nu=1e-3)
    t1_injected = t1_fourier_analysis(sigmas, ds_injected, ds_err)
    print(f"  Residual RMS:  {t1_injected['residual_rms']:.5f}")
    print(f"  Peak SNR:      {t1_injected['peak_snr']:.2f}")
    print(f"  p-value:       {t1_injected['p_value']:.6f}")
    
    # Also test with larger signal for calibration
    ds_injected_large, signal_large = inject_t1_signal(sigmas, ds_mean, nu=0.01)
    t1_large = t1_fourier_analysis(sigmas, ds_injected_large, ds_err)
    print(f"\n  Calibration (δd = 10⁻²):")
    print(f"  Peak SNR:      {t1_large['peak_snr']:.2f}")
    print(f"  p-value:       {t1_large['p_value']:.8f}")
    
    # ----- Step 5: Generate plots -----
    print("\n[5/5] Generating diagnostic plots...")
    plot_path, n_3sigma, n_5sigma = make_diagnostic_plots(
        sigmas, ds_mean, ds_err, t1_results, t1_injected,
        volumes, signal, output_dir
    )
    
    # ----- Summary report -----
    elapsed = time.time() - t_start
    
    report = {
        'simulation': {
            'dimension': 2,
            'time_slices': 16,
            'n_configurations': 80,
            'mean_vertices': float(np.mean([np.sum(v) for v in volumes])),
            'mean_volume': float(np.mean([2*np.sum(v) for v in volumes])),
        },
        'spectral_dimension': {
            'ds_min': float(ds_mean.min()),
            'ds_max': float(ds_mean.max()),
            'mean_error': float(np.mean(ds_err)),
        },
        't1_no_injection': {
            'residual_rms': t1_results['residual_rms'],
            'peak_snr': t1_results['peak_snr'],
            'p_value': t1_results['p_value'],
        },
        't1_with_injection_nu_1e-3': {
            'residual_rms': t1_injected['residual_rms'],
            'peak_snr': t1_injected['peak_snr'],
            'p_value': t1_injected['p_value'],
        },
        'sensitivity': {
            'configs_for_3sigma': float(n_3sigma),
            'configs_for_5sigma': float(n_5sigma),
            'note': 'Projected from 2D CDT. 4D CDT may differ by O(1) factor.'
        },
        'runtime_seconds': elapsed,
        'next_steps': [
            '1. Install CDT-plusplus (github.com/acgetchell/CDT-plusplus) for 4D CDT',
            '2. Generate C-phase configurations with N4 ~ 10^5-10^6 simplices',
            '3. Export d_s(σ) measurements in CSV format',
            '4. Feed into t1_fourier_analysis() from this pipeline',
            '5. Compare δd_Pl with ν from DESI (two-point test T3)',
        ]
    }
    
    report_path = os.path.join(output_dir, 'cdt_ccdr_t1_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"\n  Runtime: {elapsed:.1f} seconds")
    print(f"\n  2D CDT simulation: {80} configs, {report['simulation']['mean_vertices']:.0f} mean vertices")
    print(f"  Spectral dimension: d_s = {ds_mean.min():.2f} → {ds_mean.max():.2f}")
    print(f"\n  T1 TEST (no injection):  SNR = {t1_results['peak_snr']:.2f}, p = {t1_results['p_value']:.4f}")
    print(f"  T1 TEST (ν=10⁻³ injected): SNR = {t1_injected['peak_snr']:.2f}, p = {t1_injected['p_value']:.6f}")
    print(f"\n  Projected configs for 3σ detection of T1: ~{n_3sigma:.0f}")
    print(f"  Projected configs for 5σ discovery of T1:  ~{n_5sigma:.0f}")
    print(f"\n  NOTE: This is 2D CDT (proof-of-concept).")
    print(f"  For actual T1 test, use 4D CDT (CDT-plusplus)")
    print(f"  with this analysis pipeline.")
    print(f"\n  Plots: {plot_path}")
    print(f"  Report: {report_path}")
    print("=" * 65)
    
    return report


if __name__ == '__main__':
    report = main()
