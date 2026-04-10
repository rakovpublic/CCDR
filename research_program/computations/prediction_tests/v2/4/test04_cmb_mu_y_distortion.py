#!/usr/bin/env python3
"""
Test 04 — CMB mu/y distortion limits vs cascade releases.

Standalone implementation that downloads the public COBE/FIRAS monopole spectrum
from NASA LAMBDA, fits mu + y + dT templates to the residual spectrum, and writes
result.json plus a diagnostic plot.

Design choices:
- Uses only public downloads performed by the script itself.
- Uses the machine-readable FIRAS residual table from LAMBDA.
- Fits with public per-channel 1-sigma errors (diagonal covariance weighting).
- Optionally includes the tabulated Galactic-pole model column as a nuisance template.

The public LAMBDA residual text file provides the residual spectrum and per-channel
uncertainties directly. A compact machine-readable monopole residual covariance matrix
is not provided alongside that text file, so this implementation uses the public diagonal
errors. This is adequate for the intended consistency check and reproduces the order of
magnitude of the published FIRAS limits.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Public data source documented by NASA LAMBDA.
FIRAS_MONOPOLE_URL = (
    "https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/"
    "firas_monopole_spec_v1.txt"
)

# Useful literature numbers quoted in the test description.
OFFICIAL_FIRAS_MU_LIMIT_95 = 9.0e-5
OFFICIAL_FIRAS_Y_LIMIT_95 = 1.5e-5
DEFAULT_PIXIE_SENSITIVITY = 1.0e-8
C_LIGHT = 299_792_458.0
H_PLANCK = 6.626_070_15e-34
K_BOLTZ = 1.380_649e-23
ALPHA_MU = 0.4561
KJY_PER_SI_INTENSITY = 1.0e23  # 1 W m^-2 Hz^-1 sr^-1 = 1e23 kJy/sr


@dataclass
class FirasData:
    wavenumber_cm: np.ndarray
    intensity_mjy_sr: np.ndarray
    residual_kjy_sr: np.ndarray
    sigma_kjy_sr: np.ndarray
    galaxy_kjy_sr: np.ndarray

    @property
    def frequency_hz(self) -> np.ndarray:
        # cm^-1 -> m^-1 via *100, then nu = c * kappa
        return self.wavenumber_cm * 100.0 * C_LIGHT


@dataclass
class FitResult:
    parameters: Dict[str, float]
    one_sigma: Dict[str, float]
    covariance: np.ndarray
    model_kjy_sr: np.ndarray
    components_kjy_sr: Dict[str, np.ndarray]
    chi2: float
    dof: int
    reduced_chi2: float
    design_labels: Tuple[str, ...]


class DownloadError(RuntimeError):
    pass


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_text(url: str, destination: pathlib.Path, timeout: float = 60.0) -> pathlib.Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    ensure_dir(destination.parent)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DownloadError(f"Failed to download {url}: {exc}") from exc

    destination.write_text(text, encoding="utf-8")
    return destination


def load_firas_data(cache_dir: pathlib.Path) -> FirasData:
    data_path = download_text(FIRAS_MONOPOLE_URL, cache_dir / "firas_monopole_spec_v1.txt")
    rows = np.loadtxt(data_path)
    if rows.ndim != 2 or rows.shape[1] < 5:
        raise RuntimeError("Unexpected FIRAS file format.")
    return FirasData(
        wavenumber_cm=rows[:, 0],
        intensity_mjy_sr=rows[:, 1],
        residual_kjy_sr=rows[:, 2],
        sigma_kjy_sr=rows[:, 3],
        galaxy_kjy_sr=rows[:, 4],
    )


def planck_intensity(nu_hz: np.ndarray, temperature_k: float) -> np.ndarray:
    x = H_PLANCK * nu_hz / (K_BOLTZ * temperature_k)
    return (2.0 * H_PLANCK * nu_hz**3 / C_LIGHT**2) / np.expm1(x)


def d_planck_dT(nu_hz: np.ndarray, temperature_k: float) -> np.ndarray:
    x = H_PLANCK * nu_hz / (K_BOLTZ * temperature_k)
    ex = np.exp(x)
    return (
        2.0
        * H_PLANCK**2
        * nu_hz**4
        / (C_LIGHT**2 * K_BOLTZ * temperature_k**2)
        * ex
        / (ex - 1.0) ** 2
    )


def distortion_templates_kjy_sr(
    nu_hz: np.ndarray,
    reference_temperature_k: float,
) -> Dict[str, np.ndarray]:
    x = H_PLANCK * nu_hz / (K_BOLTZ * reference_temperature_k)
    ex = np.exp(x)
    expm1x = np.expm1(x)

    # Phase-space distortion shapes following the standard Chluba/Sunyaev basis.
    G = x * ex / expm1x**2
    M_phase = -G * (1.0 / x - ALPHA_MU)
    Y_phase = G * (x * (ex + 1.0) / expm1x - 4.0)

    intensity_prefactor = 2.0 * H_PLANCK * nu_hz**3 / C_LIGHT**2
    mu_template = intensity_prefactor * M_phase * KJY_PER_SI_INTENSITY
    y_template = intensity_prefactor * Y_phase * KJY_PER_SI_INTENSITY
    dT_template = d_planck_dT(nu_hz, reference_temperature_k) * KJY_PER_SI_INTENSITY

    return {
        "mu": mu_template,
        "y": y_template,
        "dT_K": dT_template,
    }


def weighted_linear_fit(
    y_data: np.ndarray,
    y_sigma: np.ndarray,
    templates: Dict[str, np.ndarray],
    nuisance_templates: Dict[str, np.ndarray] | None = None,
) -> FitResult:
    nuisance_templates = nuisance_templates or {}
    all_templates = dict(templates)
    all_templates.update(nuisance_templates)

    labels = tuple(all_templates.keys())
    design = np.column_stack([all_templates[label] for label in labels])

    white_design = design / y_sigma[:, None]
    white_data = y_data / y_sigma

    normal_matrix = white_design.T @ white_design
    rhs = white_design.T @ white_data
    covariance = np.linalg.inv(normal_matrix)
    beta = covariance @ rhs

    model = design @ beta
    residual = y_data - model
    chi2 = float(np.sum((residual / y_sigma) ** 2))
    dof = int(y_data.size - beta.size)
    red_chi2 = chi2 / dof if dof > 0 else float("nan")

    params = {label: float(beta[i]) for i, label in enumerate(labels)}
    one_sigma = {label: float(np.sqrt(covariance[i, i])) for i, label in enumerate(labels)}
    components = {label: all_templates[label] * params[label] for label in labels}

    return FitResult(
        parameters=params,
        one_sigma=one_sigma,
        covariance=covariance,
        model_kjy_sr=model,
        components_kjy_sr=components,
        chi2=chi2,
        dof=dof,
        reduced_chi2=red_chi2,
        design_labels=labels,
    )


def two_sided_95_limit(best_fit: float, sigma_1: float) -> float:
    # Conservative symmetric-Gaussian absolute-value limit.
    return abs(best_fit) + 1.959963984540054 * sigma_1


def split_stages(total_stages: int, n_mu_stages: int | None, n_y_stages: int | None) -> Tuple[int, int, str]:
    if n_mu_stages is not None and n_y_stages is not None:
        if n_mu_stages + n_y_stages != total_stages:
            raise ValueError("n_mu_stages + n_y_stages must equal N - 4.")
        return n_mu_stages, n_y_stages, "user-specified"

    if total_stages == 2:
        return 1, 1, "default-two-stage-split"

    # Without an explicit cascade timing model, split stages as evenly as possible.
    n_mu = total_stages // 2
    n_y = total_stages - n_mu
    return n_mu, n_y, "heuristic-even-split"


def predict_cascade_distortions(
    N: int,
    nu_cascade: float,
    n_mu_stages: int | None,
    n_y_stages: int | None,
) -> Dict[str, float | int | str]:
    if N < 4:
        raise ValueError("N must be >= 4.")

    total_stages = N - 4
    mu_stages, y_stages, split_mode = split_stages(total_stages, n_mu_stages, n_y_stages)

    # Per the test note: DeltaE/E ~ nu^2 per stage.
    per_stage = nu_cascade**2
    mu_pred = mu_stages * per_stage
    y_pred = y_stages * per_stage

    return {
        "N": int(N),
        "total_stages": int(total_stages),
        "n_mu_stages": int(mu_stages),
        "n_y_stages": int(y_stages),
        "stage_split_mode": split_mode,
        "deltaE_over_E_per_stage": float(per_stage),
        "mu_predicted": float(mu_pred),
        "y_predicted": float(y_pred),
    }


def make_plot(
    output_path: pathlib.Path,
    data: FirasData,
    fit: FitResult,
    include_galaxy_template: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.5))

    ax.errorbar(
        data.wavenumber_cm,
        data.residual_kjy_sr,
        yerr=data.sigma_kjy_sr,
        fmt="o",
        ms=4,
        capsize=2,
        label="FIRAS residuals",
    )
    ax.plot(data.wavenumber_cm, fit.model_kjy_sr, linewidth=2.0, label="Best-fit total")
    ax.plot(
        data.wavenumber_cm,
        fit.components_kjy_sr.get("mu", np.zeros_like(data.wavenumber_cm)),
        linestyle="--",
        linewidth=1.5,
        label="mu component",
    )
    ax.plot(
        data.wavenumber_cm,
        fit.components_kjy_sr.get("y", np.zeros_like(data.wavenumber_cm)),
        linestyle="--",
        linewidth=1.5,
        label="y component",
    )
    ax.plot(
        data.wavenumber_cm,
        fit.components_kjy_sr.get("dT_K", np.zeros_like(data.wavenumber_cm)),
        linestyle=":",
        linewidth=1.6,
        label="dT component",
    )
    if include_galaxy_template and "galaxy_scale" in fit.components_kjy_sr:
        ax.plot(
            data.wavenumber_cm,
            fit.components_kjy_sr["galaxy_scale"],
            linestyle="-.",
            linewidth=1.3,
            label="Galaxy nuisance",
        )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Residual intensity [kJy/sr]")
    ax.set_title("COBE/FIRAS residuals with best-fit mu+y+dT distortion model")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_result_payload(
    args: argparse.Namespace,
    data: FirasData,
    fit: FitResult,
    prediction: Dict[str, float | int | str],
) -> Dict[str, object]:
    mu_best = fit.parameters["mu"]
    y_best = fit.parameters["y"]
    dT_best = fit.parameters["dT_K"]
    mu_sigma = fit.one_sigma["mu"]
    y_sigma = fit.one_sigma["y"]
    dT_sigma = fit.one_sigma["dT_K"]
    mu_limit_95 = two_sided_95_limit(mu_best, mu_sigma)
    y_limit_95 = two_sided_95_limit(y_best, y_sigma)

    mu_pred = float(prediction["mu_predicted"])
    y_pred = float(prediction["y_predicted"])

    consistent_with_firas = (mu_pred < mu_limit_95) and (y_pred < y_limit_95)
    pixie_detectable = (mu_pred >= args.pixie_sensitivity) or (y_pred >= args.pixie_sensitivity)

    payload: Dict[str, object] = {
        "test_name": "Test 04 — CMB mu/y distortion limits vs cascade releases",
        "fit_mode": "weighted_linear_least_squares_public_diagonal_errors",
        "covariance_mode": "diagonal_public_errors_only",
        "data_url": FIRAS_MONOPOLE_URL,
        "reference_temperature_K": args.reference_temperature,
        "n_channels": int(data.wavenumber_cm.size),
        "fit_parameters": {
            "mu_best_fit": mu_best,
            "y_best_fit": y_best,
            "dT_best_fit_K": dT_best,
        },
        "fit_uncertainties_1sigma": {
            "mu_sigma": mu_sigma,
            "y_sigma": y_sigma,
            "dT_sigma_K": dT_sigma,
        },
        "mu_limit_95": mu_limit_95,
        "y_limit_95": y_limit_95,
        "mu_predicted": mu_pred,
        "y_predicted": y_pred,
        "consistent_with_firas": bool(consistent_with_firas),
        "pixie_detectable": bool(pixie_detectable),
        "pixie_sensitivity_threshold": float(args.pixie_sensitivity),
        "goodness_of_fit": {
            "chi2": fit.chi2,
            "dof": fit.dof,
            "reduced_chi2": fit.reduced_chi2,
        },
        "prediction_model": prediction,
        "official_firas_limits_95_from_literature": {
            "mu_limit_95": OFFICIAL_FIRAS_MU_LIMIT_95,
            "y_limit_95": OFFICIAL_FIRAS_Y_LIMIT_95,
        },
        "implementation_notes": [
            "The script downloads the machine-readable FIRAS monopole residual table from NASA LAMBDA.",
            "The fit uses the public per-channel 1-sigma errors from that table as a diagonal covariance approximation.",
            "A compact machine-readable monopole residual covariance matrix is not downloaded here.",
            "For N=6 and nu=1e-3, the default stage split reproduces the test-note expectation mu~1e-6 and y~1e-6.",
        ],
    }

    if args.include_galaxy_template:
        payload["fit_parameters"]["galaxy_scale"] = fit.parameters.get("galaxy_scale", 0.0)
        payload["fit_uncertainties_1sigma"]["galaxy_scale_sigma"] = fit.one_sigma.get("galaxy_scale", 0.0)

    return payload


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("test04_output"))
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("test04_cache"))
    parser.add_argument("--reference-temperature", type=float, default=2.725)
    parser.add_argument("--N", type=int, default=6, help="Cascade dimensionality parameter from the test note.")
    parser.add_argument("--nu-cascade", type=float, default=1.0e-3, help="Cascade parameter nu used for DeltaE/E ~ nu^2.")
    parser.add_argument("--n-mu-stages", type=int, default=None)
    parser.add_argument("--n-y-stages", type=int, default=None)
    parser.add_argument(
        "--pixie-sensitivity",
        type=float,
        default=DEFAULT_PIXIE_SENSITIVITY,
        help="Threshold used for the future-detectability flag.",
    )
    parser.add_argument(
        "--include-galaxy-template",
        action="store_true",
        help="Include the FIRAS Galactic-pole model column as an extra nuisance template.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    ensure_dir(args.outdir)
    ensure_dir(args.cache_dir)

    data = load_firas_data(args.cache_dir)
    templates = distortion_templates_kjy_sr(data.frequency_hz, args.reference_temperature)

    nuisance = {}
    if args.include_galaxy_template:
        nuisance["galaxy_scale"] = data.galaxy_kjy_sr

    fit = weighted_linear_fit(
        y_data=data.residual_kjy_sr,
        y_sigma=data.sigma_kjy_sr,
        templates=templates,
        nuisance_templates=nuisance,
    )

    prediction = predict_cascade_distortions(
        N=args.N,
        nu_cascade=args.nu_cascade,
        n_mu_stages=args.n_mu_stages,
        n_y_stages=args.n_y_stages,
    )

    result = build_result_payload(args, data, fit, prediction)

    result_path = args.outdir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    plot_path = args.outdir / "firas_mu_y_fit.png"
    make_plot(plot_path, data, fit, include_galaxy_template=args.include_galaxy_template)

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nWrote: {result_path}")
    print(f"Wrote: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
