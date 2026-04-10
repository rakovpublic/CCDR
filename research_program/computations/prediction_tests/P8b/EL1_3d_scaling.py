#!/usr/bin/env python3
"""EL1_3d_scaling.py — test area vs volume scaling in 3D NAND throughput.

Implements the EL1 note as a runnable script with:
  - built-in public benchmark-style dataset from the note
  - closed-form least-squares fits for area and volume hypotheses
  - log-space power-law fit T ~ a * L^alpha
  - residual/AIC/BIC/R^2 comparison
  - optional CSV, JSON, and plot outputs

No SciPy dependency is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DEFAULT_DATA = np.array([
    [32,  40,  256],
    [48,  50,  512],
    [64,  60,  512],
    [96,  80,  1024],
    [128, 100, 2048],
    [176, 130, 2048],
    [232, 160, 4096],
    [300, 200, 4096],
], dtype=float)


@dataclass
class FitResult:
    name: str
    a: float
    b: float | None
    exponent: float | None
    sse: float
    rmse: float
    r2: float
    aic: float
    bic: float


@dataclass
class AnalysisResult:
    preferred_model: str
    area_fit: FitResult
    volume_fit: FitResult
    power_law_fit: FitResult
    alpha_interpretation: str
    layers: list[float]
    throughput_mb_s: list[float]
    capacity_gb: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test area vs volume scaling in 3D NAND throughput."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help=(
            "Optional CSV with columns: layers, throughput_mb_s, capacity_gb. "
            "If omitted, the built-in dataset from the EL1 note is used."
        ),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("EL1_3d_scaling.png"),
        help="Output path for the comparison plot. Use --no-plot to disable.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("EL1_3d_scaling_results.json"),
        help="Path to save machine-readable results.",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        help="Optional path to export the active dataset as CSV.",
    )
    return parser.parse_args()


def load_csv_dataset(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"layers", "throughput_mb_s", "capacity_gb"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            rows.append([
                float(row["layers"]),
                float(row["throughput_mb_s"]),
                float(row["capacity_gb"]),
            ])
    if not rows:
        raise ValueError("Input CSV had no usable rows")
    data = np.array(rows, dtype=float)
    if np.any(data[:, 0] <= 0) or np.any(data[:, 1] <= 0) or np.any(data[:, 2] <= 0):
        raise ValueError("All layers, throughput, and capacity values must be positive")
    return data


def export_csv_dataset(path: Path, data: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layers", "throughput_mb_s", "capacity_gb"])
        writer.writerows(data.tolist())


def linear_least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([x, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    a, b = coeffs
    return float(a), float(b)


def model_metrics(y_true: np.ndarray, y_pred: np.ndarray, k_params: int) -> tuple[float, float, float, float, float]:
    residuals = y_true - y_pred
    sse = float(np.sum(residuals ** 2))
    n = len(y_true)
    rmse = float(math.sqrt(sse / n))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")

    # Use the Gaussian-likelihood AIC/BIC form up to an additive constant.
    eps = 1e-15
    aic = n * math.log(max(sse / n, eps)) + 2 * k_params
    bic = n * math.log(max(sse / n, eps)) + k_params * math.log(n)
    return sse, rmse, r2, aic, bic


def fit_area_model(layers: np.ndarray, throughput: np.ndarray) -> tuple[FitResult, np.ndarray]:
    a, b = linear_least_squares(layers, throughput)
    pred = a * layers + b
    sse, rmse, r2, aic, bic = model_metrics(throughput, pred, k_params=2)
    result = FitResult(
        name="area_model",
        a=a,
        b=b,
        exponent=1.0,
        sse=sse,
        rmse=rmse,
        r2=r2,
        aic=aic,
        bic=bic,
    )
    return result, pred


def fit_volume_model(layers: np.ndarray, throughput: np.ndarray) -> tuple[FitResult, np.ndarray]:
    transformed = layers ** 1.5
    a, b = linear_least_squares(transformed, throughput)
    pred = a * transformed + b
    sse, rmse, r2, aic, bic = model_metrics(throughput, pred, k_params=2)
    result = FitResult(
        name="volume_model",
        a=a,
        b=b,
        exponent=1.5,
        sse=sse,
        rmse=rmse,
        r2=r2,
        aic=aic,
        bic=bic,
    )
    return result, pred


def fit_power_law(layers: np.ndarray, throughput: np.ndarray) -> tuple[FitResult, np.ndarray]:
    # Fit log(T) = log(a) + alpha log(L)
    lx = np.log(layers)
    ly = np.log(throughput)
    alpha, log_a = linear_least_squares(lx, ly)
    a = math.exp(log_a)
    pred = a * np.power(layers, alpha)
    sse, rmse, r2, aic, bic = model_metrics(throughput, pred, k_params=2)
    result = FitResult(
        name="power_law",
        a=a,
        b=None,
        exponent=alpha,
        sse=sse,
        rmse=rmse,
        r2=r2,
        aic=aic,
        bic=bic,
    )
    return result, pred


def interpret_alpha(alpha: float) -> str:
    if abs(alpha - 1.0) <= 0.15:
        return "close to area scaling (alpha ≈ 1.0)"
    if abs(alpha - 1.5) <= 0.15:
        return "close to volume-like scaling (alpha ≈ 1.5)"
    if alpha < 1.0:
        return "sub-linear with layer count"
    if 1.0 < alpha < 1.5:
        return "between area and volume-like scaling"
    return "super-volume-like relative to the toy hypotheses"


def choose_preferred_model(area_fit: FitResult, volume_fit: FitResult) -> str:
    if area_fit.sse < volume_fit.sse:
        return "area_model"
    if volume_fit.sse < area_fit.sse:
        return "volume_model"
    return "tie"


def analyze(data: np.ndarray) -> tuple[AnalysisResult, dict[str, np.ndarray]]:
    layers = data[:, 0]
    throughput = data[:, 1]
    capacity = data[:, 2]

    area_fit, area_pred = fit_area_model(layers, throughput)
    volume_fit, volume_pred = fit_volume_model(layers, throughput)
    power_fit, power_pred = fit_power_law(layers, throughput)

    result = AnalysisResult(
        preferred_model=choose_preferred_model(area_fit, volume_fit),
        area_fit=area_fit,
        volume_fit=volume_fit,
        power_law_fit=power_fit,
        alpha_interpretation=interpret_alpha(power_fit.exponent or float("nan")),
        layers=layers.tolist(),
        throughput_mb_s=throughput.tolist(),
        capacity_gb=capacity.tolist(),
    )
    preds = {
        "area": area_pred,
        "volume": volume_pred,
        "power": power_pred,
    }
    return result, preds


def print_summary(result: AnalysisResult) -> None:
    print("3D NAND Throughput Scaling")
    print("=" * 40)
    print(f"Area model   (T ~ L):     SSE = {result.area_fit.sse:.2f}, RMSE = {result.area_fit.rmse:.2f}, R² = {result.area_fit.r2:.4f}")
    print(f"Volume model (T ~ L^1.5): SSE = {result.volume_fit.sse:.2f}, RMSE = {result.volume_fit.rmse:.2f}, R² = {result.volume_fit.r2:.4f}")
    print()
    print(f"Power law fit: T ~ a * L^α with α = {result.power_law_fit.exponent:.4f}")
    print(f"Interpretation: {result.alpha_interpretation}")
    print()

    if result.preferred_model == "area_model":
        print("✓ AREA scaling preferred under the two-hypothesis comparison")
    elif result.preferred_model == "volume_model":
        print("✗ VOLUME scaling preferred under the two-hypothesis comparison")
    else:
        print("• Area and volume models tie on SSE")

    print()
    print("Model comparison (lower is better):")
    print(f"  AIC area   = {result.area_fit.aic:.3f}")
    print(f"  AIC volume = {result.volume_fit.aic:.3f}")
    print(f"  BIC area   = {result.area_fit.bic:.3f}")
    print(f"  BIC volume = {result.volume_fit.bic:.3f}")


def save_json(path: Path, result: AnalysisResult) -> None:
    payload = {
        "preferred_model": result.preferred_model,
        "alpha_interpretation": result.alpha_interpretation,
        "layers": result.layers,
        "throughput_mb_s": result.throughput_mb_s,
        "capacity_gb": result.capacity_gb,
        "area_fit": asdict(result.area_fit),
        "volume_fit": asdict(result.volume_fit),
        "power_law_fit": asdict(result.power_law_fit),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def maybe_plot(path: Path, data: np.ndarray, preds: dict[str, np.ndarray], result: AnalysisResult) -> None:
    if plt is None:
        print("[warn] matplotlib not available; skipping plot")
        return

    layers = data[:, 0]
    throughput = data[:, 1]
    capacity = data[:, 2]

    x_grid = np.linspace(np.min(layers), np.max(layers), 300)
    area_curve = result.area_fit.a * x_grid + (result.area_fit.b or 0.0)
    volume_curve = result.volume_fit.a * x_grid ** 1.5 + (result.volume_fit.b or 0.0)
    power_curve = result.power_law_fit.a * x_grid ** (result.power_law_fit.exponent or 1.0)

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.scatter(layers, throughput, label="Throughput data", s=60)
    ax1.plot(x_grid, area_curve, label="Area model: T ~ L")
    ax1.plot(x_grid, volume_curve, label="Volume model: T ~ L^1.5")
    ax1.plot(x_grid, power_curve, label=f"Power law: α={result.power_law_fit.exponent:.2f}")
    ax1.set_xlabel("Number of layers")
    ax1.set_ylabel("Sequential read throughput (MB/s)")
    ax1.set_title("EL1: 3D NAND throughput scaling")

    ax2 = ax1.twinx()
    ax2.plot(layers, capacity, linestyle="--", marker="s", alpha=0.7, label="Capacity")
    ax2.set_ylabel("Capacity (GB)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    try:
        data = load_csv_dataset(args.input_csv) if args.input_csv else DEFAULT_DATA.copy()
    except Exception as exc:
        print(f"[error] failed to load dataset: {exc}")
        return 2

    if args.export_csv:
        export_csv_dataset(args.export_csv, data)

    result, preds = analyze(data)
    print_summary(result)

    try:
        save_json(args.json_output, result)
        print(f"Saved JSON: {args.json_output}")
    except Exception as exc:
        print(f"[warn] failed to write JSON: {exc}")

    if not args.no_plot:
        try:
            maybe_plot(args.plot, data, preds, result)
            if plt is not None:
                print(f"Saved plot: {args.plot}")
        except Exception as exc:
            print(f"[warn] failed to generate plot: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
