from __future__ import annotations

"""
EL1 public-source test, strict edition.

The previous EL1 script mixed incompatible performance metrics. This version makes
that impossible to over-interpret by splitting EL1 into three layers:

1) curated_note_benchmark
   Reproduces the EL1 note's hand-curated sequential-read MB/s series. This is a
   hypothesis check on the note's own dataset, not an independent scrape.

2) live_strict_same_physics_groups
   Only fits live-scraped records when they share both unit and metric class, e.g.
   NAND interface Gbps vs NAND interface Gbps. Mixed drive MB/s, MT/s and NAND
   interface Gbps are NEVER pooled.

3) live_mixed_audit
   Reports all recovered public points and explains why they are invalid as a
   single regression if metric classes/units differ.

Verdict policy:
- A curated-only area result is labelled curated_support_only.
- Live confirmation requires at least one valid strict live group with >=4 points,
  same metric class, same unit, and area beating L^1.5 by AIC.
- If a strict live group prefers L^1.5 or has too few points, EL1 remains
  inconclusive, not falsified, because the scraped metrics are still proxies.
"""

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from _common_public_data import (
    ensure_dir, extract_float, fit_linear, fit_powerlaw, json_dump, save_plot,
    structured_report, download_text,
)

ROOT = Path(__file__).resolve().parent
OUT = ensure_dir(ROOT / "outputs" / "el1")

# The EL1 note's benchmark-style dataset. Kept only as a reproducibility check.
CURATED_NOTE_DATA = [
    {"label": "32-layer benchmark-style", "layers": 32.0, "value": 40.0, "unit": "MB/s", "metric_class": "curated_sequential_read", "source_type": "curated_note"},
    {"label": "48-layer benchmark-style", "layers": 48.0, "value": 50.0, "unit": "MB/s", "metric_class": "curated_sequential_read", "source_type": "curated_note"},
    {"label": "64-layer benchmark-style", "layers": 64.0, "value": 60.0, "unit": "MB/s", "metric_class": "curated_sequential_read", "source_type": "curated_note"},
    {"label": "96-layer benchmark-style", "layers": 96.0, "value": 80.0, "unit": "MB/s", "metric_class": "curated_sequential_read", "source_type": "curated_note"},
    {"label": "128-layer benchmark-style", "layers": 128.0, "value": 100.0, "unit": "MB/s", "metric_class": "curated_sequential_read", "source_type": "curated_note"},
    {"label": "176-layer benchmark-style", "layers": 176.0, "value": 130.0, "unit": "MB/s", "metric_class": "curated_sequential_read", "source_type": "curated_note"},
    {"label": "232-layer benchmark-style", "layers": 232.0, "value": 160.0, "unit": "MB/s", "metric_class": "curated_sequential_read", "source_type": "curated_note"},
    {"label": "300-layer benchmark-style", "layers": 300.0, "value": 200.0, "unit": "MB/s", "metric_class": "curated_sequential_read", "source_type": "curated_note"},
]

# Public sources. metric_class is deliberately explicit; regression grouping uses it.
SOURCES = [
    {
        "label": "Micron 3400 SSD product brief",
        "url": "https://assets.micron.com/adobe/assets/urn%3Aaaid%3Aaem%3Ab1ec169c-b345-4593-a2da-3a3b2a94eca0/renditions/original/as/micron-3400-ssd-product-brief.pdf",
        "layer_patterns": [r"(176)-layer"],
        "value_patterns": [r"Sequential Read \(MB/s\)\D+([0-9,]{3,6})", r"sequential read[^0-9]{0,80}([0-9,]{3,6})\s*MB/s"],
        "unit": "MB/s",
        "metric_class": "ssd_sequential_read",
    },
    {
        "label": "Micron 5400 SSD product brief",
        "url": "https://assets.micron.com/adobe/assets/urn%3Aaaid%3Aaem%3A0eed6388-5671-428a-9231-6e83459c86ea/renditions/original/as/5400-product-brief.pdf",
        "layer_patterns": [r"(176)-layer"],
        "value_patterns": [r"Seq\. Read\s*\(MB/s\)\D+([0-9,]{3,6})", r"sequential read[^0-9]{0,80}([0-9,]{3,6})\s*MB/s"],
        "unit": "MB/s",
        "metric_class": "ssd_sequential_read",
    },
    {
        "label": "Micron UFS 4.0",
        "url": "https://www.micron.com/about/blog/company/innovations/micron-delivers-worlds-fastest-ufs-4-0-for-flagship-smartphones",
        "layer_patterns": [r"(232)-layer"],
        "value_patterns": [r"With ([0-9,]{3,6}) megabytes per second \(MBps\) sequential read"],
        "unit": "MB/s",
        "metric_class": "ufs_device_sequential_read",
    },
    {
        "label": "Samsung 9th-gen V-NAND",
        "url": "https://semiconductor.samsung.com/news-events/news/samsung-electronics-begins-industrys-first-mass-production-of-9th-gen-v-nand/",
        "layer_patterns": [r"([0-9]{3})-layer V-NAND", r"9th-generation V-NAND.*?([0-9]{3})-layer", r"double-stack.*?([0-9]{3})"],
        "layer_fallback": 286.0,
        "value_patterns": [r"up to ([0-9.]+) gigabits-per-second \(Gbps\)", r"up to ([0-9.]+) Gbps"],
        "unit": "Gbps",
        "metric_class": "nand_interface_speed",
    },
    {
        "label": "KIOXIA Gen-8 announcement",
        "url": "https://apac.kioxia.com/en-apac/business/news/2023/20230330-1.html",
        "layer_patterns": [r"(218)-layer"],
        "value_patterns": [r"over ([0-9.]+)Gb/s", r"over ([0-9.]+) Gb/s"],
        "unit": "Gbps",
        "metric_class": "nand_interface_speed",
    },
    {
        "label": "KIOXIA generation-10 history page",
        "url": "https://www.kioxia.com/en-jp/rd/technology/history.html",
        "layer_patterns": [r"(332)-layer"],
        "value_patterns": [r"Achieving ([0-9.]+)Gb/s NAND Interface Speed", r"Achieving ([0-9.]+) Gb/s NAND Interface Speed"],
        "unit": "Gbps",
        "metric_class": "nand_interface_speed",
    },
    {
        "label": "KIOXIA CBA technology page",
        "url": "https://www.kioxia.com/en-jp/business/topics/bics-cba-202407.html",
        "layer_patterns": [r"consisting of (162) layers", r"(162) layers"],
        "value_patterns": [r"Interface speed is ([0-9.]+) Gbps", r"Interface speed is ([0-9.]+)Gbps"],
        "unit": "Gbps",
        "metric_class": "nand_interface_speed",
    },
    {
        "label": "Micron ships world first 176-layer NAND",
        "url": "https://investors.micron.com/news-releases/news-release-details/micron-ships-worlds-first-176-layer-nand-delivering-breakthrough",
        "layer_patterns": [r"(176)-layer"],
        "value_patterns": [r"maximum data transfer rate at ([0-9,]{3,6}) megatransfers per second", r"([0-9,]{3,6}) MT/s"],
        "unit": "MT/s",
        "metric_class": "nand_io_transfer_rate_mtps",
    },
]


def extract_record(src: dict[str, Any]) -> dict[str, Any] | None:
    text = download_text(src["url"])
    layer = extract_float(src.get("layer_patterns", []), text)
    if layer is None:
        layer = src.get("layer_fallback")
    value = extract_float(src.get("value_patterns", []), text)
    if layer is None or value is None:
        return None
    return {
        "label": src["label"],
        "url": src["url"],
        "layers": float(layer),
        "value": float(value),
        "unit": src["unit"],
        "metric_class": src["metric_class"],
        "source_type": "live_vendor",
    }


def _fit_payload(fit: Any) -> dict[str, Any]:
    return {"params": fit.params, "sse": fit.sse, "rmse": fit.rmse, "r2": fit.r2, "aic": fit.aic, "bic": fit.bic}


def analyze_records(records: list[dict[str, Any]], tag: str, min_points_for_live_verdict: int = 4) -> dict[str, Any]:
    unit_set = sorted({r["unit"] for r in records})
    metric_set = sorted({r["metric_class"] for r in records})
    valid_same_physics = len(unit_set) == 1 and len(metric_set) == 1

    x = np.array([r["layers"] for r in records], dtype=float)
    y = np.array([r["value"] for r in records], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    ordered_records = [records[i] for i in order]

    result: dict[str, Any] = {
        "tag": tag,
        "unit": unit_set[0] if len(unit_set) == 1 else "mixed",
        "metric_class": metric_set[0] if len(metric_set) == 1 else "mixed",
        "n_points": int(len(x)),
        "valid_same_physics_group": bool(valid_same_physics),
        "records": ordered_records,
    }

    if not valid_same_physics:
        result.update({
            "status": "invalid_mixed_physics_group",
            "reason": "records differ by unit and/or metric_class; no regression is scientifically valid",
        })
        return result
    if len(x) < 3:
        result.update({"status": "insufficient_points", "reason": "need at least 3 points for toy model comparison"})
        return result

    linear = fit_linear(x, y)
    power = fit_powerlaw(x, y)
    vol = fit_linear(x ** 1.5, y)

    preferred = min([("linear", linear.aic), ("volume_reference_L1p5", vol.aic), ("powerlaw", power.aic)], key=lambda t: t[1])[0]
    area_vs_volume = "area_preferred" if linear.aic < vol.aic else "volume_reference_preferred"
    delta_aic_area_minus_volume = float(linear.aic - vol.aic)
    alpha = power.params.get("alpha", float("nan"))
    alpha_interpretation = (
        "sublinear" if alpha < 1.0 else
        "area_like" if abs(alpha - 1.0) <= 0.15 else
        "between_area_and_volume" if alpha < 1.5 else
        "volume_or_supervolume_like"
    )

    if tag.startswith("live") and len(x) < min_points_for_live_verdict:
        confirmation_level = "insufficient_live_points_for_confirmation"
        verdict = "no_live_physics_verdict"
    elif area_vs_volume == "area_preferred" and alpha < 1.5:
        confirmation_level = "support_like" if tag.startswith("live") else "curated_support_only"
        verdict = "support_like_area_constraint"
    elif area_vs_volume == "volume_reference_preferred":
        confirmation_level = "negative_or_mixed_proxy"
        verdict = "does_not_support_area_constraint_in_this_group"
    else:
        confirmation_level = "inconclusive"
        verdict = "inconclusive"

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(x, y, s=50, label="points")
    xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ax.plot(xx, linear.params["a"] * xx + linear.params["b"], label="linear in L")
    ax.plot(xx, power.params["a"] * xx ** power.params["alpha"], label=f"power law α={power.params['alpha']:.2f}")
    ax.plot(xx, vol.params["a"] * (xx ** 1.5) + vol.params["b"], label="L^1.5 reference")
    ax.set_xlabel("Layer count")
    ax.set_ylabel(f"Performance proxy ({result['unit']})")
    ax.set_title(f"EL1 {tag}: {result['metric_class']}")
    ax.grid(alpha=0.25)
    ax.legend()
    safe_tag = f"{tag}_{result['metric_class']}_{result['unit']}".replace("/", "_").replace(" ", "_")
    save_plot(fig, OUT / f"el1_{safe_tag}.png")
    plt.close(fig)

    result.update({
        "status": "ok",
        "linear": _fit_payload(linear),
        "volume_reference_L1p5": _fit_payload(vol),
        "powerlaw": _fit_payload(power),
        "preferred_by_aic": preferred,
        "area_vs_volume_aic": area_vs_volume,
        "delta_aic_area_minus_volume": delta_aic_area_minus_volume,
        "alpha": alpha,
        "alpha_interpretation": alpha_interpretation,
        "verdict": verdict,
        "confirmation_level": confirmation_level,
    })
    return result


def main() -> None:
    recovered: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for src in SOURCES:
        try:
            rec = extract_record(src)
            if rec is None:
                failures.append({"label": src["label"], "url": src["url"], "reason": "pattern_not_found"})
            else:
                recovered.append(rec)
        except Exception as e:
            reason = repr(e)
            failures.append({"label": src["label"], "url": src["url"], "reason": reason})
            # Avoid slow cascades when DNS/network is globally unavailable.
            if "NameResolutionError" in reason or "Failed to resolve" in reason:
                failures.append({
                    "label": "live_scrape_aborted",
                    "url": None,
                    "reason": "network/DNS unavailable after first failed live source; curated benchmark still reported, live scrape marked incomplete",
                })
                break

    curated_analysis = analyze_records(CURATED_NOTE_DATA, "curated_note_benchmark", min_points_for_live_verdict=4)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in recovered:
        grouped.setdefault((rec["unit"], rec["metric_class"]), []).append(rec)

    strict_live_analyses = []
    for (_unit, _metric), records in sorted(grouped.items()):
        strict_live_analyses.append(analyze_records(records, "live_strict_same_physics", min_points_for_live_verdict=4))

    # Deliberately invalid pooled audit, included to prevent accidental overuse.
    mixed_audit = analyze_records(recovered, "live_mixed_audit") if recovered else {
        "tag": "live_mixed_audit", "status": "no_records_recovered"
    }

    live_support = any(a.get("confirmation_level") == "support_like" for a in strict_live_analyses)
    live_negative = any(a.get("verdict") == "does_not_support_area_constraint_in_this_group" for a in strict_live_analyses)
    live_sufficient = any(a.get("n_points", 0) >= 4 and a.get("status") == "ok" for a in strict_live_analyses)
    curated_support = curated_analysis.get("confirmation_level") == "curated_support_only"

    if live_support and curated_support:
        overall = "support_like_but_still_proxy"
    elif curated_support and live_negative:
        overall = "mixed_evidence_curated_support_live_strict_not_confirming"
    elif curated_support and not live_sufficient:
        overall = "curated_support_only_live_data_insufficient"
    elif live_negative:
        overall = "live_proxy_does_not_support_area_constraint"
    else:
        overall = "inconclusive"

    report = structured_report(
        "EL1", "ok",
        verdict=overall,
        scientific_verdict=(
            "EL1 is not robustly confirmed by independent live public scrape. "
            "The curated note dataset supports area-like/sublinear scaling, but live vendor data are sparse and heterogeneous. "
            "Strict grouping prevents mixed-unit regression from being misread as a physics test."
        ),
        curated_note_analysis=curated_analysis,
        live_vendor_recovered_points=recovered,
        live_vendor_failed_sources=failures,
        live_scrape_complete=not any(f.get("label") == "live_scrape_aborted" for f in failures),
        live_strict_same_physics_analyses=strict_live_analyses,
        live_mixed_audit=mixed_audit,
        required_for_decisive_test=[
            "same product class or raw NAND die family",
            "same metric definition, preferably sequential read MB/s or NAND interface Gbps only",
            "same interface/controller generation or explicit normalization by interface bandwidth",
            "at least 4 independent layer-count points in one strict group",
        ],
    )
    json_dump(report, OUT / "el1_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
